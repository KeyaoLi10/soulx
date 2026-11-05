import os
import base64
import faulthandler
import json
import signal
import threading
import time
import asyncio
from typing import Dict

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager

from service.engine import TurnTakingEngine
from service.session import TurnSession
from service.model import load_turn_model

# kill -USR1 <pid> dumps all thread tracebacks (goes to turn.log)
faulthandler.register(signal.SIGUSR1)

SESSION_TTL_SEC = 60  # Recycle if no audio for 60 seconds
GC_INTERVAL_SEC = 10  # Session GC interval

# Global state
sessions: Dict[str, TurnSession] = {}
# Config overrides (e.g. max_wait_num) keyed by session_id. Kept SEPARATE
# from sessions: a session idle >SESSION_TTL_SEC is GC'd, but the client
# keeps its WS connection (e.g. the web app ran in frc mode for a while,
# where no audio reaches this server) — a per-session-only override would
# be silently lost when the session is recreated from later audio. Entries
# expire only once their session is gone AND the client stopped pushing.
session_config: Dict[str, dict] = {}


# FastAPI Lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[TurnTaking] loading model ...")
    app.state.model = load_turn_model(
        config_path=os.path.join(os.path.dirname(__file__), "config/config.yaml")
    )
    print("[TurnTaking] model loaded")

    gc_task = asyncio.create_task(session_gc_loop())

    yield

    gc_task.cancel()
    print("[TurnTaking] shutdown")


app = FastAPI(lifespan=lifespan)


def _correct_and_push(ws, loop, model, session_id, commit_id, audio, live_text, t_commit):
    """Two-pass ASR correction (方案A, 2026-11-05) — runs on a daemon
    thread so the websocket loop is never blocked by the re-recognition.

    Re-recognizes the whole turn with the already-loaded paraformer and
    pushes an unsolicited turn_final message; the client stashes it (its
    recv loop reads until turn_state) and the web app's pipeline worker
    waits briefly for it, updating the bubble and the LLM input when the
    text differs from the live one. text="" (recognition unusable) is
    still pushed — an explicit "no correction" unblocks the app's wait
    immediately instead of burning the full timeout."""
    text = model.correct_turn_text(audio)
    lat = time.time() - t_commit
    changed = bool(text) and text != live_text
    print(
        f"[TurnTaking] FINAL session {session_id[:8]} id={commit_id} "
        f"lat={lat:.2f}s changed={int(changed)} text={text or live_text}",
        flush=True,
    )

    async def _push():
        try:
            await ws.send_text(
                json.dumps(
                    {
                        "type": "turn_final",
                        "session_id": session_id,
                        "commit_id": commit_id,
                        "text": text,
                        "ts": time.time(),
                    },
                    ensure_ascii=False,
                )
            )
        except Exception:
            # client disconnected mid-correction — nothing to deliver to
            pass

    try:
        loop.call_soon_threadsafe(lambda: asyncio.ensure_future(_push()))
    except RuntimeError:
        # event loop already shut down (server stopping)
        pass


async def session_gc_loop():
    while True:
        now = time.time()
        expired = []

        for sid, sess in sessions.items():
            if now - sess.last_active_ts > SESSION_TTL_SEC:
                expired.append(sid)

        for sid in expired:
            print(f"[TurnTaking] GC session {sid}")
            del sessions[sid]

        # config overrides outlive their session (see session_config note),
        # but not forever: drop them once the session is gone and the
        # client hasn't re-pushed within the same TTL.
        stale_cfg = [
            sid
            for sid, cfg in session_config.items()
            if sid not in sessions and now - cfg["ts"] > SESSION_TTL_SEC
        ]
        for sid in stale_cfg:
            del session_config[sid]

        await asyncio.sleep(GC_INTERVAL_SEC)


@app.websocket("/turn")
async def turn_ws(ws: WebSocket):
    await ws.accept()
    print("[TurnTaking] websocket connected")

    try:
        while True:
            raw = await ws.receive_text()
            data = json.loads(raw)

            if data.get("type") != "audio":
                # {"type":"config","session_id":...,"max_wait_num":N}
                # — unsolicited control message, NO reply is sent (the
                # client's recv loop expects exactly one response per audio
                # chunk and never reads for control messages).
                if data.get("type") == "config" and "session_id" in data:
                    session_id = data["session_id"]
                    if session_id not in sessions:
                        engine = TurnTakingEngine(model=ws.app.state.model)
                        sessions[session_id] = TurnSession(engine)
                        print(f"[TurnTaking] new session {session_id} (config)")
                    session = sessions[session_id]
                    session.touch()
                    if "max_wait_num" in data:
                        try:
                            n = int(data["max_wait_num"])
                        except (TypeError, ValueError):
                            n = None
                        # clamp: 1..50 chunks (160ms each) — 0 would fire
                        # "wait" on the first idle chunk, >50 is unusable
                        n = n if n is not None and 1 <= n <= 50 else None
                        session.set_max_wait_num(n)
                        # persist beyond session GC (see session_config note)
                        session_config[session_id] = {
                            "max_wait_num": n,
                            "ts": time.time(),
                        }
                        print(
                            f"[TurnTaking] session {session_id} "
                            f"max_wait_num -> {n if n is not None else 'default'}",
                            flush=True,
                        )
                continue

            session_id = data["session_id"]

            if session_id not in sessions:
                engine = TurnTakingEngine(model=ws.app.state.model)
                sessions[session_id] = TurnSession(engine)
                print(f"[TurnTaking] new session {session_id}")
                # re-apply any surviving config override (session was GC'd
                # and is being recreated from fresh audio)
                cfg = session_config.get(session_id)
                if cfg is not None and cfg.get("max_wait_num") is not None:
                    sessions[session_id].set_max_wait_num(cfg["max_wait_num"])

            session = sessions[session_id]
            session.touch()  # Update timestamp

            try:
                audio = np.frombuffer(base64.b64decode(data["audio"]), dtype=np.float32)
            except Exception:
                continue

            t_feed = time.time()
            state = session.feed_audio(audio)
            dt_feed = time.time() - t_feed
            if dt_feed > 0.5:
                print(
                    f"[TurnTaking] SLOW feed_audio {dt_feed:.2f}s "
                    f"session {session_id}",
                    flush=True,
                )

            if state is not None:
                if state.get("gate"):
                    # blip gate fired on this chunk (prob / rms /
                    # mistake_no_text / commit_no_text) — one line per gated
                    # chunk, rare by construction. flush: stdout is piped
                    # through tee.
                    print(
                        f"[TurnTaking] GATE session {session_id[:8]} "
                        f"{state['gate']}",
                        flush=True,
                    )
                if state.get("state_log"):
                    # state-token transition (idle/nonidle/incomplete/
                    # complete/...) — one line per CHANGE, so a turn
                    # boundary is readable in turn.log without
                    # developer_mode. Same flush discipline as GATE.
                    sl = state["state_log"]
                    print(
                        f"[TurnTaking] STATE session {session_id[:8]} "
                        f"{sl['from']} -> {sl['to']}",
                        flush=True,
                    )
                if state.get("commit"):
                    # the turn committed on this chunk: which path fired
                    # (complete = instant <|user_complete|>, wait =
                    # silence-verified after <|user_incomplete|>) and the
                    # recognized text
                    c = state["commit"]
                    print(
                        f"[TurnTaking] TURN session {session_id[:8]} "
                        f"commit trigger={c.get('trigger')} "
                        f"text={c.get('text', '')}",
                        flush=True,
                    )
                # two-pass ASR (方案A): the engine attached the whole-turn
                # audio to the commit — pop it BEFORE json.dumps (a numpy
                # array is not serializable), stamp the commit_id the
                # client will pair the turn_final against, and spawn the
                # correction AFTER the turn_state below goes out (the
                # client's recv for THIS chunk must be answered first).
                corr_audio = state.pop("corr_audio", None)
                if corr_audio is not None:
                    session.commit_seq += 1
                    state["commit_id"] = session.commit_seq
                await ws.send_text(
                    json.dumps(
                        {
                            "type": "turn_state",
                            "session_id": session_id,
                            "state": state,
                            "ts": time.time(),
                        },
                        ensure_ascii=False,
                    )
                )
                if corr_audio is not None:
                    threading.Thread(
                        target=_correct_and_push,
                        args=(
                            ws,
                            asyncio.get_running_loop(),
                            ws.app.state.model,
                            session_id,
                            session.commit_seq,
                            corr_audio,
                            (state.get("commit") or {}).get("text", ""),
                            time.time(),
                        ),
                        daemon=True,
                    ).start()

    except WebSocketDisconnect:
        print("[TurnTaking] websocket disconnected", flush=True)

    except Exception as e:
        # flush is load-bearing: stdout is piped through tee (block-buffered)
        # — an unflushed exception here is invisible in turn.log
        print(f"[TurnTaking] websocket error: {e}", flush=True)
        import traceback

        traceback.print_exc()
