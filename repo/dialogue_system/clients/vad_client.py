import time
import uuid
import base64

import json
import websocket
import numpy as np


class TurnTaking:
    def __init__(
        self,
        status_callback=None,
        transcription_callback=None,
        circle_callback=None,
        server_url="ws://localhost:8000/turn",
        client_id=None,
        timeout=1.0,
    ):
        self.client_id = client_id or uuid.uuid4().hex
        self.timeout = timeout
        self.status_callback = status_callback
        self.transcription_callback = transcription_callback
        self.circle_callback = circle_callback
        self.server_url = server_url
        self.ws = None
        self._last_status = None
        self._last_circle_status = None
        self._last_transcription = None
        # What ended the last turn: "complete" (the model predicted
        # <|user_complete|> and the reply fired instantly, no silence
        # confirmation) or "wait" (silence-verified after an incomplete).
        # Read by app.py at turn-launch time to decide whether the reply
        # needs a premature-turn grace window. Only written on "speak".
        self.last_turn_trigger = None
        # Per-session wait-path override (chunks), pushed to the turn server
        # as a config message after EVERY (re)connect: the server GCs idle
        # sessions after 60s, so a one-shot push on set would silently
        # expire. None = server/model default.
        self.max_wait_num = None
        # Two-pass ASR (2026-11-05 方案A): commit_id of the last "speak"
        # response (echoed by the server's async turn_final push), and the
        # stash of received turn_final texts keyed by commit_id. Written by
        # the recv loop / process(); read by wait_turn_final() from the
        # app's pipeline worker thread.
        self.last_commit_id = None
        self._turn_finals = {}

    def _emit_status(self, state, message, force=False):
        if not self.status_callback:
            return
        payload = (state, message)
        if not force and self._last_status == payload:
            return
        self._last_status = payload
        try:
            self.status_callback(state, message)
        except Exception as exc:
            print(f"[VadClient] status callback failed: {exc}")

    def _emit_circle_status(self, status, force=False):
        if not self.circle_callback:
            return
        if not force and self._last_circle_status == status:
            return
        self._last_circle_status = status
        try:
            self.circle_callback(status)
        except Exception as exc:
            print(f"[VadClient] circle callback failed: {exc}")

    def _emit_transcription(self, text, force=False):
        if not self.transcription_callback:
            return
        cleaned = text.strip()
        if not cleaned and not force:
            return
        if not force and self._last_transcription == cleaned:
            return
        self._last_transcription = cleaned
        try:
            self.transcription_callback(cleaned)
        except Exception as exc:
            print(f"[VadClient] transcription callback failed: {exc}")

    def connect(self):
        # Hard timeout: without it a stuck turn server blocks the web app's
        # event loop forever inside create_connection.
        self.ws = websocket.create_connection(self.server_url, timeout=5.0)
        self.ws.settimeout(self.timeout)
        # Re-push session config: the server may have GC'd this session
        # (60s idle) since the last connection, dropping our override.
        if self.max_wait_num is not None:
            try:
                self._send_config()
            except Exception as exc:
                # Non-fatal: worst case the server uses its default until
                # the next successful push.
                print(
                    f"[VadClient] max_wait_num re-push failed "
                    f"({type(exc).__name__}: {exc})",
                    flush=True,
                )

    def _send_config(self):
        # Unsolicited control message — the server never replies to it, so
        # this must NOT go through process()'s send/recv pairing.
        self.ws.send(
            json.dumps(
                {
                    "type": "config",
                    "session_id": self.client_id,
                    "max_wait_num": self.max_wait_num,
                }
            )
        )

    def set_max_wait_num(self, n):
        """Override wait-path silence confirmation (chunks). None = default.

        Stores the value and pushes it to the turn server immediately (if
        connected); connect() re-pushes after every reconnect so the
        override survives server-side session GC.
        """
        self.max_wait_num = n
        if self.ws is not None and n is not None:
            try:
                self._send_config()
            except Exception as exc:
                print(
                    f"[VadClient] set_max_wait_num push failed "
                    f"({type(exc).__name__}: {exc}); will re-push on next connect",
                    flush=True,
                )

    def reset(self, client_id=None):
        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass
        self.client_id = client_id or uuid.uuid4().hex
        self.ws = None
        try:
            self.connect()
        except Exception as exc:
            # Leave ws=None: process() reconnects lazily on first use, so
            # this instance can still return to the pool instead of leaking.
            print(
                f"[VadClient] reset reconnect failed ({type(exc).__name__}: {exc}); "
                f"will retry on next process()",
                flush=True,
            )

    def _send_and_recv(self, payload_str):
        """Send one audio payload, recv until the paired turn_state.

        The turn server pushes unsolicited turn_final messages (two-pass
        ASR correction) after a commit — they can land ahead of any later
        chunk's response. Stash them and keep reading; only the paired
        turn_state terminates the loop. Raises on socket errors, exactly
        like the bare send/recv it replaced."""
        self.ws.send(payload_str)
        while True:
            data = json.loads(self.ws.recv())
            if isinstance(data, dict) and data.get("type") == "turn_final":
                self._stash_turn_final(data)
                continue
            return data

    def _stash_turn_final(self, data):
        cid = data.get("commit_id")
        if cid is None:
            return
        # bounded: only the last few commits' finals are ever needed (the
        # app waits at most ~0.5s after each commit)
        self._turn_finals[cid] = (data.get("text") or "").strip()
        while len(self._turn_finals) > 8:
            self._turn_finals.pop(next(iter(self._turn_finals)))

    def wait_turn_final(self, commit_id, timeout=0.5, poll=0.02):
        """Block up to `timeout` for the turn server's corrected whole-turn
        text (two-pass ASR). Runs in the app's pipeline worker thread —
        never call from the event loop. Returns the corrected text, or
        None when nothing usable arrived in time (caller falls back to the
        live text). An empty correction also returns None — the live text
        is kept."""
        if commit_id is None:
            return None
        deadline = time.time() + timeout
        while time.time() < deadline:
            text = self._turn_finals.pop(commit_id, None)
            if text is not None:
                return text or None
            time.sleep(poll)
        return None

    def process(self, audio_chunk: np.ndarray):
        if audio_chunk is None:
            return None
        t0 = time.time()
        if self.ws is None:
            # First use after pool init, or a reset whose reconnect failed
            self.connect()
        payload = {
            "type": "audio",
            "session_id": self.client_id,
            "audio": base64.b64encode(
                np.asarray(audio_chunk, dtype=np.float32).tobytes()
            ).decode(),
        }
        try:
            data = self._send_and_recv(json.dumps(payload))
        except Exception as exc:
            # Never swallow the reason silently: log it, it is the key
            # diagnostic when the turn pipeline degrades.
            print(
                f"[VadClient] recv/send failed after {time.time()-t0:.2f}s "
                f"({type(exc).__name__}: {exc}); reconnecting and resending",
                flush=True,
            )
            self.connect()
            data = self._send_and_recv(json.dumps(payload))
        if time.time() - t0 > 0.5:
            print(
                f"[VadClient] slow process(): {time.time()-t0:.2f}s "
                f"(session {self.client_id[:8]})",
                flush=True,
            )

        assert isinstance(data, dict), ValueError(
            f"Unexpected response type: {type(data)}"
        )

        if data["state"]["state"] == "idle":
            self._emit_status("silence", "Silence detected")
            return None
        elif data["state"]["state"] == "nonidle":
            self._emit_status("listening", "Voice detected")
            self._emit_circle_status("LISTENING")
            asr_buffer = data["state"].get("asr_buffer", "")
            if asr_buffer.strip():
                self._emit_transcription(asr_buffer)
            return [None]
        elif data["state"]["state"] == "speak":
            self._emit_status("taking turn", "User finished, taking turn")
            asr_buffer = data["state"].get("asr_buffer", "")
            if asr_buffer.strip():
                self._emit_transcription(asr_buffer)
            self.last_turn_trigger = data["state"].get("trigger") or "wait"
            self.last_commit_id = data["state"].get("commit_id")
            return data["state"].get("text", "")
