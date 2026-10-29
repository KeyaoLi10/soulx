"""Reconnect test: mimic the frontend's auto-reconnect + heartbeat logic.

Streams real speech, then kills the server-side socket mid-stream (restart of
the web app). A client implementing the same policy as frontend/script.js
(connect + exponential backoff + app-level ping with staleness timeout) must
reconnect and transcription must resume — proving the fix for the "page goes
silent forever after a dropped WS" bug.

Usage: venv/bin/python -I scripts/reconnect_test.py
"""
import json
import subprocess
import threading
import time
import wave
from pathlib import Path

import numpy as np
import websocket

SR = 16000
CHUNK = 2560
HEARTBEAT_S = 8.0
STALE_S = 25.0
URL = "ws://127.0.0.1:8090/ws"
INPUT = Path(
    "/home/research/krisl/soulx/data/eval_inputs/eotbench_zh/zh__2284/input.wav"
)

with wave.open(str(INPUT), "rb") as w:
    WAV = (
        np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32)
        / 32768.0
    )

state = {
    "ws": None,          # current live socket (None while disconnected)
    "last_server_msg": time.time(),
    "asr_count": 0,
    "last_asr": "",
}
lock = threading.Lock()
stop = threading.Event()


def log(tag, text):
    print(f"{time.strftime('%H:%M:%S')} [{tag}] {text}", flush=True)


def on_message(raw):
    if isinstance(raw, bytes):
        return
    try:
        msg = json.loads(raw)
    except Exception:
        return
    if msg.get("event") == "user_transcription":
        state["asr_count"] += 1
        state["last_asr"] = msg["data"].get("text", "")


def spawn_reader(ws):
    """One reader thread per connection; clears state['ws'] when it dies."""

    def rd():
        try:
            while True:
                raw = ws.recv()
                with lock:
                    state["last_server_msg"] = time.time()
                on_message(raw)
        except Exception:
            pass
        finally:
            with lock:
                if state["ws"] is ws:
                    state["ws"] = None
            log("DISCONN", "socket died")

    threading.Thread(target=rd, daemon=True).start()


def connect_loop():
    """Reconnect policy identical to frontend/script.js: only ever one live
    socket, exponential backoff while the server is unreachable."""
    attempt = 0
    n_connected = 0
    while not stop.is_set():
        if state["ws"] is not None:
            time.sleep(0.2)  # still alive — nothing to do
            continue
        try:
            ws = websocket.create_connection(URL, timeout=5)
            ws.settimeout(120)
        except Exception:
            delay = min(5.0, 0.5 * 2**attempt)
            attempt += 1
            time.sleep(delay)
            continue
        attempt = 0
        n_connected += 1
        with lock:
            state["ws"] = ws
            state["last_server_msg"] = time.time()
        log("CONN", f"connected (#{n_connected})")
        spawn_reader(ws)


def heartbeat():
    """Same policy as the frontend: ping every 8s, force-close if stale."""
    while not stop.is_set():
        time.sleep(HEARTBEAT_S)
        ws = state["ws"]
        if ws is not None:
            try:
                ws.send(json.dumps({"event": "ping"}))
            except Exception:
                pass
        if time.time() - state["last_server_msg"] > STALE_S:
            log("HEARTBEAT", f"no server traffic for {STALE_S:.0f}s, forcing close")
            try:
                ws.close()
            except Exception:
                pass


def send(samples):
    payload = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    ws = state["ws"]
    if ws is not None:
        try:
            ws.send(payload, opcode=websocket.ABNF.OPCODE_BINARY)
        except Exception:
            pass


def wait_connected(timeout):
    t0 = time.time()
    while state["ws"] is None and time.time() - t0 < timeout:
        time.sleep(0.2)
    return state["ws"] is not None


def stream_speech(seconds):
    for i in range(0, seconds * SR, CHUNK):
        send(WAV[i : i + CHUNK])
        time.sleep(0.16)


def main():
    threading.Thread(target=connect_loop, daemon=True).start()
    threading.Thread(target=heartbeat, daemon=True).start()

    assert wait_connected(10), "no initial connection"
    state["ws"].send(json.dumps({"event": "config_audio", "data": {"sample_rate": SR}}))

    # round 1: 8s of speech
    stream_speech(8)
    time.sleep(2)
    n_before = state["asr_count"]
    log("PHASE1", f"asr events: {n_before} (last: {state['last_asr']})")

    # kill the server-side socket: restart the web app process mid-session
    pid = subprocess.check_output(["pgrep", "-f", "venv/bin/python app.py"]).split()[0]
    log("KILL", f"restarting web app (pid {pid.decode()}) mid-session")
    subprocess.run(["tmux", "send-keys", "-t", "soulx:3", "C-c"])
    time.sleep(1)
    subprocess.run(["kill", pid.decode()])
    subprocess.run(
        [
            "tmux",
            "send-keys",
            "-t",
            "soulx:3",
            "cd /home/research/krisl/soulx/repo/dialogue_system && "
            "/home/research/krisl/soulx/venv/bin/python app.py 2>&1 | "
            "tee -a /home/research/krisl/soulx/logs/web.log",
        ]
    )
    subprocess.run(["tmux", "send-keys", "-t", "soulx:3", "Enter"])

    # reconnect (app takes ~15-20s to load), then re-announce audio config
    # exactly like the frontend onopen handler
    assert wait_connected(90), "never reconnected"
    state["ws"].send(json.dumps({"event": "config_audio", "data": {"sample_rate": SR}}))
    time.sleep(1)

    # round 2: same speech again
    stream_speech(8)
    time.sleep(2)

    n_after = state["asr_count"] - n_before
    log("PHASE2", f"asr events after reconnect: {n_after} (last: {state['last_asr']})")
    stop.set()
    if n_before > 0 and n_after > 0:
        log("RESULT", "PASS — transcription resumed after server restart")
    else:
        log("RESULT", f"FAIL (before={n_before} after={n_after})")


if __name__ == "__main__":
    main()
