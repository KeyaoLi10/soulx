"""WS test for the live settings panel (update_config / config_applied).

Single reader thread (started right after connect) collects all events into
a list; the main thread only sends. Sharing recv across phases with a stale
socket timeout corrupts websocket-client's frame state — don't do that.

1. connect, confirm handshake
2. persona + TTS params update -> config_applied with merged values
3. hostile values (wrong types, out-of-range) -> clamped/ignored, no crash
4. one short speech round -> turn_start + text_response + audio_chunk
   (proves the settings snapshot flows into the live pipeline)

Usage: venv/bin/python -I scripts/config_ws_test.py
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
URL = "ws://127.0.0.1:8090/ws"
INPUT = Path(
    "/home/research/krisl/soulx/data/eval_inputs/eotbench_zh/zh__2284/input.wav"
)

with wave.open(str(INPUT), "rb") as w:
    WAV = (
        np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32)
        / 32768.0
    )

events = []          # [(event_name, data_or_nbytes)]
events_lock = threading.Lock()
failures = []


def log(tag, text):
    print(f"{time.strftime('%H:%M:%S')} [{tag}] {text}", flush=True)


def check(name, cond, detail=""):
    if cond:
        log("PASS", name)
    else:
        failures.append(name)
        log("FAIL", f"{name} {detail}")


def reader(ws):
    ws.settimeout(120)
    try:
        while True:
            raw = ws.recv()
            if isinstance(raw, bytes):
                with events_lock:
                    events.append(("audio_chunk", len(raw)))
                continue
            msg = json.loads(raw)
            with events_lock:
                events.append((msg.get("event"), msg.get("data")))
    except Exception:
        pass


def wait_for(name, timeout=30, after=0):
    """Wait for event `name` at index >= after (returns latest data)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        with events_lock:
            for idx in range(after, len(events)):
                if events[idx][0] == name:
                    return events[idx][1]
        time.sleep(0.2)
    return None


def applied_settings():
    # only consider config_applied echoes newer than what we've seen,
    # otherwise the first echo is returned for every update
    with events_lock:
        mark = len(events)
    data = wait_for("config_applied", timeout=10, after=mark)
    return (data or {}).get("settings", {})


def send_update(ws, data):
    """Send an update_config and return the freshly echoed settings."""
    with events_lock:
        mark = len(events)
    ws.send(json.dumps({"event": "update_config", "data": data}))
    got = wait_for("config_applied", timeout=10, after=mark)
    return (got or {}).get("settings", {})


ws = websocket.create_connection(URL, timeout=10)
threading.Thread(target=reader, args=(ws,), daemon=True).start()

check("connect_ack", wait_for("connect_ack", timeout=30) is not None)
wait_for("vad_loading", timeout=60)  # last one is the ready state
log("INFO", "session ready")

# --- 1. normal update ---
s = send_update(ws, {
    "system_prompt": "You are a pirate captain. Answer in one short sentence.",
    "max_history": 12,
    "min_seg_len": 6,
    "tts_engine": "voxcpm",
    "tts_cfg": 3.5,
    "tts_timesteps": 20,
    "turn_wait_ms": 1600,
    "turn_grace_ms": 1500,
})
check("config_applied persona", s.get("system_prompt") == "You are a pirate captain. Answer in one short sentence.")
check("config_applied max_history=12", s.get("max_history") == 12)
check("config_applied min_seg_len=6", s.get("min_seg_len") == 6)
check("config_applied cfg=3.5", s.get("tts_cfg") == 3.5)
check("config_applied steps=20", s.get("tts_timesteps") == 20)
check("config_applied turn_wait_ms=1600", s.get("turn_wait_ms") == 1600)
check("config_applied turn_grace_ms=1500", s.get("turn_grace_ms") == 1500)
check("removed key tts_voice_desc ignored", "tts_voice_desc" not in s)
check("config_applied engine", s.get("tts_engine") == "voxcpm")

# turn_wait_ms=1600 must reach the turn server as max_wait_num=10 chunks.
# The web app pushes it via the config message, BUT the pooled TurnTaking
# client only opens its turn-server socket on the first process() call —
# before any audio, set_max_wait_num() just stores the value. So: feed a
# second of audio to trigger the connect (connect() re-pushes the stored
# override), then check the turn server log.
ws.send(json.dumps({"event": "config_audio", "data": {"sample_rate": SR}}))
time.sleep(0.3)
for i in range(0, SR, CHUNK):  # 1s of speech
    payload = (np.clip(WAV[i:i + CHUNK], -1, 1) * 32767).astype("<i2").tobytes()
    try:
        ws.send(payload, opcode=websocket.ABNF.OPCODE_BINARY)
    except Exception as e:
        log("FAIL", f"send broke: {e}")
        break
    time.sleep(0.16)
time.sleep(1.0)
log_tail = subprocess.run(
    ["grep", "max_wait_num", "/home/research/krisl/soulx/logs/turn.log"],
    capture_output=True, text=True,
).stdout
check("turn server got max_wait_num=10",
      "max_wait_num -> 10" in log_tail,
      f"(grep: {log_tail[-200:]!r})")

# --- 2. hostile update: clamp + ignore bad types ---
s2 = send_update(ws, {
    "max_history": 999,          # clamp to 20
    "min_seg_len": -5,           # clamp to 4
    "tts_cfg": "not-a-number",   # ignored, keeps 3.5
    "tts_timesteps": 1e9,        # clamp to 40
    "system_prompt": 12345,      # wrong type, ignored
    "tts_engine": "voiceclone",  # invalid, ignored
    "turn_wait_ms": 99999,       # clamp to 3200
    "turn_grace_ms": -100,       # clamp to 0 (window disabled)
    "evil_key": "x",             # unknown, ignored
})
check("clamp max_history=20", s2.get("max_history") == 20)
check("clamp min_seg_len=4", s2.get("min_seg_len") == 4)
check("bad cfg ignored (3.5)", s2.get("tts_cfg") == 3.5)
check("clamp timesteps=40", s2.get("tts_timesteps") == 40)
check("bad persona ignored", s2.get("system_prompt") == "You are a pirate captain. Answer in one short sentence.")
check("bad engine ignored", s2.get("tts_engine") == "voxcpm")
check("clamp turn_wait_ms=3200", s2.get("turn_wait_ms") == 3200)
check("clamp turn_grace_ms=0", s2.get("turn_grace_ms") == 0)
check("unknown key ignored", "evil_key" not in s2)

# --- 2b. restore sane timing before the live round (1600ms wait would
# slow the wait-path trigger; grace 0 would disable the hold window) ---
send_update(ws, {"turn_wait_ms": 800, "turn_grace_ms": 900})

# --- 3. one real round with the custom persona/params ---
with events_lock:
    events.clear()
ws.send(json.dumps({"event": "config_audio", "data": {"sample_rate": SR}}))
time.sleep(0.3)

t0 = time.time()
for i in range(0, 6 * SR, CHUNK):
    payload = (np.clip(WAV[i:i + CHUNK], -1, 1) * 32767).astype("<i2").tobytes()
    try:
        ws.send(payload, opcode=websocket.ABNF.OPCODE_BINARY)
    except Exception as e:
        log("FAIL", f"send broke: {e}")
        break
    time.sleep(0.16)
# silence tail: without it the trailing speech in the clip reads as a
# barge-in and legitimately interrupts the reply before its first chunk
for _ in range(int(6 * SR / CHUNK)):
    try:
        ws.send(np.zeros(CHUNK, dtype="<i2").tobytes(),
                opcode=websocket.ABNF.OPCODE_BINARY)
    except Exception as e:
        log("FAIL", f"send broke: {e}")
        break
    time.sleep(0.16)

deadline = time.time() + 60
while time.time() < deadline:
    with events_lock:
        names = {ev for ev, _ in events}
    if {"turn_start", "text_response", "audio_chunk"} <= names:
        break
    time.sleep(0.5)

with events_lock:
    names = {ev for ev, _ in events}
    n_audio = sum(1 for ev, _ in events if ev == "audio_chunk")
check("turn_start received", "turn_start" in names)
check("text_response received", "text_response" in names)
check("audio_chunk received", "audio_chunk" in names, f"(n={n_audio})")
log("INFO", f"round finished in {time.time()-t0:.1f}s, audio chunks: {n_audio}")

try:
    ws.close()
except Exception:
    pass
if failures:
    log("RESULT", f"FAIL ({len(failures)}): {failures}")
    raise SystemExit(1)
log("RESULT", "PASS — all config tests + live round OK")
