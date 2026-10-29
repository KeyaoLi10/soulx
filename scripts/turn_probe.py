"""Direct turn-server protocol probe: stream an utterance, print states.

Bypasses the web app entirely — used to prove the turn server itself
is (or is not) healthy while a web session is wedged.
Usage: venv/bin/python -I scripts/turn_probe.py [seconds_of_audio]
"""
import base64
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np
import websocket

SR = 16000
CHUNK = 2560
INPUT = Path(
    "/home/research/krisl/soulx/data/eval_inputs/eotbench_zh/zh__2284/input.wav"
)
dur = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0

with wave.open(str(INPUT), "rb") as w:
    wav = (
        np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        .astype(np.float32)
        / 32768.0
    )
wav = wav[: int(dur * SR)]

sid = f"probe-{int(time.time())}"
ws = websocket.create_connection("ws://127.0.0.1:8000/turn", timeout=2.0)
t0 = time.time()
counts = {}
for i in range(0, len(wav), CHUNK):
    payload = {
        "type": "audio",
        "session_id": sid,
        "audio": base64.b64encode(
            wav[i : i + CHUNK].astype(np.float32).tobytes()
        ).decode(),
    }
    ws.send(json.dumps(payload))
    try:
        resp = json.loads(ws.recv())
    except Exception as e:
        print(f"[probe] recv error after {time.time()-t0:.1f}s: {e}")
        break
    st = resp["state"]["state"]
    counts[st] = counts.get(st, 0) + 1
    if st == "speak":
        print(
            f"[probe] speak at {time.time()-t0:.1f}s: "
            f"{resp['state'].get('text', '')}"
        )
    time.sleep(max(0.0, 0.16 - (time.time() - t0 - 0.16 * (i // CHUNK + 1))))
ws.close()
print(f"[probe] done in {time.time()-t0:.1f}s, states: {counts}")
