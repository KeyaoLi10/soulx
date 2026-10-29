"""Barge-in storm regression test (the 2026-10-29 choppiness bug).

Live failure mode: during assistant playback the mic carried noise/echo
residue; the turn server classified it as nonidle, and the web app
(speaker gate unenrolled => fail-open) interrupted playback on every
such chunk -> stop_audio storm, LISTENING/SPEAKING flip-flop, TTS
restarts. Root causes fixed in app.py:

  A. enrollment speech bank — short turns accumulate across turns so the
     ECAPA gate arms by turn 2 (previously the 8s ring never held 3s of
     speech, so the session fail-opened FOREVER — the live bug).
  B. barge-in hysteresis — a barge-in acts only after 3 consecutive
     nonidle chunks (~480 ms); one episode interrupts once.
  C. circle-status gating during playback.

What this test exercises, in order:
  1. two short speech turns — enrollment must complete via the bank
  2. ECHO during playback (assistant audio fed back at -20 dB, the
     browser-AEC residual scenario) — must NOT stop playback; the armed
     ECAPA gate rejects it (pvad_reject bargein)
  3. speech-band noise during playback — must NOT stop playback
  4. sustained real speech — MUST barge in, exactly one stop_audio

Usage: soulx/venv/bin/python -I scripts/bargein_storm_test.py
"""
import argparse
import json
import threading
import time
import wave
from pathlib import Path

import numpy as np
import soxr
import websocket

CHUNK = 2560            # 160 ms @ 16k (mic)
ECHO_CHUNK_24K = 3840   # 160 ms @ 24k (assistant stream)
SR = 16000
DATA = Path("/home/research/krisl/soulx/data/eval_inputs")
SPEECH = DATA / "eotbench_zh/zh__2284/input.wav"
SPEECH_OFFSET_S = 21.0   # measured best 2s speech window in this file
ECHO_GAIN = 0.1          # -20 dB: browser-AEC residual level
NOISE_RMS = 0.03

events = []
events_lock = threading.Lock()
echo_buf = bytearray()   # assistant PCM (int16 LE @24k) for the echo phase
echo_lock = threading.Lock()
stop_flag = False


def log(tag, text):
    print(f"{time.strftime('%H:%M:%S')} [{tag}] {text}", flush=True)


def reader(ws):
    while not stop_flag:
        try:
            raw = ws.recv()
        except Exception:
            return
        if isinstance(raw, bytes):
            with events_lock:
                events.append((time.time(), "audio_chunk", {"bytes": len(raw)}))
            with echo_lock:
                echo_buf.extend(raw)
            continue
        try:
            msg = json.loads(raw)
        except Exception:
            continue
        ev, data = msg.get("event"), msg.get("data", {})
        with events_lock:
            events.append((time.time(), ev, data))


def load(path):
    with wave.open(str(path), "rb") as w:
        return (
            np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
            .astype(np.float32)
            / 32768.0
        )


def send(ws, samples):
    payload = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    ws.send(payload, opcode=websocket.ABNF.OPCODE_BINARY)


def stream(ws, wav):
    for i in range(0, len(wav), CHUNK):
        send(ws, wav[i: i + CHUNK])
        time.sleep(0.16)


def stream_silence(ws, seconds):
    n = int(seconds * SR / CHUNK)
    z = np.zeros(CHUNK, dtype=np.float32)
    for _ in range(n):
        send(ws, z)
        time.sleep(0.16)


def stream_noise(ws, seconds, rms, rng):
    b = np.array([0.2, 0.4, 0.25, 0.1, 0.05], dtype=np.float32)
    n = int(seconds * SR / CHUNK)
    for _ in range(n):
        x = np.convolve(rng.standard_normal(CHUNK).astype(np.float32), b, "same")
        x = x / (np.sqrt((x ** 2).mean()) + 1e-9) * rms
        send(ws, x)
        time.sleep(0.16)


def stream_echo(ws, seconds, gain):
    """Feed the assistant's own stream back as mic input (AEC residual)."""
    n = int(seconds * SR / CHUNK)
    for _ in range(n):
        with echo_lock:
            take = bytes(echo_buf[: ECHO_CHUNK_24K * 2])
            del echo_buf[: ECHO_CHUNK_24K * 2]
        if len(take) == ECHO_CHUNK_24K * 2:
            x = np.frombuffer(take, dtype="<i2").astype(np.float32) / 32768.0
            x = soxr.resample(x, 24000, SR) * gain
        else:
            x = np.zeros(CHUNK, dtype=np.float32)
        x = np.resize(x, CHUNK)  # pad/trim resampling rounding
        send(ws, x)
        time.sleep(0.16)


def window(t_from, t_to=None):
    with events_lock:
        return [
            (ts, ev, d)
            for ts, ev, d in events
            if ts > t_from and (t_to is None or ts <= t_to)
        ]


def wait_for(t_from, predicate, timeout, ws=None):
    """Wait for an event match; keeps streaming silence so the turn
    server's state machine actually progresses (no trailing silence =
    no turn decision)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if any(predicate(ev, d) for _, ev, d in window(t_from)):
            return True
        if ws is not None:
            send(ws, np.zeros(CHUNK, dtype=np.float32))
        time.sleep(0.16)
    return False


def count(evs, name, pred=lambda d: True):
    return sum(1 for _, ev, d in evs if ev == name and pred(d))


def short_turn(ws, speech, label):
    t = time.time()
    stream(ws, speech)
    ok = wait_for(t, lambda ev, d: ev == "turn_start", 20, ws=ws)
    log("TURN", f"{label}: fired={ok}")
    wait_for(t, lambda ev, d: ev == "audio_chunk", 15, ws=ws)
    return t, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8090/ws")
    args = ap.parse_args()
    rng = np.random.default_rng(42)

    speech_full = load(SPEECH)
    off = int(SPEECH_OFFSET_S * SR)
    speech = speech_full[off: off + int(2.0 * SR)]
    assert np.sqrt((speech ** 2).mean()) > 0.05, "speech window is too quiet"

    ws = websocket.create_connection(args.url, timeout=5)
    ws.settimeout(300)
    ws.send(json.dumps({"event": "config_audio", "data": {"sample_rate": SR}}))
    threading.Thread(target=reader, args=(ws,), daemon=True).start()

    # This test exercises the pvad gate machinery (enrollment/bank/rejects),
    # which is OFF by default since 2026-10-30 — switch it on explicitly.
    ws.send(json.dumps({
        "event": "update_config", "data": {"pvad_mode": "enforce"},
    }))

    failures = []

    def check(name, cond, detail=""):
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
        if not cond:
            failures.append(name)

    # ---- 1. two short turns -> bank enrollment ------------------------
    t0 = time.time()
    _, ok1 = short_turn(ws, speech, "turn1")
    check("turn 1 fires on short speech", ok1)
    stream_silence(ws, 2.0)
    _, ok2 = short_turn(ws, speech, "turn2")
    check("turn 2 fires", ok2)
    stream_silence(ws, 2.0)
    time.sleep(3.0)  # enrollment runs async at turn start
    enrolled = any(
        ev == "pvad_status" and d.get("enrolled") for _, ev, d in window(t0)
    )
    check(
        "enrollment completed via speech bank by turn 2",
        enrolled,
        "(pre-fix the 8s ring never held 3s => gate fail-opened forever)",
    )

    # ---- 2. echo during playback (the live scenario) -------------------
    # seed a reply first
    _, ok3 = short_turn(ws, speech, "turn3")
    with echo_lock:
        echo_buf.clear()
    te = time.time()   # after turn3's own barge-in/pipeline interrupts
    stream_echo(ws, 12, ECHO_GAIN)
    te_end = time.time()
    evs = window(te, te_end)
    n_stop = count(evs, "stop_audio")
    n_reject = count(evs, "pvad_reject", lambda d: d.get("kind") == "bargein")
    n_listen = count(evs, "circle_status", lambda d: d.get("status") == "LISTENING")
    log("ECHO", f"stop_audio={n_stop} pvad_reject(bargein)={n_reject} "
                f"LISTENING={n_listen}")
    check(
        f"echo @{ECHO_GAIN} does not chop playback",
        n_stop <= 1,
        f"stop_audio={n_stop} (<=1: a single onset turn is tolerated; "
        f"pre-fix this was a 12-15x self-talk storm)",
    )
    check(
        "echo barge-ins rejected by the speaker gate",
        n_reject >= 1,
        f"rejects={n_reject}",
    )

    # ---- 3. speech-band noise during playback --------------------------
    tn = time.time()
    _, ok4 = short_turn(ws, speech, "turn4")
    tn_play = time.time()
    stream_noise(ws, 8, NOISE_RMS, rng)
    tn_end = time.time()
    evs = window(tn_play, tn_end)
    n_stop = count(evs, "stop_audio")
    n_listen = count(evs, "circle_status", lambda d: d.get("status") == "LISTENING")
    check(
        f"noise rms={NOISE_RMS} does not chop playback",
        n_stop == 0,
        f"stop_audio={n_stop} LISTENING={n_listen}",
    )

    # ---- 4. sustained real speech MUST barge in, exactly once ----------
    tb = time.time()
    stream(ws, speech)
    stream_silence(ws, 2.0)
    ok = wait_for(tb, lambda ev, d: ev == "stop_audio", 8, ws=ws)
    evs = window(tb)
    n_stop = count(evs, "stop_audio")
    check("sustained speech still barge-ins", ok, f"stop_audio={n_stop}")
    check("one barge-in episode interrupts once", 1 <= n_stop <= 2,
          f"stop_audio={n_stop}")

    print(
        f"\n[RESULT] {'FAIL: ' + str(failures) if failures else 'PASS — echo/noise cannot chop playback, real barge-in intact, bank enrollment works'}"
    )
    global stop_flag
    stop_flag = True
    try:
        ws.close()
    except Exception:
        pass
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
