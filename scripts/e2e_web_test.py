"""E2E repro: simulate a browser session against the SoulX web app.

Connects to /ws on the web app, streams real user speech (16 kHz int16,
160 ms chunks) in multiple rapid rounds -- mimicking a human that keeps
talking and barge-in interrupting -- and records every websocket event
with timestamps. A stall is flagged when speech is actively being sent
but no transcription/user events arrive for WATCHDOG_S seconds.

Usage:
  venv/bin/python -I scripts/e2e_web_test.py [--rounds 6] [--url ws://127.0.0.1:8090/ws]
"""
import argparse
import json
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np
import websocket

CHUNK = 2560  # samples @16k = 160 ms, same as the frontend
SR = 16000
WATCHDOG_S = 12.0
INPUT = Path(
    "/home/research/krisl/soulx/data/eval_inputs/eotbench_zh/zh__2284/input.wav"
)

events = []
events_lock = threading.Lock()
last_user_event = time.time()  # last transcription/status/audio from server
speech_active = False
stop_flag = False


def reader(ws):
    global last_user_event
    while not stop_flag:
        try:
            raw = ws.recv()
        except Exception as e:
            log("EVENT", f"reader exit: {type(e).__name__}: {e}")
            return
        now = time.time()
        if isinstance(raw, bytes):
            with events_lock:
                events.append((now, "audio_chunk", len(raw)))
            last_user_event = now
            continue
        try:
            msg = json.loads(raw)
        except Exception:
            continue
        ev = msg.get("event")
        data = msg.get("data", {})
        with events_lock:
            events.append((now, ev, str(data)[:90]))
        if ev not in ("audio_chunk",):
            log("EVT", f"{ev}: {str(data)[:80]}")
        if ev in (
            "user_transcription",
            "text_response",
            "vad_status",
            "circle_status",
            "stop_audio",
        ):
            last_user_event = now
        if ev == "user_transcription":
            log("ASR", data.get("text", ""))


def log(tag, text):
    print(f"{time.strftime('%H:%M:%S')} [{tag}] {text}", flush=True)


_n_sends = 0


def send_audio(ws, samples):
    global _n_sends
    # samples are float32 in [-1, 1]: scale back to the int16 range before
    # packing. (astype alone truncates to 0 — every earlier run of this
    # script silently streamed pure silence and the "wedge" it reported was
    # the server correctly answering idle.)
    payload = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    _n_sends += 1
    if _n_sends <= 3:
        print(
            f"[SEND] #{_n_sends} len={len(payload)} "
            f"head={payload[:12].hex()} max={float(np.max(np.abs(samples))):.4f}",
            flush=True,
        )
    ws.send(payload, opcode=websocket.ABNF.OPCODE_BINARY)


def stream_round(ws, wav, label, tail_silence=3.0):
    """Stream one full utterance file at 1x realtime, then some silence."""
    global speech_active
    log("ROUND", f"start {label} ({len(wav)/SR:.1f}s speech)")
    speech_active = True
    for i in range(0, len(wav), CHUNK):
        send_audio(ws, wav[i : i + CHUNK])
        # realtime pacing: 160 ms per chunk
        target = 0.16 * (i // CHUNK + 1)
        sent = 0.16 * (i // CHUNK)
        time.sleep(max(0.0, min(0.16, target - sent)))
    speech_active = False
    # silence while the agent replies / user "listens"
    n_sil = int(tail_silence * SR / CHUNK)
    for _ in range(n_sil):
        send_audio(ws, np.zeros(CHUNK, dtype=np.float32))
        time.sleep(0.16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8090/ws")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--gap", type=float, default=1.0,
                    help="silence between rounds (s)")
    args = ap.parse_args()

    with wave.open(str(INPUT), "rb") as w:
        assert w.getframerate() == SR, f"expected 16k, got {w.getframerate()}"
        wav = (
            np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
            .astype(np.float32)
            / 32768.0
        )

    ws = websocket.create_connection(args.url, timeout=5)
    ws.settimeout(120)  # reader must survive long silences between events
    ws.send(json.dumps({"event": "config_audio", "data": {"sample_rate": SR}}))
    threading.Thread(target=reader, args=(ws,), daemon=True).start()
    log("CONN", "connected, sample rate sent")

    watchdog_last_report = time.time()

    def watchdog():
        while not stop_flag:
            time.sleep(1.0)
            idle = time.time() - last_user_event
            if idle > WATCHDOG_S and speech_active:
                log(
                    "STALL",
                    f"no server events for {idle:.0f}s while streaming speech "
                    "-- transcription pipeline appears wedged",
                )
            if time.time() - watchdog_last_report > 30:
                with events_lock:
                    n = len(events)
                log("ALIVE", f"{n} events so far")

    threading.Thread(target=watchdog, daemon=True).start()

    for r in range(args.rounds):
        # vary pacing: even rounds = rapid-fire (short gap, mimics barge-in),
        # odd rounds = normal with pause while agent talks
        gap = 0.4 if r % 2 == 0 else args.gap + 2.0
        stream_round(ws, wav, f"round {r+1}/{args.rounds}", tail_silence=gap)

    # final listen window for the last reply
    for _ in range(int(8 * SR / CHUNK)):
        send_audio(ws, np.zeros(CHUNK, dtype=np.float32))
        time.sleep(0.16)

    global stop_flag
    stop_flag = True
    time.sleep(0.5)
    with events_lock:
        counts = {}
        for _, ev, _d in events:
            counts[ev] = counts.get(ev, 0) + 1
    log("DONE", f"event counts: {json.dumps(counts, ensure_ascii=False)}")
    try:
        ws.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
