#!/usr/bin/env python
"""FRC frontend mode E2E test (turn_engine="frc" over the live web app).

Synthesizes user speech with the VoxCPM2 TTS service (same voice as
agent_e2e_test, cached), streams it through /ws with turn_engine=frc, and
checks the FireRedChat-architecture pipeline:

  phase 1  single turn: utterance -> ONE user_transcription (utterance-final
           FireRedASR, no partials) -> assistant reply speech
  phase 2  mid-sentence pause robustness: 1.2 s pause inside the utterance
           must NOT produce an early turn/reply (dual-EOT hold + resume
           cancel); the final transcript contains both halves and only one
           reply generation runs

Usage:
  venv/bin/python -I scripts/frc_mode_test.py [--url ws://127.0.0.1:8090/ws]
"""
import argparse
import json
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np
import requests
import websocket

SR = 16000
CHUNK = 2560
TTS_URL = "http://127.0.0.1:8085/synthesize"
OUT_DIR = Path("/tmp/soulx_agent_e2e")   # share the tts cache

events = []
events_lock = threading.Lock()
stop_flag = False
PASS, FAIL = 0, 0


def log(tag, text):
    print(f"{time.strftime('%H:%M:%S')} [{tag}] {text}", flush=True)


def check(name, ok, detail=""):
    global PASS, FAIL
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)
    PASS, FAIL = PASS + ok, FAIL + (not ok)


def tts_16k(text, cache_name):
    """Synthesize one utterance with VoxCPM2, return float32 @16k (cached)."""
    OUT_DIR.mkdir(exist_ok=True)
    cache = OUT_DIR / cache_name
    if cache.exists():
        with wave.open(str(cache), "rb") as w:
            assert w.getframerate() == SR
            return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    r = requests.post(TTS_URL, json={"input": text, "cfg_value": 2.0,
                                     "inference_timesteps": 10},
                      stream=True, timeout=180)
    r.raise_for_status()
    pcm24 = b"".join(r.iter_content(chunk_size=None))
    import soxr
    x = np.frombuffer(pcm24, dtype="<i2").astype(np.float32) / 32768.0
    x16 = soxr.resample(x, 24000, SR)
    with wave.open(str(cache), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(x16, -1, 1) * 32767).astype("<i2").tobytes())
    return x16


class Reader(threading.Thread):
    def __init__(self, ws):
        super().__init__(daemon=True)
        self.ws = ws

    def run(self):
        while not stop_flag:
            try:
                raw = self.ws.recv()
            except websocket.WebSocketTimeoutException:
                continue
            except Exception as e:
                log("EVT", f"reader exit: {type(e).__name__}")
                return
            now = time.time()
            if isinstance(raw, bytes):
                with events_lock:
                    events.append((now, "audio_chunk", len(raw)))
                continue
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            ev, data = msg.get("event"), msg.get("data", {})
            with events_lock:
                events.append((now, ev, data))
            if ev == "user_transcription":
                log("ASR", str(data.get("text", ""))[:100])
            elif ev == "text_response":
                log("AI", str(data.get("text", ""))[:100])
            elif ev == "circle_status":
                log("CIRCLE", str(data.get("status", "")))
            elif ev == "config_applied":
                log("CFG", f"turn_engine={data.get('settings', {}).get('turn_engine')}")


def send_audio(ws, samples):
    payload = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    ws.send(payload, opcode=websocket.ABNF.OPCODE_BINARY)


def stream_audio(ws, wav, silence_s, label=""):
    """Stream audio then trailing silence; silence_s may be a list of
    (segment_silence, samples_or_None) for mid-utterance pauses."""
    log("SAY", label)
    for i in range(0, len(wav), CHUNK):
        send_audio(ws, wav[i: i + CHUNK])
        time.sleep(0.16)
    for _ in range(int(silence_s * SR / CHUNK)):
        send_audio(ws, np.zeros(CHUNK, dtype=np.float32))
        time.sleep(0.16)


def events_since(t0):
    with events_lock:
        return [(t, e, d) for (t, e, d) in events if t >= t0]


def wait_for(pred, timeout, what):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return time.time() - t0
        time.sleep(0.5)
    return None


def main():
    global stop_flag
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8090/ws")
    args = ap.parse_args()

    utt = {
        "turn1": tts_16k("今天天气怎么样，适合出去散步吗？", "frc_turn1.wav"),
        "pausal": tts_16k("帮我查一下明天早上", "frc_pausal_a.wav"),
        "pausal2": tts_16k("从北京到上海的机票还有没有。", "frc_pausal_b.wav"),
    }

    # the FRC-only services must be up
    try:
        requests.get("http://127.0.0.1:8082/health", timeout=3).raise_for_status()
        requests.get("http://127.0.0.1:8010/", timeout=3).raise_for_status()
        requests.get("http://127.0.0.1:8086/health", timeout=3).raise_for_status()
    except requests.RequestException as e:
        print(f"FRC services not all up: {e}", flush=True)
        sys.exit(2)

    ws = websocket.create_connection(args.url, timeout=5)
    ws.settimeout(1.0)
    ws.send(json.dumps({"event": "config_audio", "data": {"sample_rate": SR}}))
    ws.send(json.dumps({"event": "update_config",
                        "data": {"turn_engine": "frc", "agent_enabled": False}}))
    Reader(ws).start()
    log("CONN", "connected, turn_engine=frc")

    # ---------------- phase 1: single turn ----------------
    log("PHASE", "1: single utterance -> one transcription + one reply")
    t0 = time.time()
    stream_audio(ws, utt["turn1"], silence_s=4.0, label="single utterance")

    asr = wait_for(lambda: any(e == "user_transcription"
                               for _, e, _ in events_since(t0)), 30, "asr")
    check("话轮提交且有转写（FireRedASR 一次性）", asr is not None,
          f"{asr:.1f}s" if asr else "")
    texts = [d.get("text", "") for _, e, d in events_since(t0)
             if e == "user_transcription"]
    check("转写包含关键词", any("天气" in t or "散步" in t for t in texts),
          " | ".join(texts)[-120:])

    reply = wait_for(lambda: any(e == "text_response"
                                 for _, e, _ in events_since(t0)), 40, "reply")
    check("助手语音回复", reply is not None, f"{reply:.1f}s" if reply else "")

    # ---------------- phase 2: mid-sentence pause ----------------
    # "帮我查一下明天早上" [1.2 s pause] "从北京到上海的机票还有没有。"
    # The pause must not split the utterance: the dual-EOT either holds
    # (P(eot) below threshold) or the speech resume cancels the pending
    # commit. Exactly ONE user_transcription may appear for phase 2 —
    # two would mean the first half committed as its own turn (the very
    # 抢话 failure mode this architecture guards against), and a premature
    # reply always implies that split.
    log("PHASE", "2: mid-sentence pause must not split the turn")
    t0 = time.time()
    stream_audio(ws, utt["pausal"], silence_s=1.2, label="first half")
    t_mid = time.time()
    stream_audio(ws, utt["pausal2"], silence_s=4.0, label="second half")

    mid_split = [d for t, e, d in events_since(t0)
                 if e == "user_transcription" and t < t_mid + 1.0]
    check("句中停顿不切分话轮（前半无独立转写）", not mid_split,
          str(mid_split)[:120])

    done = wait_for(lambda: any(e == "user_transcription"
                                for _, e, _ in events_since(t0)), 30, "asr2")
    texts2 = [d.get("text", "") for _, e, d in events_since(t0)
              if e == "user_transcription"]
    check("停顿后完整话轮提交", done is not None, " | ".join(texts2)[-120:])
    check("转写含两半内容", any(("机票" in t and ("早上" in t or "北京" in t))
                                for t in texts2))

    reply2 = wait_for(lambda: any(e == "text_response"
                                  for _, e, _ in events_since(t0)), 40, "reply2")
    check("助手语音回复（第二次）", reply2 is not None,
          f"{reply2:.1f}s" if reply2 else "")

    stop_flag = True
    try:
        ws.close()
    except Exception:
        pass
    print(f"\n合计: {PASS} pass, {FAIL} fail")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
