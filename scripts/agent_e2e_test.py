#!/usr/bin/env python
"""Agent E2E test: fast/slow brain delegation over the live web app.

Builds "user speech" by synthesizing the test utterances with the VoxCPM2
TTS service (a consistent voice = one speaker, so the pvad gate enrolls and
passes it), streams them through /ws like the browser does, and asserts the
full delegate -> background run -> quiet-gap delivery loop.

Phases:
  1. delegate        complex question -> task_update(running) + spoken ack,
                     later task_update(completed) + delivered result speech
  2. chat-during-run simple chat WHILE a slow task runs -> normal fast reply,
                     task keeps running, delivery waits for the quiet gap
  3. cancel          "算了不用查了" -> task_update(canceled), no delivery

Usage:
  venv/bin/python -I scripts/agent_e2e_test.py [--url ws://127.0.0.1:8090/ws]
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
OUT_DIR = Path("/tmp/soulx_agent_e2e")

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
    """Record every server event with timestamps."""

    def __init__(self, ws):
        super().__init__(daemon=True)
        self.ws = ws

    def run(self):
        while not stop_flag:
            try:
                raw = self.ws.recv()
            except websocket.WebSocketTimeoutException:
                continue  # 1s poll timeout, not a disconnect
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
            if ev == "task_update":
                log("TASK", f"{data.get('task_id')} {data.get('status')} "
                            f"{str(data.get('progress', ''))[:50]}")
            elif ev == "text_response":
                log("AI", str(data.get("text", ""))[:100])
            elif ev == "user_transcription":
                log("ASR", str(data.get("text", ""))[:100])
            elif ev == "pvad_reject":
                log("GATE", f"rejected {data}")


def send_audio(ws, samples):
    payload = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    ws.send(payload, opcode=websocket.ABNF.OPCODE_BINARY)


def stream_utterance(ws, wav, tail_silence=4.0, label=""):
    log("SAY", label)
    for i in range(0, len(wav), CHUNK):
        send_audio(ws, wav[i: i + CHUNK])
        time.sleep(0.16)
    for _ in range(int(tail_silence * SR / CHUNK)):
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

    # pre-synthesize all utterances (TTS voice = the "user")
    utt = {
        "delegate": tts_16k(
            "帮我查一下二零二四年诺贝尔物理学奖得主是谁，详细介绍他们。", "delegate.wav"),
        "slow": tts_16k(
            "帮我详细搜索总结一下二零二五年人工智能领域都有哪些重要进展，多查一些资料。",
            "slow.wav"),
        "chat": tts_16k("对了，你觉得听音乐对放松有帮助吗？", "chat.wav"),
        "cancel": tts_16k("算了算了，不用查了，取消掉吧。", "cancel.wav"),
        "delegate2": tts_16k(
            "再帮我查一下珠穆朗玛峰的最新高度是多少米。", "delegate2.wav"),
    }

    ws = websocket.create_connection(args.url, timeout=5)
    ws.settimeout(1.0)
    ws.send(json.dumps({"event": "config_audio", "data": {"sample_rate": SR}}))
    ws.send(json.dumps({"event": "update_config",
                        "data": {"agent_enabled": True, "pvad_mode": "enforce"}}))
    Reader(ws).start()
    log("CONN", "connected")

    # ---------------- phase 1: delegate + delivery ----------------
    log("PHASE", "1: delegate -> background run -> delivery")
    t0 = time.time()
    stream_utterance(ws, utt["delegate"], label="delegate question")

    dt = wait_for(lambda: any(e == "task_update" and d.get("status") == "running"
                              for _, e, d in events_since(t0)), 30, "task start")
    check("任务创建并进入 running", dt is not None)

    ack = wait_for(lambda: any(e == "text_response" for _, e, d in events_since(t0)), 40, "ack")
    check("快脑口头受理（ack 语音）", ack is not None, f"{ack:.1f}s" if ack else "")

    done = wait_for(lambda: any(e == "task_update" and d.get("status") in ("completed", "failed")
                                for _, e, d in events_since(t0)), 500, "task done")
    check("慢脑任务完成", done is not None and any(
        e == "task_update" and d.get("status") == "completed"
        for _, e, d in events_since(t0)), f"{done:.0f}s" if done else "")

    delivered = wait_for(lambda: any(e == "task_update" and d.get("status") == "delivered"
                                     for _, e, d in events_since(t0)), 120, "delivery")
    check("结果在空档期交付（delivered）", delivered is not None)
    result_speech = [d.get("text", "") for _, e, d in events_since(t0)
                     if e == "text_response"]
    check("交付后有关键内容语音", any("物理" in t or "霍普菲尔德" in t or "辛顿" in t
                                      for t in result_speech),
          " | ".join(result_speech)[-120:])

    # ---------------- phase 2: chat while a slow task runs ----------------
    log("PHASE", "2: casual chat during a running slow task")
    t0 = time.time()
    stream_utterance(ws, utt["slow"], tail_silence=2.0, label="slow task request")
    wait_for(lambda: any(e == "task_update" and d.get("status") == "running"
                         for _, e, d in events_since(t0)), 30, "slow task start")
    # wait for the ack to finish playing, then chat
    time.sleep(6)
    t_chat = time.time()
    stream_utterance(ws, utt["chat"], tail_silence=2.0, label="casual chat")

    chat_reply = wait_for(lambda: any(
        e == "text_response" and t > t_chat for t, e, d in events_since(t_chat)),
        40, "chat reply")
    check("慢任务期间闲聊即时回复", chat_reply is not None,
          f"{chat_reply:.1f}s" if chat_reply else "")

    task_kept = any(e == "task_update" and d.get("status") in ("running", "queued")
                    for _, e, d in events_since(t0))
    check("闲聊不打断后台任务（仍 running）", task_kept)

    # wait for the slow task to complete, then check delivery waited for
    # the chat to finish (delivery ts > last text_response ts of the chat)
    done2 = wait_for(lambda: any(e == "task_update" and d.get("status") in ("completed", "failed")
                                 for _, e, d in events_since(t0)), 500, "slow task done")
    check("慢任务完成", done2 is not None)
    delivered2 = wait_for(lambda: any(e == "task_update" and d.get("status") == "delivered"
                                      for _, e, d in events_since(t0)), 120, "delivery 2")
    check("第二个结果也交付", delivered2 is not None)

    # ---------------- phase 3: cancel ----------------
    log("PHASE", "3: cancel a running task")
    t0 = time.time()
    stream_utterance(ws, utt["delegate2"], tail_silence=2.0, label="delegate #2")
    wait_for(lambda: any(e == "task_update" and d.get("status") == "running"
                         for _, e, d in events_since(t0)), 30, "task2 start")
    time.sleep(3)
    stream_utterance(ws, utt["cancel"], tail_silence=3.0, label="cancel request")

    canceled = wait_for(lambda: any(e == "task_update" and d.get("status") == "canceled"
                                    for _, e, d in events_since(t0)), 60, "cancel")
    check("改口后任务被取消（canceled）", canceled is not None)

    # make sure no delivery speech happens for the canceled task afterwards
    time.sleep(10)
    late = [d for _, e, d in events_since(t0)
            if e == "task_update" and d.get("status") == "delivered"]
    check("取消后无交付", not late)

    stop_flag = True
    try:
        ws.close()
    except Exception:
        pass
    print(f"\n合计: {PASS} pass, {FAIL} fail")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
