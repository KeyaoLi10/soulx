#!/usr/bin/env python
"""毛刺话轮（blip turn）回归测试。

2026-10-30 用户实测：环境安静仍有"幻听"——paraformer 在回声/噪声残留上
幻觉出"怎么样""是不是""怎么办""平时"这类短碎片并被当成真话轮送进 LLM。
根因：抢话路径有 3-chunk（480ms）迟滞，但话轮完成路径没有——1-2 chunk
的 episode 也能"完成"一个话轮。修复：完整话轮到达时检查 voice_run，
低于阈值直接丢弃（emit pvad_reject kind=blip_turn）。

本测试验证：
  1. <480ms 的语音毛刺（即使能量足够高）不产生话轮，收到 blip_turn 拒绝
  2. >=480ms 的正常短句照常触发话轮
  3. 长话语被 turn server 拆分出的尾段（文本在音频转 idle 后才到达，
     voice_run 读数 0-2）不被误杀 —— 2026-10-30 首版过滤器的实际回归

用法：venv/bin/python -I scripts/blip_turn_test.py [--wav 语音wav]
（默认借用 /tmp/soulx_agent_e2e/ 的合成语音；没有则用 TTS 合成）

开关说明：过滤器部署状态由 app.py 的 _BLIPTURN_FILTER 决定（2026-10-30
晚下线：真人 WebRTC 语音 nonidle/idle 逐块交替、幻听碎片同形态，计数判据
无法区分，详见 docs/修改日志.md）。本测试自动读取该开关：开着 → 断言毛刺
被丢弃；关着 → 断言所有话轮照常触发（"不误杀真话轮"是任何时候都必须成立
的回归底线）。
"""
import argparse
import json
import re
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np
import websocket

SR = 16000
CHUNK = 2560

APP_SRC = (Path(__file__).resolve().parent.parent
           / "repo" / "dialogue_system" / "app.py").read_text()
FILTER_ON = re.search(r"^_BLIPTURN_FILTER = True", APP_SRC, re.M) is not None

events = []
stop_flag = False
PASS, FAIL = 0, 0


def log(tag, text):
    print(f"{time.strftime('%H:%M:%S')} [{tag}] {text}", flush=True)


def check(name, ok, detail=""):
    global PASS, FAIL
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)
    PASS, FAIL = PASS + ok, FAIL + (not ok)


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
            if isinstance(raw, bytes):
                continue
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            with threading.Lock():
                events.append((time.time(), msg.get("event"), msg.get("data", {})))


def send(ws, samples):
    payload = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    ws.send(payload, opcode=websocket.ABNF.OPCODE_BINARY)


def stream(ws, wav, label=""):
    for i in range(0, len(wav), CHUNK):
        send(ws, wav[i: i + CHUNK])
        time.sleep(0.16)


def stream_silence(ws, seconds):
    for _ in range(int(seconds * SR / CHUNK)):
        send(ws, np.zeros(CHUNK, dtype=np.float32))
        time.sleep(0.16)


def since(t0):
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
    ap.add_argument("--wav", default="/tmp/soulx_agent_e2e/delegate.wav")
    args = ap.parse_args()

    with wave.open(args.wav, "rb") as w:
        assert w.getframerate() == SR, f"wav must be {SR}Hz, got {w.getframerate()}"
        speech = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0

    ws = websocket.create_connection(args.url, timeout=5)
    ws.settimeout(1.0)
    ws.send(json.dumps({"event": "config_audio", "data": {"sample_rate": SR}}))
    ws.send(json.dumps({"event": "update_config",
                        "data": {"pvad_mode": "enforce", "agent_enabled": False}}))
    Reader(ws).start()
    log("CONN", "connected")

    # --- 1) blip turns: 1-chunk and 2-chunk episodes of REAL speech energy ---
    for n_chunks, label in ((1, "1 chunk (160ms)"), (2, "2 chunks (320ms)")):
        t0 = time.time()
        stream(ws, speech[: n_chunks * CHUNK], label=label)
        stream_silence(ws, 3.0)
        fired = wait_for(
            lambda: any(e == "turn_start" for _, e, _ in since(t0)), 8, "turn")
        rejected = any(e == "pvad_reject" and d.get("kind") == "blip_turn"
                       for _, e, d in since(t0))
        if FILTER_ON:
            check(f"{label} 毛刺不触发话轮", fired is None)
            check(f"{label} 收到 blip_turn 拒绝", rejected)
        else:
            print(f"  [SKIP] 过滤器已下线（_BLIPTURN_FILTER=False），"
                  f"毛刺丢弃断言跳过")

    # --- 2) control: a real short sentence (>=480ms voice) still fires ---
    t0 = time.time()
    stream(ws, speech[: int(2.0 * SR)], label="2s speech")
    stream_silence(ws, 4.0)
    fired = wait_for(lambda: any(e == "turn_start" for _, e, _ in since(t0)), 20, "turn")
    if fired is not None:
        fired = next(t - t0 for t, e, _ in since(t0) if e == "turn_start")
    check("正常短句照常触发话轮", fired is not None,
          f"话轮在发话开始后 {fired:.1f}s" if fired else "")

    # --- 3) regression: split-utterance trailing segments must NOT be
    #        dropped (they arrive after idle with voice_run 0-2) ---
    t0 = time.time()
    stream(ws, speech, label="full utterance")
    stream_silence(ws, 5.0)
    wait_for(lambda: any(e == "turn_start" for _, e, _ in since(t0)), 20, "turn")
    fired_any = any(e == "turn_start" for _, e, _ in since(t0))
    misdrops = [(e, d) for _, e, d in since(t0)
                if e == "pvad_reject" and d.get("kind") == "blip_turn"]
    check("完整长句触发话轮", fired_any)
    check("长句拆分尾段零误杀", not misdrops,
          f"误杀: {[d.get('text') for _, d in misdrops]}" if misdrops else "")

    stop_flag = True
    try:
        ws.close()
    except Exception:
        pass
    print(f"\n合计: {PASS} pass, {FAIL} fail")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
