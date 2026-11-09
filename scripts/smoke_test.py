#!/usr/bin/env python3
"""Smoke test for the SoulX-Duplug deployment (run AFTER services are up).

1. VoxCPM2 TTS health + a short synthesis (both LoRA checkpoints)
2. fused LLM server /chat (plain text stream)
3. turn-taking server /turn websocket: feed 2s of speech-like audio and
   check we get valid state responses back

2026-11-09: FireRedTTS / /chat_indextts checks removed along with the
FireRedTTS retirement (VoxCPM2 is the only TTS).

Usage: python -I smoke_test.py [--llm-url http://localhost:6007]
"""
import argparse
import base64
import json
import sys
import wave
import io

import numpy as np
import requests


def test_tts():
    print("== VoxCPM2 TTS (:8085) ==")
    r = requests.get("http://127.0.0.1:8085/health", timeout=10)
    health = r.json()
    print("health:", health)
    # synthesize on every available LoRA checkpoint (dual-checkpoint mode)
    loras = health.get("lora")
    if isinstance(loras, dict):
        targets = sorted(loras.keys())
    elif isinstance(loras, list):
        targets = sorted(loras)
    else:
        targets = [None]
    for lora in targets:
        body = {"input": "你好，这是一个语音合成测试。"}
        if lora:
            body["lora"] = lora
        r = requests.post(
            "http://127.0.0.1:8085/synthesize",
            json=body,
            stream=True,
            timeout=180,
        )
        n = 0
        for chunk in r.iter_content(chunk_size=None):
            n += len(chunk)
        print(f"synthesize lora={lora}:", r.status_code, n, "bytes")
        assert r.status_code == 200 and n > 48000


def test_chat(llm):
    print("== LLM /chat ==")
    msgs = [
        {"role": "system", "content": "You are a helpful voice assistant."},
        {"role": "user", "content": "用一句话介绍你自己。"},
    ]
    r = requests.post(f"{llm}/chat", json={"messages": msgs}, stream=True, timeout=120)
    text = ""
    for chunk in r.iter_content(chunk_size=None):  # decode after concat: chunk
        text += chunk.decode("utf-8", errors="replace")  # boundaries split mb chars
    print("reply:", text.strip()[:200])
    assert len(text.strip()) > 5


def test_turn():
    print("== turn server (:8000) ==")
    import websocket

    ws = websocket.create_connection("ws://127.0.0.1:8000/turn", timeout=15)
    ws.settimeout(15)
    # 2s of 440Hz-ish tone bursts so the model sees "audio"
    sr = 16000
    t = np.arange(sr * 2) / sr
    audio = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    # speech-like: amplitude modulated
    audio *= (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t)).astype(np.float32)
    responses = 0
    for off in range(0, len(audio), 2560):
        chunk = audio[off : off + 2560]
        payload = {
            "type": "audio",
            "session_id": "smoke-test-1",
            "audio": base64.b64encode(chunk.astype(np.float32).tobytes()).decode(),
        }
        ws.send(json.dumps(payload))
        try:
            resp = json.loads(ws.recv())
            assert resp["type"] == "turn_state"
            state = resp["state"]["state"]
            if responses < 3 or state != "idle":
                pass
            responses += 1
        except websocket.WebSocketTimeoutException:
            pass
    print(f"got {responses} state responses")
    ws.close()
    assert responses > 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm-url", default="http://localhost:6007")
    ap.add_argument("--skip", nargs="*", default=[])
    args = ap.parse_args()
    if "tts" not in args.skip:
        test_tts()
    if "chat" not in args.skip:
        test_chat(args.llm_url)
    if "turn" not in args.skip:
        test_turn()
    print("\nALL SMOKE TESTS PASSED")
