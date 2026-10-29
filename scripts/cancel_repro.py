#!/usr/bin/env python
"""快脑取消语义复现：构造 E2E phase-3 同款上下文打 /agent_chat，
统计 cancel_task 被真实调用的比例（口头答应不调用 = 失败模式）。

历史结论（2026-10-30）：prompt 强化后仍 0/4——模型把"已取消"当社交性
叙述直接说出口。这促成了 runtime.auto_cancel 的确定性取消方案（生成前
正则检测改口意图并先行取消）。本脚本保留作 prompt 质量探针：若某天
模型能稳定自行调用，可考虑撤掉确定性路径。

用法：venv/bin/python -I scripts/cancel_repro.py [次数]
"""
import json
import os
import sys

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "repo", "dialogue_system"))

from agent.runtime import AGENT_SYSTEM_ADDON, FAST_TOOLS  # noqa: E402

URL = "http://127.0.0.1:6007/agent_chat"

PERSONA = (
    "You are a gentle and natural voice conversation assistant."
    "Your name is Elva."
    "You are communicating with the user via speech; please respond in a natural, brief, and colloquial manner."
    "Do not output extra explanations, do not use lists or markdown format."
)

MESSAGES = [
    {"role": "system", "content": PERSONA + "\n\n" + AGENT_SYSTEM_ADDON},
    {"role": "user", "content": "再帮我查一下珠穆朗玛峰的最新高度是多少米。"},
    {"role": "assistant", "content":
        "好的，我去帮您查一下珠穆朗玛峰的最新高度，马上处理。"},
    {"role": "user", "content": "算了算了，不用查了，取消掉吧。"},
    # the Agent Runtime's task view, injected as a user note before generation
    {"role": "user", "content": "当前后台任务：\n- T3（已运行 8 秒）：查询珠穆朗玛峰的最新高度是多少米"},
]


def one_round():
    r = requests.post(
        URL,
        json={"messages": MESSAGES, "tools": FAST_TOOLS, "thinking": False},
        stream=True, timeout=120,
    )
    r.raise_for_status()
    speech, calls = [], []
    for chunk in r.iter_content(chunk_size=None):
        if not chunk:
            continue
        tag, payload = chunk[:1], chunk[1:].decode("utf-8", errors="replace")
        if tag == b"S":
            speech.append(payload)
        elif tag == b"T":
            calls.append(json.loads(payload))
    return "".join(speech), calls


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    ok = 0
    for i in range(n):
        text, calls = one_round()
        called = any(c["name"] == "cancel_task" for c in calls)
        ok += called
        print(f"[{i+1}/{n}] {'CALL' if called else 'NO-CALL'}  "
              f"calls={[c['name'] for c in calls]}")
        print(f"        speech: {text.strip()[:90]}")
    print(f"\n取消调用率: {ok}/{n}")
    sys.exit(0 if ok == n else 1)
