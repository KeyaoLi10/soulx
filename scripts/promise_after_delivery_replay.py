# -*- coding: utf-8 -*-
"""一次性诊断复现（2026-11-10，会话 c41a43ce 02:29 事故）：

症状：T3（黄仁勋新闻）结果 02:29:15-35 已完整播报（Ratio 1.00，
played_confirmed@02:29:21），交付原文 02:29:35 以 assistant 进共享
历史；但 02:29:36 闲聊路回"你就安心等着听结果吧"、02:29:50 回
"行，那我先帮你记着，等后面查到具体是什么再跟你细说。"（与
02:29:08 逐字相同）——面板明明写着"结果已播报过"。

方法：按 web.log 逐字重建 02:29:49 / 02:29:36 两代请求的
generate_with_history 实际 messages（含 add_message 的 groupby 连续
同角色 \\n 合并——承诺句与交付原文会并成一条 assistant 消息），
面板用 sqlite 当前状态（三任务 played_confirmed，与当时一致）渲染
render_chitchat_context，POST 活的 :6007 /chat 复现。

Run: cd /home/research/krisl/soulx && PYTHONPATH=repo/dialogue_system \
  venv/bin/python -I scripts/promise_after_delivery_replay.py
"""
import json
import sqlite3
import sys
import requests

sys.path.insert(0, "repo/dialogue_system")
from clients.llm_client import QwenLLM_stream  # noqa: E402
from agent.context import render_chitchat_context  # noqa: E402

DB = ("file:repo/dialogue_system/checkpoints/agent_v2/"
      "300a88bb2d65a7ad.sqlite3?mode=ro")
API = "http://localhost:6007/chat"

# ---- 面板（与 02:29:49 时一致：三任务 completed / played_confirmed）----
conn = sqlite3.connect(DB, uri=True)
conn.row_factory = sqlite3.Row
rows = []
for t in conn.execute(
        "SELECT t.task_id, t.execution_status, t.cancel_intent, r.goal "
        "FROM tasks t JOIN task_revisions r ON r.session_id=t.session_id "
        "AND r.task_id=t.task_id AND r.revision=t.current_revision "
        "WHERE t.session_id LIKE 'c41a43ce%' ORDER BY t.created_at"):
    rows.append({"task_id": t["task_id"],
                 "goal": t["goal"],
                 "execution_status": t["execution_status"],
                 "cancel_intent": t["cancel_intent"],
                 "delivered": True, "suppressed": False,
                 "interrupted": False, "heard": ""})
conn.close()
PANEL = render_chitchat_context(rows)
SYSTEM = QwenLLM_stream.compose_system_prompt(None)

DELIVERY = ("黄仁勋最近最大的新闻是：英伟达正在调查黄仁勋与疑似中国间谍的"
            "婚外情，同时黄仁勋也被认为是中国间谍行动的目标。此外，黄仁勋在"
            "2025财年薪酬猛增46%，接近5000万美元。")
PROMISE = "行，那我先帮你记着，等后面查到具体是什么再跟你细说。"

# 合并后的共享历史尾部（web.log 逐字重建；add_message 把连续同角色
# 消息合并成一条，所以"留意一下/先记着/交付原文"是同一条 assistant）
HIST = [
    {"role": "assistant",
     "content": "好嘞，我马上帮你查查黄仁勋最近的新闻，看看具体是什么情况。"},
    {"role": "user",
     "content": "嗯，好。\n呃，对，然后你可以。"},
    # M1：三次连续 assistant 语音（留意一下[截断] + 先记着 + 交付）合并
    {"role": "assistant",
     "content": "好的，那我帮你留意一下\n" + PROMISE + "\n" + DELIVERY},
    {"role": "user", "content": "嗯嗯，好的，了解了了解了。"},
    {"role": "assistant",
     "content": "嗯嗯，都安排好了，你就安心等着听结果吧。"},
    {"role": "user",
     "content": "嗯，好的好的。\n嗯，好的好的，我知道了。"},
]


def run(name, history, n=3):
    print(f"\n===== {name} =====")
    for i in range(n):
        msgs = ([{"role": "system", "content": SYSTEM}] + history)
        if PANEL:
            msgs.append({"role": "user", "content": PANEL})
        r = requests.post(API, json={"messages": msgs}, timeout=60)
        r.raise_for_status()
        print(f"  [{i}] {r.text.strip()}")


# R1：02:29:49 精确窗口（max_history=5 → 最后 5 条 + 面板 note）
run("R1 02:29:49 exact window", HIST[-5:])
# R2：02:29:36 窗口（当时历史只到"了解了了解了"）
run("R2 02:29:36 window", HIST[:4][-5:])
# R3 消融：M1 里去掉两句承诺（只剩交付原文）——承诺语域是否根因
h3 = [dict(m) for m in HIST]
h3[2] = {"role": "assistant", "content": DELIVERY}
run("R3 no promise lines in M1", h3[-5:])
# R4 消融：去掉 02:29:36 那条错误承诺（M3 换成普通接话）
h4 = [dict(m) for m in HIST]
h4[4] = {"role": "assistant", "content": "嗯，好嘞。"}
run("R4 M3 not a promise", h4[-5:])

# R5 探针：面板纪律里加一行"已播报=话题闭环"不变量能否压住承诺语域
panel5 = PANEL.replace(
    "- 结果播报由后台负责，你不主动报任何结果内容；用户问起任务情况，按面板状态直接回答。",
    "- 结果播报由后台负责，你不主动报任何结果内容；用户问起任务情况，按面板状态直接回答。\n"
    "- 标了「结果已播报过」的任务话题已经闭环，不要再承诺去查、去记、去留意。")
assert panel5 != PANEL
PANEL_SAVE = PANEL
globals()["PANEL"] = panel5
run("R5 + one closure invariant line", HIST[-5:])
globals()["PANEL"] = PANEL_SAVE

# R6 探针：完全没有面板注记
globals()["PANEL"] = ""
run("R6 no panel note", HIST[-5:])
