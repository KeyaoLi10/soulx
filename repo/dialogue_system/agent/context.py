# -*- coding: utf-8 -*-
"""agent/context.py — 路由输入快照、焦点与待答问题（手册 §7.1/§7.5）。

只读构建，不写任何状态。控制轮始终看到全部活跃任务与最近终态任务索引；
不注入网页正文或慢脑推理。焦点只由确定性事件推出（最近创建/更新的活跃
任务），后台完成顺序、面板排序不得自动改焦点。
"""
from __future__ import annotations

import json
from typing import Optional

from agent import contracts as C

# 终态任务进入索引时保留的字段（§7.1「最近 10 项终态摘要」）
_TERMINAL_FIELDS = ("task_id", "revision", "label", "goal", "execution_status")
_TERMINAL_GOAL_CHARS = 80
_TERMINAL_LIMIT = 10


def default_capabilities():
    """v2 工具面暴露给路由器的能力清单（agent/tools 的白名单）。"""
    try:
        from agent import tools as T
        return [s["function"]["name"] for s in T.NEW_TOOL_SCHEMAS]
    except Exception:  # noqa: BLE001 — 能力清单缺失不致命
        return ["web_search", "fetch_page", "calculate", "get_datetime"]


def build_router_snapshot(store, session_id: str, user_text: str,
                          request_id: Optional[str] = None,
                          capabilities=None,
                          timezone: Optional[str] = None,
                          history=None) -> dict:
    """构建 §7.1 形状的路由快照。history 由适配层传入（最近对话，
    [{role, content}]），本模块不持有会话历史。"""
    capabilities = capabilities or default_capabilities()
    sess = store.session_row(session_id)
    if timezone is None:
        timezone = sess["timezone"] if sess else "Asia/Shanghai"
    tasks = store.task_view(session_id)

    active, terminal = [], []
    pending_question = None
    focus_task_id = None
    for t in tasks:
        st = t["execution_status"]
        if st in (C.EXEC_QUEUED, C.EXEC_RUNNING, C.EXEC_WAITING_INPUT,
                  C.EXEC_CANCEL_REQUESTED, C.EXEC_BLOCKED_DEPENDENCY):
            entry = {
                "task_id": t["task_id"], "revision": t["revision"],
                "label": t["label"], "goal": t["goal"],
                "constraints": t["constraints"],
                "execution_status": st,
            }
            active.append(entry)
            if st == C.EXEC_WAITING_INPUT:
                row = store.task_row(session_id, t["task_id"])
                q = (row["pending_question"] if row else None) or ""
                if q:
                    pending_question = {
                        "task_id": t["task_id"], "revision": t["revision"],
                        "question_id": (row["pending_question_id"]
                                        if row else "") or "",
                        "question": q,
                    }
        elif st in (C.EXEC_COMPLETED, C.EXEC_FAILED, C.EXEC_CANCELED,
                    C.EXEC_SUPERSEDED):
            terminal.append({
                "task_id": t["task_id"], "revision": t["revision"],
                "label": t["label"],
                "goal": str(t["goal"])[:_TERMINAL_GOAL_CHARS],
                "execution_status": st,
            })

    # 焦点（§7.5）：确定性推导——最近创建的活跃任务；无活跃任务则为空。
    # 不看终态顺序、不看未播出文本。
    if active:
        focus_task_id = active[-1]["task_id"]

    # 终态索引按更新时间倒序取最近 N 项
    terminal.sort(key=lambda e: e["task_id"])
    terminal = terminal[-_TERMINAL_LIMIT:]

    # snapshot_version：任务 revision 只增不减，用它做单调版本号
    version = sum(t["revision"] for t in tasks) + len(tasks)

    snap = {
        "snapshot_version": version,
        "request_id": request_id,
        "user_text": user_text,
        "focus_task_id": focus_task_id,
        "pending_question": pending_question,
        "tasks": active,
        "recent_terminal": terminal,
        "capabilities": list(capabilities),
        "timezone": timezone,
    }
    if history:
        snap["recent_history"] = [
            {"role": h.get("role"), "content": h.get("content")}
            for h in history[-8:] if h.get("content")]
    return snap


def render_snapshot_message(snapshot: dict, user_text: str) -> str:
    """把快照 + 用户原文组成控制轮的用户消息（§7.4：用户原文是需求，
    tasks 是 runtime 提供的真实状态）。"""
    return (
        "【任务快照】\n"
        + json.dumps(snapshot, ensure_ascii=False)
        + "\n【用户原文】\n" + user_text
        + "\n请调用 route_turn 给出本轮控制计划。"
    )


# ---------------------------------------------------------------------------
# 闲聊路（voice lane）注入（2026-11-06 双路并行设计）
# 用户话同时进闲聊路与路由路：闲聊路负责本轮全部口播（看任务面板如实
# 报情况/应承新任务/正常闲聊），路由路只在后台做任务机器。此模板经
# generate_with_history 的 extra_context 注入（user-role note，仅本轮）。
# ---------------------------------------------------------------------------

_CHITCHAT_STATUS_ZH = {
    "queued": "排队中", "running": "进行中", "completed": "已完成",
    "failed": "失败", "canceled": "已取消", "superseded": "已被新版本取代",
    "cancel_requested": "正在停止", "blocked_dependency": "等前置任务",
    "waiting_input": "等你补充信息",
}

CHITCHAT_CONTEXT_TEMPLATE = """【任务面板】（后台任务系统实时状态，仅供口播参考）
{panel}

【本轮口播纪律】
用户这轮话由你负责口播；后台另有一套任务系统会实际执行任务，创建、取消、修改都不用你操心，你只管把话说好。
- 用户提到面板里的任务：如实讲它的情况；已完成且有结果的，把结果讲出来（两三句内）。
- 用户提出新的任务需求（要查、要算、要多步工作）：口头应承一句（如"好，我去查一下"），绝对不要编造结果。
- 用户要取消或修改任务：口头应承（如"好的，我把它取消"）即可。
- 纯聊天、问知识、问你的情况：正常回答。
- 用户在说"停一下/等一下/别说了"：你不要出声，保持安静。
- 任务需求信息不全（没说哪天、哪个、去哪）：直接反问，不要替用户猜。
- 除非面板上该任务确实已完成且有结果，绝不说"已经查到了/已经做完了"。
保持口语、简短，一般一到三句。"""


def render_chitchat_context(panel_rows, max_tasks: int = 8):
    """任务面板行（store.task_panel_rows）→ 闲聊路 extra_context 文本。
    max_tasks 限界防止长会话把面板撑爆 prompt。"""
    lines = []
    for r in (panel_rows or [])[:max_tasks]:
        st = _CHITCHAT_STATUS_ZH.get(r.get("execution_status"),
                                     r.get("execution_status", ""))
        bits = ["- {}「{}」{}".format(
            r.get("task_id", ""), r.get("goal") or "", st)]
        if r.get("progress"):
            bits.append(f"进度：{r['progress']}")
        if r.get("result"):
            bits.append(f"结果：{r['result']}")
        lines.append("，".join(bits))
    panel = "\n".join(lines) if lines else "（当前没有任何任务）"
    return CHITCHAT_CONTEXT_TEMPLATE.format(panel=panel)
