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
