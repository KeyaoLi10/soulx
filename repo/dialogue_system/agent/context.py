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

    # 结果交付状态（2026-11-09 get_task 目标守卫）：终态条目带上
    # "结果已完整播报/播报被取消"标注——路由器据此判断泛泛的进度
    # 追问（"查了吗"）不该指向已播报的旧任务
    dstates = {}
    try:
        dstates = store.result_delivery_states(session_id)
    except Exception:  # noqa: BLE001 — 标注缺失不拦路由
        pass

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
                # get_task 目标守卫（2026-11-09）：已完整播报/已取消播报
                # 的任务，泛泛追问（无话题词、无重播词）不得指向它
                "delivered": dstates.get(t["task_id"]) in (
                    C.DLV_PLAYED_CONFIRMED, C.DLV_SENT_UNCONFIRMED),
                "suppressed": bool(t["cancel_intent"])
                or dstates.get(t["task_id"]) == C.DLV_SUPPRESSED,
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
# 用户话同时进闲聊路与路由路：闲聊路负责本轮全部常规口播，路由路只在
# 后台做任务机器。此模板经 generate_with_history 的 extra_context 注入
# （user-role note，仅本轮）。
#
# 2026-11-09 渐进式披露重构（三事故复盘：笑话/数数被委派语域吞掉、
# 播报后疯狂提起旧任务）：注入从"全量任务上下文 + 12 条纪律"瘦身为
# "状态摘要 + 三条不变量"——
#   1. 只暴露任务简介与播报状态，执行过程（进度）与结果原文一律不给；
#      被打断的交付附带"用户实际听到部分"，用户问起时闲聊路凭摘要
#      直接回答，不需要承诺话术或回执兜底；
#   2. 播报被打断只记录状态，绝不附带"想听要再说一声"之类的行为
#      指令——那是一个泵永不关闭的开放环，每轮注入都在教闲聊路主动
#      吊用户（本次事故：同一句 NBA 邀请连播两遍，逐字相同）；
#   3. 纪律压缩为三条不变量，砍掉全部示例句与事故注记——厚纪律
#      （单条 200+ 字）实测拦不住违规，示例句反而成了措辞锚定源。
# ---------------------------------------------------------------------------

_CHITCHAT_STATUS_ZH = {
    "queued": "排队中", "running": "进行中", "completed": "已完成",
    "failed": "失败", "canceled": "已取消", "superseded": "已被新版本取代",
    "cancel_requested": "正在停止", "blocked_dependency": "等前置任务",
    "waiting_input": "等你补充信息",
}

# 用户实际听到部分的注入上限：够模型回答"刚才说到哪/我听到了什么"，
# 不至于把面板变回结果全文注入（单一出口纪律）
_HEARD_CHARS = 120

CHITCHAT_CONTEXT_TEMPLATE = """【任务面板】（后台任务实时状态摘要，仅供参考）
{panel}

【口播纪律】
用户这轮话由你负责口播；任务的实际执行、创建、取消都是后台系统的事，不用你操心，结果出来了后台会播报。
- 你没有任何实时信息：凡是"现在/今天"的事实（比赛、天气、新闻、价格），面板上没有已完成结果的，只能应承去查，绝不给出具体值。
- 结果播报由后台负责，你不主动报任何结果内容；用户问起任务情况，按面板状态直接回答。
- 像朋友聊天一样自然简短，同一意思别连着两轮用同一个说法。"""


def render_chitchat_context(panel_rows, max_tasks: int = 8):
    """任务面板行（store.task_panel_rows）→ 闲聊路 extra_context 文本。

    【单一出口纪律（2026-11-07 复播事故根修）】结果原文（待播/已播/
    被打断的未播出部分）绝不注入：结果的首播/重播全部走交付泵。
    唯一例外是"用户实际听到部分"（delivery_heard）——那是用户耳朵
    里已经有的内容，暴露它只会让模型答得更准，不会造成复播。
    max_tasks 限界防止长会话把面板撑爆 prompt。"""
    lines = []
    for r in (panel_rows or [])[:max_tasks]:
        st = _CHITCHAT_STATUS_ZH.get(r.get("execution_status"),
                                     r.get("execution_status", ""))
        bits = ["- {}「{}」{}".format(
            r.get("task_id", ""), r.get("goal") or "", st)]
        if r.get("execution_status") == "completed":
            if r.get("suppressed"):
                # 交付被取消的任务绝不显示"待播报"——泵永远不会播，
                # 闲聊路会无限吊着用户（2026-11-07 晚死循环事故）
                bits.append("结果已取消播报")
            elif r.get("interrupted"):
                # 只记录状态 + 用户听到的部分（渐进式披露）。绝不再
                # 附带"想听要用户再说一声"——打断是终态，这句提示是
                # 永不关闭的开放环（2026-11-09 复读事故根因之一）
                heard = str(r.get("heard") or "").strip()
                if heard:
                    bits.append("结果播报被打断（用户听到了：{}{}）".format(
                        heard[:_HEARD_CHARS],
                        "…" if len(heard) > _HEARD_CHARS else ""))
                else:
                    bits.append("结果播报被打断")
            else:
                bits.append("结果已播报过" if r.get("delivered")
                            else "结果待播报")
        lines.append("，".join(bits))
    panel = "\n".join(lines) if lines else "（当前没有任何任务）"
    return CHITCHAT_CONTEXT_TEMPLATE.format(panel=panel)
