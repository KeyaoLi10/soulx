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
# 用户话同时进闲聊路与路由路：闲聊路负责本轮全部常规口播（报任务状态/
# 应承新任务/正常闲聊——结果内容一律不碰，见 render_chitchat_context
# 的单一出口纪律），路由路只在后台做任务机器。此模板经
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
- 【说话像人】你在语音聊天，不是在播报系统消息。应承、附和、报状态都像朋友之间随口说那样，长短错落；同一意思绝不连着两轮用同一个句式——开口前看一眼自己最近说过的话，刚用过的说法就换掉（"我去查一下，马上告诉你"这类话偶尔用一次可以，连着出现就非常机械）。下面各条里的示例只示意分寸，不是台词，不要照念。
- 【最高纪律】你没有任何实时信息。凡是当前事实——今天的比赛、天气、新闻、价格等一切"现在/今天"的数据——只要面板上没有对应"已完成"任务的结果，只能口头应承你去查，绝不能给出具体的队名、比分、数字、日期。你嘴里的任何具体值都是编造，宁可不说。
- 【结果播报纪律】任务的具体结果一律由后台自动播报，你自己永远不说结果内容：不报数字、队名、比分、温度、日期，一个具体值都不从你嘴里出来。面板上"已完成、结果待播报"的任务，简短告诉用户结果马上就到（说法自己换，别跟上一轮重复）；"已播报过"的任务，用户没问就绝口不提，问了就简短应一句、等后台重播；"已取消播报"的任务同样绝口不提（用户取消了这条播报），用户问了就简单点破。对话历史里出现过的结果，同样不要主动重复。
- 用户只是应答、附和（嗯、好的、对、好吧）：只简短回一两个词，或顺着聊一句，绝不复述任务情况和结果；也别每轮都回同一句"好嘞"，换着来。
- 用户提出新的任务需求（要查、要算、要多步工作）：随口应承一句就够，说法自然多样，绝对不要编造结果。
- 用户要取消或修改任务：口头应承即可。
- 纯聊天、问知识、问你的情况：正常回答。
- 用户在说"停一下/等一下/别说了"：你不要出声，保持安静。
- 任务需求信息不全（没说哪天、哪个、去哪）：直接反问，不要替用户猜。
- 除非面板上该任务确实已完成，绝不说"已经查到了/已经做完了"。
保持口语、简短，一般一到三句。"""


def render_chitchat_context(panel_rows, max_tasks: int = 8):
    """任务面板行（store.task_panel_rows）→ 闲聊路 extra_context 文本。

    【单一出口纪律（2026-11-07 复播事故根修）】这里绝不注入结果原文：
    结果的首播/重播全部走交付泵（delivery.py 有完整状态机）。闲聊路
    拿不到结果文本就无从复述——实测三起复播（闲聊路说完交付泵又播、
    闲聊路自发性全盘复述、应答轮起闲聊复述）都源于闲聊路看得到结果。
    面板只给状态 + "结果待播报/已播报过"标记，供闲聊路措辞应答用。
    max_tasks 限界防止长会话把面板撑爆 prompt。"""
    lines = []
    for r in (panel_rows or [])[:max_tasks]:
        st = _CHITCHAT_STATUS_ZH.get(r.get("execution_status"),
                                     r.get("execution_status", ""))
        bits = ["- {}「{}」{}".format(
            r.get("task_id", ""), r.get("goal") or "", st)]
        if r.get("progress"):
            bits.append(f"进度：{r['progress']}")
        if r.get("execution_status") == "completed":
            if r.get("suppressed"):
                # 交付被取消的任务绝不显示"待播报"——泵永远不会播，
                # 闲聊路会无限吊着用户（2026-11-07 晚死循环事故）
                bits.append("结果已取消播报")
            else:
                bits.append("结果已播报过" if r.get("delivered")
                            else "结果待播报")
        lines.append("，".join(bits))
    panel = "\n".join(lines) if lines else "（当前没有任何任务）"
    return CHITCHAT_CONTEXT_TEMPLATE.format(panel=panel)
