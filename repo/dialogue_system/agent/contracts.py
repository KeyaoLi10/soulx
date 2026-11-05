# -*- coding: utf-8 -*-
"""agent/contracts.py — 控制平面的数据契约与严格解析（手册 §5/§7/§13）。

纯 stdlib，不 import app/clients（runtime 不再为测试拖入整个语音客户端栈
——手册 §15 的 TOOL_QUIET 解耦要求）。所有进入业务执行的计划都必须先过
parse_plan 的白名单校验：结构合法 != 语义正确，但结构不合法绝不能执行。

设计要点（手册 §4 不变量的落地）：
  - I3 幂等：operation_id 由 request_id + plan 版本 + op 序号拼成，重放同一
    计划得到同一 ID，store 据此去重。
  - I4 快脑只能引用快照中的 task_id 或本批次 local_ref；task_id 模式 ^T\\d+$，
    local_ref 模式 ^L\\d+$，最终 ID 由 runtime 分配。
  - §7.3 discriminated union：每个 op 一个类，不裸传 args 字典。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

SCHEMA_VERSION = 1

# ---- 手册 §14.2 的实施初值（非实测承诺） ------------------------------
MAX_OPS_PER_REQUEST = 8
MAX_GOAL_CHARS = 600
MAX_LABEL_CHARS = 40
MAX_QUESTION_CHARS = 200
MAX_ANSWER_CHARS = 600
MAX_DIRECT_CHARS = 1200
MAX_OPS_IN_BATCH = MAX_OPS_PER_REQUEST

# 执行状态（手册 §5.3）
EXEC_QUEUED = "queued"
EXEC_RUNNING = "running"
EXEC_COMPLETED = "completed"
EXEC_FAILED = "failed"
EXEC_WAITING_INPUT = "waiting_input"
EXEC_CANCEL_REQUESTED = "cancel_requested"
EXEC_CANCELED = "canceled"
EXEC_SUPERSEDED = "superseded"

# 交付状态（手册 §5.3）
DLV_PENDING = "pending"
DLV_RENDERING = "rendering"
DLV_SENDING = "sending"
DLV_SENT_UNCONFIRMED = "sent_unconfirmed"
DLV_INTERRUPTED_PARTIAL = "interrupted_partial"
DLV_INTERRUPTED_BEFORE_AUDIO = "interrupted_before_audio"
DLV_FAILED = "failed"
DLV_SUPPRESSED = "suppressed"
DLV_RETRY_EXHAUSTED = "retry_exhausted"
# 播放证据分级：当前只有服务端估计（手册 §13.4），ACK 是 P5 可选项
DLV_PLAYED_CONFIRMED = "played_confirmed"

# 回执 outcome（手册 §13.1）
RC_ACCEPTED = "accepted"
RC_UPDATED = "updated"
RC_CANCEL_REQUESTED = "cancel_requested"
RC_CANCELED = "canceled"
RC_ALREADY_COMPLETED = "already_completed"
RC_DELIVERY_SUPPRESSED = "delivery_suppressed"
RC_REJECTED = "rejected"
RC_ANSWERED = "answered"
RC_INFO = "info"

# 与 clients/llm_client.TOOL_QUIET 同值（legacy 路径继续用那边；此处解耦
# 供新 runtime 与测试使用，避免 agent 包 import 语音客户端）
TOOL_QUIET = "[[quiet]]"

_TASK_ID_RE = re.compile(r"^T\d+$")
_LOCAL_REF_RE = re.compile(r"^L\d+$")

# amend patch 的白名单键：模型只能改目标/约束，绝不能写 status/result/run_id
# （手册 §7.3）
AMEND_PATCH_KEYS = frozenset({
    "time_range", "location", "query", "constraints", "language",
    "output_style", "detail_level", "extra",
})
RUNTIME_QUERIES = frozenset({"datetime"})
REPLY_MODES = frozenset({"direct", "receipts", "clarify", "mixed"})
CANCEL_SCOPES = frozenset({"explicit_targets", "all_active"})


class PlanError(Exception):
    """计划不合法（解析/校验失败）。携带面向日志的原因，绝不含可播报文本。"""


# ---------------------------------------------------------------------------
# 操作对象（discriminated union）
# ---------------------------------------------------------------------------

@dataclass
class Op:
    op: str
    evidence: str = ""


@dataclass
class CreateTask(Op):
    local_ref: str = ""
    label: str = ""
    goal: str = ""
    constraints: dict = field(default_factory=dict)
    depends_on_local_refs: list = field(default_factory=list)


@dataclass
class AmendTask(Op):
    target_task_id: str = ""
    expected_revision: int = 0
    patch: dict = field(default_factory=dict)


@dataclass
class CancelTasks(Op):
    targets: list = field(default_factory=list)   # [{"task_id","expected_revision"}]
    scope: str = "explicit_targets"


@dataclass
class RerunTask(Op):
    target_task_id: str = ""
    expected_revision: int = 0
    refresh_reason: str = ""


@dataclass
class GetTask(Op):
    target_task_id: Optional[str] = None
    list_scope: Optional[str] = None    # "active" | "all_recent"
    fields: list = field(default_factory=list)


@dataclass
class AnswerTaskQuestion(Op):
    target_task_id: str = ""
    expected_revision: int = 0
    question_id: str = ""
    answer: str = ""


@dataclass
class FollowupTask(Op):
    parent_task_id: str = ""
    parent_revision: int = 0
    result_ref: str = ""
    goal: str = ""
    local_ref: str = ""


@dataclass
class RuntimeQuery(Op):
    query: str = ""


@dataclass
class StopSpeech(Op):
    pass


@dataclass
class Plan:
    schema_version: int
    operations: list                     # [Op, ...]
    reply_mode: str = "direct"
    direct_question: Optional[str] = None
    clarification: Optional[str] = None
    unhandled_spans: list = field(default_factory=list)

    def op_types(self):
        return [o.op for o in self.operations]


# ---------------------------------------------------------------------------
# 严格解析
# ---------------------------------------------------------------------------

def _need_str(d, key, max_len, required=True):
    v = d.get(key)
    if v is None:
        if required:
            raise PlanError(f"{key} missing")
        return ""
    if not isinstance(v, str):
        raise PlanError(f"{key} not a string")
    if len(v) > max_len:
        raise PlanError(f"{key} too long ({len(v)} > {max_len})")
    return v


def _need_int(d, key, lo=1):
    v = d.get(key)
    if not isinstance(v, int) or isinstance(v, bool) or v < lo:
        raise PlanError(f"{key} not an int >= {lo}")
    return v


def _parse_op(d) -> Op:
    if not isinstance(d, dict):
        raise PlanError("op entry not an object")
    kind = d.get("op")
    ev = _need_str(d, "quote", 200, required=False) or _need_str(
        d, "evidence_quote", 200, required=False)
    if kind == "create_task":
        return CreateTask(
            op=kind, evidence=ev,
            local_ref=_need_str(d, "local_ref", 16),
            label=_need_str(d, "label", MAX_LABEL_CHARS, required=False),
            goal=_need_str(d, "goal", MAX_GOAL_CHARS),
            constraints=d.get("constraints") if isinstance(
                d.get("constraints"), dict) else {},
            depends_on_local_refs=[
                r for r in (d.get("depends_on_local_refs") or [])
                if isinstance(r, str)],
        )
    if kind == "amend_task":
        patch = d.get("patch")
        if not isinstance(patch, dict):
            raise PlanError("amend patch not an object")
        bad = set(patch) - AMEND_PATCH_KEYS
        if bad:
            raise PlanError(f"amend patch illegal keys: {sorted(bad)}")
        return AmendTask(
            op=kind, evidence=ev,
            target_task_id=_need_str(d, "target_task_id", 16),
            expected_revision=_need_int(d, "expected_revision"),
            patch={k: (v if isinstance(v, (str, int, float, bool)) else str(v))
                   for k, v in patch.items()},
        )
    if kind == "cancel_tasks":
        targets = d.get("targets")
        scope = d.get("scope", "explicit_targets")
        if scope not in CANCEL_SCOPES:
            raise PlanError(f"cancel scope invalid: {scope!r}")
        if scope == "explicit_targets":
            # 缺 target 是校验失败，绝不再解释成"全部"（手册 §7.3）
            if not isinstance(targets, list) or not targets:
                raise PlanError("cancel_tasks explicit_targets requires targets")
            norm = []
            for t in targets:
                if not isinstance(t, dict):
                    raise PlanError("cancel target not an object")
                tid = _need_str(t, "task_id", 16)
                if not _TASK_ID_RE.match(tid):
                    raise PlanError(f"cancel target not a snapshot task id: {tid}")
                norm.append({
                    "task_id": tid,
                    "expected_revision": _need_int(t, "expected_revision", 0),
                })
            targets = norm
        else:
            targets = []
        return CancelTasks(op=kind, evidence=ev, targets=targets, scope=scope)
    if kind == "rerun_task":
        return RerunTask(
            op=kind, evidence=ev,
            target_task_id=_need_str(d, "target_task_id", 16),
            expected_revision=_need_int(d, "expected_revision"),
            refresh_reason=_need_str(d, "refresh_reason", 200, required=False),
        )
    if kind == "get_task":
        tgt = d.get("target_task_id")
        scope = d.get("list_scope")
        if tgt is None and scope is None:
            raise PlanError("get_task needs target_task_id or list_scope")
        if tgt is not None and not _TASK_ID_RE.match(str(tgt)):
            raise PlanError(f"get_task target invalid: {tgt!r}")
        if scope is not None and scope not in ("active", "all_recent"):
            raise PlanError(f"get_task list_scope invalid: {scope!r}")
        return GetTask(
            op=kind, evidence=ev,
            target_task_id=tgt, list_scope=scope,
            fields=[f for f in (d.get("fields") or []) if isinstance(f, str)],
        )
    if kind == "answer_task_question":
        return AnswerTaskQuestion(
            op=kind, evidence=ev,
            target_task_id=_need_str(d, "target_task_id", 16),
            expected_revision=_need_int(d, "expected_revision"),
            question_id=_need_str(d, "question_id", 32),
            answer=_need_str(d, "answer", MAX_ANSWER_CHARS),
        )
    if kind == "followup_task":
        return FollowupTask(
            op=kind, evidence=ev,
            parent_task_id=_need_str(d, "parent_task_id", 16),
            parent_revision=_need_int(d, "parent_revision"),
            result_ref=_need_str(d, "result_ref", 64, required=False),
            goal=_need_str(d, "goal", MAX_GOAL_CHARS),
            local_ref=_need_str(d, "local_ref", 16),
        )
    if kind == "runtime_query":
        q = _need_str(d, "query", 32)
        if q not in RUNTIME_QUERIES:
            raise PlanError(f"runtime_query not whitelisted: {q!r}")
        return RuntimeQuery(op=kind, evidence=ev, query=q)
    if kind == "stop_speech":
        return StopSpeech(op=kind, evidence=ev)
    raise PlanError(f"unknown op: {kind!r}")


def parse_plan(plan_json: str) -> Plan:
    """严格解析 route_turn 的 plan_json。任何不合法都抛 PlanError，
    调用方必须丢弃整批计划（手册 §7.3：截断/不闭合/类型错误/未知 op
    不执行任何该批计划）。"""
    if not isinstance(plan_json, str) or not plan_json.strip():
        raise PlanError("plan_json empty")
    try:
        d = json.loads(plan_json)
    except (json.JSONDecodeError, ValueError) as e:
        raise PlanError(f"plan_json not valid JSON: {e}") from None
    if not isinstance(d, dict):
        raise PlanError("plan not an object")
    if d.get("schema_version") != SCHEMA_VERSION:
        raise PlanError(f"schema_version must be {SCHEMA_VERSION}")
    ops = d.get("operations")
    if not isinstance(ops, list) or not ops:
        raise PlanError("operations missing/empty")
    if len(ops) > MAX_OPS_PER_REQUEST:
        raise PlanError(f"too many operations ({len(ops)} > {MAX_OPS_PER_REQUEST})")
    operations = [_parse_op(o) for o in ops]
    mode = d.get("reply_mode", "direct")
    if mode not in REPLY_MODES:
        raise PlanError(f"reply_mode invalid: {mode!r}")
    plan = Plan(
        schema_version=SCHEMA_VERSION,
        operations=operations,
        reply_mode=mode,
        direct_question=_need_str(d, "direct_question", MAX_DIRECT_CHARS,
                                  required=False) or None,
        clarification=_need_str(d, "clarification", MAX_QUESTION_CHARS,
                                required=False) or None,
        unhandled_spans=[s for s in (d.get("unhandled_spans") or [])
                         if isinstance(s, str)][:MAX_OPS_PER_REQUEST],
    )
    # 引用完整性：本批 create 的 local_ref 必须唯一；depends/followup 引用的
    # local_ref 必须在本批 create 中（task_id 引用的存在性由 runtime 对快照校验）
    local_refs = {o.local_ref for o in operations
                  if isinstance(o, (CreateTask, FollowupTask)) and o.local_ref}
    if len(local_refs) < sum(
            1 for o in operations
            if isinstance(o, (CreateTask, FollowupTask)) and o.local_ref):
        raise PlanError("duplicate local_ref in batch")
    for o in operations:
        if isinstance(o, CreateTask):
            for dep in o.depends_on_local_refs:
                if dep not in local_refs:
                    raise PlanError(f"depends_on_local_refs unknown: {dep}")
    if mode == "clarify" and not plan.clarification:
        raise PlanError("clarify mode requires clarification text")
    return plan


# ---------------------------------------------------------------------------
# 回执与可播报模板（手册 §13.1：程序产模板，不让模型自由解释操作是否发生）
# ---------------------------------------------------------------------------

@dataclass
class Receipt:
    operation_id: str
    op: str
    task_id: Optional[str] = None
    revision: int = 0
    outcome: str = RC_REJECTED
    current_state: str = ""
    reason_code: str = ""
    event_seq: int = 0

    def to_dict(self):
        return {
            "operation_id": self.operation_id, "op": self.op,
            "task_id": self.task_id, "revision": self.revision,
            "outcome": self.outcome, "current_state": self.current_state,
            "reason_code": self.reason_code, "event_seq": self.event_seq,
        }


def receipt_text(receipt: Receipt, label: str = "") -> str:
    """受控回执话术。label 必须来自经过校验的任务名（runtime 传入），
    模板外的内容不进 TTS。"""
    name = label or receipt.task_id or "这个任务"
    if receipt.outcome == RC_ACCEPTED:
        return f"{name}已经安排好了，稍后告诉你结果。"
    if receipt.outcome == RC_UPDATED:
        return f"好的，{name}已经按你说的改了。"
    if receipt.outcome == RC_CANCEL_REQUESTED:
        return f"正在停止{name}。"
    if receipt.outcome == RC_CANCELED:
        return f"{name}已经取消了。"
    if receipt.outcome == RC_ALREADY_COMPLETED:
        return f"{name}刚好已经完成了，结果先不重复说，你需要的话我再讲。"
    if receipt.outcome == RC_DELIVERY_SUPPRESSED:
        return f"好的，{name}的结果先不说。"
    if receipt.outcome == RC_REJECTED:
        reason = {
            "stale_revision": "它的情况刚刚变过，我再确认一下最新状态。",
            "unknown_task": "我没有找到你说的那个任务。",
            "ambiguous": "你说的任务我不太确定是哪一个，能再说下名字吗？",
            "queue_full": "现在排队的任务有点多，这个先没受理。"
        }.get(receipt.reason_code, "这项操作没有提交成功。")
        return reason
    return ""


# ---------------------------------------------------------------------------
# 慢脑执行快照与事件信封（手册 §10）
# ---------------------------------------------------------------------------

@dataclass
class TaskExecutionSnapshot:
    session_id: str
    task_id: str
    revision: int
    run_id: str
    goal: str
    constraints: dict
    context: str
    user_text: str              # 用户原文引用——不只传快脑压缩摘要（§9.1）
    timezone: str = "Asia/Shanghai"
    budget_steps: int = 4
    budget_timeout_s: int = 480
    depends: list = field(default_factory=list)   # [{task_id, revision, result}]

    def to_dict(self):
        return dict(self.__dict__)


def event_envelope(event_id, session_id, task_id, revision, run_id,
                   etype, payload: dict) -> dict:
    """统一事件信封。payload 只放已发生的事实（进度描述来自真实步骤，
    不由模型声称"查了一半"——§10）。"""
    return {
        "event_id": event_id, "session_id": session_id,
        "task_id": task_id, "revision": revision, "run_id": run_id,
        "type": etype, "payload": payload,
    }


# ---------------------------------------------------------------------------
# SpeechEnvelope（手册 §13.2）：任务发言的有效性凭证
# ---------------------------------------------------------------------------

@dataclass
class SpeechEnvelope:
    speech_id: str
    session_id: str
    connection_epoch: int
    kind: str                    # receipt | result | question
    task_id: Optional[str] = None
    revision: int = 0
    run_id: Optional[str] = None
    operation_id: Optional[str] = None
    generation_id: Optional[str] = None
    dependency_versions: dict = field(default_factory=dict)
    validity_token: int = 0      # runtime 每次使失效时递增

    def valid_for(self, task_rev: dict) -> bool:
        """按 kind 编写适用条件（手册 §13.2：不能统一成'必须 running'——
        完成结果应允许 completed，取消确认应允许 canceled）。"""
        if self.task_id is None:
            return True
        cur = task_rev.get(self.task_id)
        if cur is None:
            return False
        if self.revision and cur.get("revision") != self.revision:
            return False     # 已被新版本取代
        st = cur.get("execution_status")
        if self.kind == "result":
            return st in (EXEC_COMPLETED, EXEC_FAILED, EXEC_WAITING_INPUT)
        if self.kind == "receipt":
            # 受理/修改回执在任务被改口后应抑制（§13.2）
            return st in (EXEC_QUEUED, EXEC_RUNNING, EXEC_WAITING_INPUT,
                          EXEC_CANCEL_REQUESTED, EXEC_CANCELED,
                          EXEC_COMPLETED, EXEC_FAILED)
        if self.kind == "question":
            return st == EXEC_WAITING_INPUT
        return True


def operation_id_for(request_id: str, plan_version: int, op_index: int) -> str:
    return f"{request_id}:p{plan_version}:{op_index}"
