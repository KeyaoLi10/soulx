# -*- coding: utf-8 -*-
"""agent/store.py — AgentRuntime 的 SQLite 持久层（手册 §5/§8/§14.1）。

职责边界：
  - 本模块只做**事务化状态存取**：谁可以在一个 BEGIN IMMEDIATE 里改什么。
    模型调用、worker 循环、语音交付都在锁外（§8：模型和工具在锁外运行）。
  - commit_plan 是唯一写 tasks/revisions/operations/events/outbox 的入口，
    实现：幂等（operation_id 重放返回旧回执）、批内原子（崩溃窗口不留
    半批）、expected_revision 短事务复验、队列上限拒绝。
  - run lease 用条件更新实现同一时刻单一持有者；过期可被重新领取；
    finish_run 带围栏——run 不再是任务当前 revision 的活跃 run 时只写
    审计状态，绝不更新当前任务（§9.3：过期 run 事件只进历史）。

线程模型：单连接 + 一把 RLock 串行全部写（单写入线程控制器语义），
读方法（task_view 等）也走同一锁以保证一致快照。WAL 模式。
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Optional

from agent import contracts as C

SCHEMA_VERSION = 1
DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "..", "data", "agent_runtime.sqlite3")

# 手册 §14.2 实施初值
MAX_QUEUED_PER_SESSION = 16
OUTBOX_LEASE_TTL_S = 30.0
RUN_LEASE_TTL_S = 120.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(
  key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sessions(
  session_id TEXT PRIMARY KEY,
  connection_epoch INTEGER NOT NULL DEFAULT 1,
  timezone TEXT NOT NULL DEFAULT 'Asia/Shanghai',
  task_focus TEXT NOT NULL DEFAULT '',
  config_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  closed INTEGER NOT NULL DEFAULT 0,
  closed_reason TEXT NOT NULL DEFAULT '',
  task_seq INTEGER NOT NULL DEFAULT 0,
  event_seq INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS requests(
  request_id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  command_seq INTEGER NOT NULL,
  source_commit_id TEXT,
  engine_epoch TEXT NOT NULL DEFAULT '',
  input_revision INTEGER NOT NULL DEFAULT 1,
  text TEXT NOT NULL,
  state TEXT NOT NULL DEFAULT 'registered',
  created_at TEXT NOT NULL,
  UNIQUE(session_id, command_seq));
CREATE INDEX IF NOT EXISTS idx_requests_source
  ON requests(session_id, engine_epoch, source_commit_id, input_revision);
CREATE TABLE IF NOT EXISTS operations(
  operation_id TEXT PRIMARY KEY,
  request_id TEXT NOT NULL,
  op_index INTEGER NOT NULL,
  op_type TEXT NOT NULL,
  args_json TEXT NOT NULL,
  outcome TEXT NOT NULL,
  reason_code TEXT NOT NULL DEFAULT '',
  task_id TEXT,
  receipt_json TEXT NOT NULL,
  committed_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS tasks(
  session_id TEXT NOT NULL,
  task_id TEXT NOT NULL,
  label TEXT NOT NULL DEFAULT '',
  current_revision INTEGER NOT NULL,
  execution_status TEXT NOT NULL,
  active_run_id TEXT,
  cancel_intent INTEGER NOT NULL DEFAULT 0,
  pending_question_id TEXT NOT NULL DEFAULT '',
  pending_question TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  PRIMARY KEY(session_id, task_id));
CREATE TABLE IF NOT EXISTS task_revisions(
  session_id TEXT NOT NULL,
  task_id TEXT NOT NULL,
  revision INTEGER NOT NULL,
  goal TEXT NOT NULL,
  constraints_json TEXT NOT NULL DEFAULT '{}',
  depends_json TEXT NOT NULL DEFAULT '[]',
  source_request_id TEXT,
  user_text TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL,
  PRIMARY KEY(session_id, task_id, revision));
CREATE TABLE IF NOT EXISTS runs(
  run_id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  task_id TEXT NOT NULL,
  revision INTEGER NOT NULL,
  attempt INTEGER NOT NULL DEFAULT 1,
  state TEXT NOT NULL DEFAULT 'queued',
  lease_token TEXT,
  lease_expires_at REAL,
  started_at TEXT, ended_at TEXT,
  stop_reason TEXT NOT NULL DEFAULT '',
  snapshot_json TEXT NOT NULL DEFAULT '{}');
CREATE INDEX IF NOT EXISTS idx_runs_task ON runs(session_id, task_id, revision);
CREATE TABLE IF NOT EXISTS events(
  event_id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  seq INTEGER NOT NULL,
  task_id TEXT,
  revision INTEGER,
  run_id TEXT,
  type TEXT NOT NULL,
  payload_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  UNIQUE(session_id, seq));
CREATE TABLE IF NOT EXISTS tool_calls(
  call_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  tool TEXT NOT NULL,
  input_json TEXT NOT NULL DEFAULT '{}',
  state TEXT NOT NULL DEFAULT 'pending',
  result_json TEXT,
  error TEXT,
  started_at TEXT, ended_at TEXT,
  side_effect INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS deliveries(
  delivery_id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  task_id TEXT, revision INTEGER,
  run_id TEXT, operation_id TEXT,
  speech_id TEXT, generation_id TEXT,
  state TEXT NOT NULL DEFAULT 'pending',
  retries INTEGER NOT NULL DEFAULT 0,
  dependency_versions_json TEXT NOT NULL DEFAULT '{}',
  updated_at TEXT NOT NULL);
CREATE UNIQUE INDEX IF NOT EXISTS idx_delivery_identity ON deliveries(
  session_id, kind, ifnull(task_id,''), ifnull(operation_id,''),
  ifnull(run_id,''));
CREATE TABLE IF NOT EXISTS outbox(
  outbox_id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  payload_json TEXT NOT NULL DEFAULT '{}',
  state TEXT NOT NULL DEFAULT 'pending',
  lease_token TEXT, lease_expires_at REAL,
  attempts INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_outbox_pending ON outbox(state, outbox_id);
CREATE TABLE IF NOT EXISTS route_log(
  route_id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  request_id TEXT,
  mode TEXT NOT NULL,
  snapshot_json TEXT NOT NULL DEFAULT '{}',
  plan_json TEXT,
  parse_ok INTEGER,
  error TEXT,
  latency_ms INTEGER,
  created_at TEXT NOT NULL);
"""

# 交付状态机（手册 §5.3）
_DLV_TRANSITIONS = {
    C.DLV_PENDING: {C.DLV_RENDERING, C.DLV_SENDING, C.DLV_SUPPRESSED,
                    C.DLV_FAILED, C.DLV_RETRY_EXHAUSTED},
    # rendering->pending：静默掐死重试（一个字没播，重新渲染重试）
    C.DLV_RENDERING: {C.DLV_SENDING, C.DLV_SUPPRESSED, C.DLV_FAILED,
                      C.DLV_PENDING, C.DLV_RETRY_EXHAUSTED},
    C.DLV_SENDING: {C.DLV_SENT_UNCONFIRMED, C.DLV_SUPPRESSED, C.DLV_FAILED},
    C.DLV_SENT_UNCONFIRMED: {C.DLV_PLAYED_CONFIRMED,
                             C.DLV_INTERRUPTED_PARTIAL,
                             C.DLV_INTERRUPTED_BEFORE_AUDIO},
    # 终态不再迁移（retry_exhausted 结果仍可查询，不写 delivered）
}

# 重开白名单：显式追问（get_task）可重置为 pending 并重排 deliver——
# 这不是自动重试，是 outbox 重入语义（2026-11-06 待办）。2026-11-07
# 起结果播报收归交付泵单一出口：闲聊路不再代播结果（只说桥接短语），
# 所以"已听到"（played_confirmed/sent_unconfirmed）也必须可重开——
# 用户说"再讲一遍"时唯一能说话的就是交付泵。在途
# （pending/rendering/sending）不重排：结果会自然到达。
_DLV_REOPEN_FROM = frozenset({
    C.DLV_INTERRUPTED_PARTIAL, C.DLV_INTERRUPTED_BEFORE_AUDIO,
    C.DLV_RETRY_EXHAUSTED, C.DLV_SUPPRESSED,
    C.DLV_SENT_UNCONFIRMED, C.DLV_PLAYED_CONFIRMED})


class StoreError(Exception):
    """事务级失败：整批不提交。与语义拒绝（Receipt rejected）不同。"""


@contextmanager
def _nullfail():
    yield


class CommitResult:
    def __init__(self, receipts, bindings, queries, replayed=False):
        self.receipts = receipts          # [C.Receipt]
        self.bindings = bindings          # {local_ref: task_id}
        self.queries = queries            # [{"query","value"}]（runtime_query 等）
        self.replayed = replayed          # True = 幂等重放，未产生新写入


class AgentStore:
    def __init__(self, path=None, max_queued=MAX_QUEUED_PER_SESSION):
        self.path = os.path.abspath(path or DEFAULT_DB_PATH)
        self.max_queued = max_queued
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            self.path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._migrate()

    # ------------------------------------------------------------------ 基础

    def close(self):
        with self._lock:
            self._conn.close()

    def _migrate(self):
        with self._lock:
            self._conn.executescript(_SCHEMA)
            row = self._conn.execute(
                "SELECT value FROM meta WHERE key='schema_version'").fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO meta(key,value) VALUES('schema_version',?)",
                    (str(SCHEMA_VERSION),))
            elif int(row["value"]) != SCHEMA_VERSION:
                raise StoreError(
                    f"agent_runtime db schema {row['value']} != "
                    f"code {SCHEMA_VERSION}; 写迁移再升级")

    @contextmanager
    def _tx(self):
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    def _now(self):
        return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

    # ------------------------------------------------------------- 会话/请求

    def ensure_session(self, session_id, timezone="Asia/Shanghai"):
        with self._tx():
            self._conn.execute(
                "INSERT OR IGNORE INTO sessions(session_id, timezone, "
                "created_at) VALUES(?,?,?)", (session_id, timezone, self._now()))

    def bind_connection_epoch(self, session_id, epoch):
        with self._tx():
            self._conn.execute(
                "UPDATE sessions SET connection_epoch=? WHERE session_id=?",
                (epoch, session_id))

    def close_session(self, session_id, reason="closed"):
        with self._tx():
            self._conn.execute(
                "UPDATE sessions SET closed=1, closed_reason=? "
                "WHERE session_id=?", (reason, session_id))

    def session_row(self, session_id):
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM sessions WHERE session_id=?",
                (session_id,)).fetchone()

    def register_request(self, session_id, text, source_commit_id=None,
                         engine_epoch="", input_revision=1):
        """登记请求（§8 receive_request 第一步：先登记再调模型）。
        幂等：同一 (session, engine_epoch, source_commit_id,
        input_revision) 重复登记返回原 request_id，不重新编号。"""
        with self._tx():
            if source_commit_id is not None:
                row = self._conn.execute(
                    "SELECT request_id FROM requests WHERE session_id=? AND "
                    "engine_epoch=? AND source_commit_id=? AND "
                    "input_revision=?",
                    (session_id, engine_epoch, source_commit_id,
                     input_revision)).fetchone()
                if row:
                    return row["request_id"]
            seq = (self._conn.execute(
                "SELECT COALESCE(MAX(command_seq),0)+1 AS n FROM requests "
                "WHERE session_id=?", (session_id,)).fetchone())["n"]
            request_id = f"req-{uuid.uuid4().hex[:12]}"
            self._conn.execute(
                "INSERT INTO requests(request_id, session_id, command_seq, "
                "source_commit_id, engine_epoch, input_revision, text, "
                "created_at) VALUES(?,?,?,?,?,?,?,?)",
                (request_id, session_id, seq, source_commit_id, engine_epoch,
                 input_revision, text, self._now()))
            return request_id

    def request_row(self, request_id):
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM requests WHERE request_id=?",
                (request_id,)).fetchone()

    # ------------------------------------------------------------- 任务快照

    def task_view(self, session_id):
        """路由器可见的任务快照（§7.1）。只读。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT t.*, r.goal, r.constraints_json, r.depends_json "
                "FROM tasks t JOIN task_revisions r "
                "ON r.session_id=t.session_id AND r.task_id=t.task_id "
                "AND r.revision=t.current_revision "
                "WHERE t.session_id=? ORDER BY t.created_at",
                (session_id,)).fetchall()
            out = []
            for t in rows:
                out.append({
                    "task_id": t["task_id"], "label": t["label"],
                    "revision": t["current_revision"],
                    "execution_status": t["execution_status"],
                    "active_run_id": t["active_run_id"],
                    "cancel_intent": bool(t["cancel_intent"]),
                    "goal": t["goal"],
                    "constraints": json.loads(t["constraints_json"]),
                    "depends": json.loads(t["depends_json"]),
                    "updated_at": t["updated_at"],
                })
            return out

    def task_row(self, session_id, task_id):
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM tasks WHERE session_id=? AND task_id=?",
                (session_id, task_id)).fetchone()

    def revision_row(self, session_id, task_id, revision):
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM task_revisions WHERE session_id=? AND "
                "task_id=? AND revision=?",
                (session_id, task_id, revision)).fetchone()

    def task_panel_rows(self, session_id):
        """前端任务面板只读视图（V2）：任务 + 最近进度/结果 + 结果是否
        已完整播出。供 adapter 的 UI 泵轮询——读侧专用，不进控制平面。"""
        with self._lock:
            tasks = self._conn.execute(
                "SELECT t.task_id, t.label, t.execution_status, r.goal "
                "FROM tasks t JOIN task_revisions r "
                "ON r.session_id=t.session_id AND r.task_id=t.task_id "
                "AND r.revision=t.current_revision "
                "WHERE t.session_id=? ORDER BY t.created_at DESC",
                (session_id,)).fetchall()
            progress, result, delivered = {}, {}, {}
            for e in self._conn.execute(
                    "SELECT task_id, type, payload_json FROM events WHERE "
                    "session_id=? AND task_id IS NOT NULL AND type IN "
                    "('progress','completed','failed') ORDER BY seq",
                    (session_id,)).fetchall():
                try:
                    p = json.loads(e["payload_json"])
                except (ValueError, TypeError):
                    p = {}
                if e["type"] == "progress":
                    progress[e["task_id"]] = str(p.get("desc", ""))[:120]
                else:
                    result[e["task_id"]] = str(p.get("summary", ""))[:400]
            for d in self._conn.execute(
                    "SELECT task_id, state FROM deliveries WHERE "
                    "session_id=? AND kind='result' AND task_id IS NOT NULL "
                    "ORDER BY rowid", (session_id,)).fetchall():
                delivered[d["task_id"]] = d["state"] == C.DLV_PLAYED_CONFIRMED
            return [{
                "task_id": t["task_id"], "goal": t["goal"] or t["label"],
                "execution_status": t["execution_status"],
                "progress": progress.get(t["task_id"], ""),
                "result": result.get(t["task_id"], ""),
                "delivered": delivered.get(t["task_id"], False),
            } for t in tasks]

    # -------------------------------------------------------------- 事件

    def _append_event(self, session_id, etype, task_id=None, revision=None,
                      run_id=None, payload=None):
        """必须已在 _tx 内调用。返回 event_id。"""
        seq = (self._conn.execute(
            "SELECT event_seq+1 AS n FROM sessions WHERE session_id=?",
            (session_id,)).fetchone())["n"]
        # seq 是会话内序号；event_id 全局唯一须带 session 前缀
        event_id = f"{session_id}:E{seq}"
        self._conn.execute(
            "INSERT INTO events(event_id, session_id, seq, task_id, "
            "revision, run_id, type, payload_json, created_at) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (event_id, session_id, seq, task_id, revision, run_id, etype,
             json.dumps(payload or {}, ensure_ascii=False), self._now()))
        self._conn.execute(
            "UPDATE sessions SET event_seq=? WHERE session_id=?",
            (seq, session_id))
        return event_id

    def events_for_task(self, session_id, task_id, limit=50):
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM events WHERE session_id=? AND task_id=? "
                "ORDER BY seq DESC LIMIT ?", (session_id, task_id, limit))
            return [dict(r) for r in rows]

    # -------------------------------------------------------------- 提交计划

    def commit_plan(self, session_id, request_id, plan, plan_version=1,
                    _fail_after=None):
        """应用一个已通过 parse_plan 的计划（手册 §8）。

        - 幂等：同一 (request_id, plan_version) 重放 → 返回旧回执，
          replayed=True，零新写入。
        - 原子：批内任何 StoreError → 全部回滚（含 operations/events/outbox）。
        - 语义拒绝（unknown_task/stale_revision/queue_full）不回滚整批：
          该 op 得 rejected 回执，其余 op 照常提交（显式逐项结果，§8）。
        - _fail_after：仅测试用——处理完第 N 个 op 后抛错，模拟崩溃窗口。
        """
        with self._tx():
            return self._commit_plan_locked(
                session_id, request_id, plan, plan_version, _fail_after)

    def _commit_plan_locked(self, session_id, request_id, plan, plan_version,
                            _fail_after):
        op_ids = [C.operation_id_for(request_id, plan_version, i)
                  for i in range(len(plan.operations))]
        # 幂等重放：原子提交保证 operation_id 全存在或全不存在
        done = self._conn.execute(
            "SELECT COUNT(*) AS n FROM operations WHERE operation_id IN "
            "(%s)" % ",".join("?" * len(op_ids)), op_ids).fetchone()["n"]
        if done:
            if done != len(op_ids):
                raise StoreError("partial operation commit detected")
            receipts, queries = [], []
            for oid in op_ids:
                r = self._conn.execute(
                    "SELECT receipt_json, op_type, task_id FROM operations "
                    "WHERE operation_id=?", (oid,)).fetchone()
                d = json.loads(r["receipt_json"])
                receipts.append(C.Receipt(**d))
            for o, oid in zip(plan.operations, op_ids):
                if isinstance(o, C.RuntimeQuery):
                    queries.append({"query": o.query, "value": self._now()})
            return CommitResult(receipts, {}, queries, replayed=True)

        # 批内 create 的 local_ref → 预分配 task_id（依赖解析需要）
        bindings = {}
        for o in plan.operations:
            if isinstance(o, (C.CreateTask, C.FollowupTask)) and o.local_ref:
                bindings[o.local_ref] = self._next_task_id(session_id)

        # 批内依赖成环检查（手册 §10：创建时拒绝环）
        self._check_batch_cycles(plan, bindings)

        receipts, queries = [], []
        for idx, (o, oid) in enumerate(zip(plan.operations, op_ids)):
            receipt = self._apply_op(session_id, request_id, o, oid, idx,
                                     bindings)
            receipts.append(receipt)
            if isinstance(o, C.RuntimeQuery):
                queries.append({"query": o.query, "value": self._now()})
            if _fail_after is not None and idx == _fail_after:
                raise StoreError(f"injected crash after op {idx}")

        self._conn.execute(
            "UPDATE requests SET state='committed' WHERE request_id=?",
            (request_id,))
        return CommitResult(receipts, bindings, queries, replayed=False)

    def _next_task_id(self, session_id):
        seq = (self._conn.execute(
            "SELECT task_seq+1 AS n FROM sessions WHERE session_id=?",
            (session_id,)).fetchone())["n"]
        self._conn.execute(
            "UPDATE sessions SET task_seq=? WHERE session_id=?",
            (seq, session_id))
        return f"T{seq}"

    def _check_batch_cycles(self, plan, bindings):
        # 批内 create 之间的依赖构成 DAG；引用既有任务不会成环（新任务
        # 尚无依赖者）。广度遍历检测。
        graph = {}
        for o in plan.operations:
            if isinstance(o, C.CreateTask) and o.local_ref:
                graph[o.local_ref] = list(o.depends_on_local_refs)
            elif isinstance(o, C.FollowupTask) and o.local_ref:
                graph[o.local_ref] = []  # followup 依赖既有任务，无批内依赖
        color = {}  # 0=visiting 1=done

        def visit(node):
            if color.get(node) == 0:
                raise StoreError(f"dependency cycle at {node}")
            if color.get(node) == 1:
                return
            color[node] = 0
            for d in graph.get(node, []):
                visit(d)
            color[node] = 1

        for n in graph:
            visit(n)

    def _apply_op(self, session_id, request_id, o, oid, idx, bindings):
        """单个 op 的事务内应用。返回 Receipt。语义拒绝返回 rejected 回执，
        不抛异常。"""
        if isinstance(o, C.CreateTask):
            return self._op_create(session_id, request_id, o, oid, idx,
                                   bindings, kind=C.RC_ACCEPTED)
        if isinstance(o, C.FollowupTask):
            parent = self.task_row(session_id, o.parent_task_id)
            if parent is None:
                return self._reject(session_id, request_id, o, oid, idx,
                                    reason="unknown_task",
                                    task_id=o.parent_task_id)
            return self._op_create(session_id, request_id, o, oid, idx,
                                   bindings, kind=C.RC_ACCEPTED,
                                   depends=[{"task_id": o.parent_task_id,
                                             "revision": o.parent_revision}])
        if isinstance(o, C.AmendTask):
            return self._op_amend(session_id, request_id, o, oid, idx)
        if isinstance(o, C.RerunTask):
            return self._op_rerun(session_id, request_id, o, oid, idx)
        if isinstance(o, C.CancelTasks):
            return self._op_cancel(session_id, request_id, o, oid, idx)
        if isinstance(o, C.GetTask):
            return self._op_get(session_id, request_id, o, oid, idx)
        if isinstance(o, C.AnswerTaskQuestion):
            return self._op_answer(session_id, request_id, o, oid, idx)
        if isinstance(o, C.RuntimeQuery):
            # 只读；值由上层用回执/queries 通道给表达轮
            r = C.Receipt(operation_id=oid, op=o.op, outcome=C.RC_INFO,
                          reason_code=o.query)
            self._record_op(session_id, request_id, o, oid, idx, r)
            return r
        if isinstance(o, C.StopSpeech):
            self._enqueue(session_id, "stop_speech",
                          {"operation_id": oid})
            r = C.Receipt(operation_id=oid, op=o.op, outcome=C.RC_INFO)
            self._record_op(session_id, request_id, o, oid, idx, r)
            return r
        raise StoreError(f"unhandled op {o.op}")

    # -- create / followup ------------------------------------------------

    def _op_create(self, session_id, request_id, o, oid, idx, bindings, kind,
                   depends=None):
        dep = depends if depends is not None else [
            {"task_id": bindings[d], "revision": 1}
            for d in getattr(o, "depends_on_local_refs", [])]
        # 上游依赖必须存在（task_id 引用的存在性由 runtime 对快照校验，§7.3）
        for d in dep:
            if self.revision_row(session_id, d["task_id"], d["revision"]) is None:
                return self._reject(session_id, request_id, o, oid, idx,
                                    reason="unknown_task", task_id=d["task_id"])
        # 有界队列：queued+waiting+cancel_requested 计入待执行（§8）
        active = self._conn.execute(
            "SELECT COUNT(*) AS n FROM tasks WHERE session_id=? AND "
            "execution_status IN ('queued','waiting_input')",
            (session_id,)).fetchone()["n"]
        if active >= self.max_queued:
            return self._reject(session_id, request_id, o, oid, idx,
                                reason="queue_full")
        task_id = bindings.get(getattr(o, "local_ref", ""), None) or \
            self._next_task_id(session_id)
        # FollowupTask 没有独立 constraints，继承父任务修订的约束
        constraints = getattr(o, "constraints", None)
        if constraints is None and isinstance(o, C.FollowupTask):
            prev = self.revision_row(session_id, o.parent_task_id,
                                     o.parent_revision)
            constraints = json.loads(prev["constraints_json"]) if prev else {}
        now = self._now()
        self._conn.execute(
            "INSERT INTO tasks(session_id, task_id, label, current_revision, "
            "execution_status, created_at, updated_at) "
            "VALUES(?,?,?,?,?,?,?)",
            (session_id, task_id, o.label, 1, C.EXEC_QUEUED, now, now))
        self._conn.execute(
            "INSERT INTO task_revisions(session_id, task_id, revision, goal, "
            "constraints_json, depends_json, source_request_id, user_text, "
            "created_at) VALUES(?,?,?,?,?,?,?,?,?)",
            (session_id, task_id, 1, o.goal,
             json.dumps(constraints, ensure_ascii=False),
             json.dumps(dep, ensure_ascii=False), request_id,
             getattr(o, "evidence", ""), now))
        self._append_event(session_id, "task_created", task_id=task_id,
                           revision=1,
                           payload={"goal": o.goal, "label": o.label})
        self._enqueue(session_id, "schedule", {
            "task_id": task_id, "revision": 1, "operation_id": oid})
        r = C.Receipt(operation_id=oid, op=o.op, task_id=task_id, revision=1,
                      outcome=kind)
        self._record_op(session_id, request_id, o, oid, idx, r,
                        task_id=task_id)
        return r

    # -- amend / rerun -----------------------------------------------------

    def _op_amend(self, session_id, request_id, o, oid, idx):
        t = self.task_row(session_id, o.target_task_id)
        if t is None:
            return self._reject(session_id, request_id, o, oid, idx,
                                reason="unknown_task",
                                task_id=o.target_task_id)
        if t["current_revision"] != o.expected_revision:
            return self._reject(session_id, request_id, o, oid, idx,
                                reason="stale_revision",
                                task_id=o.target_task_id,
                                current=t["execution_status"])
        return self._new_revision(session_id, request_id, o, oid, idx,
                                  t, patch=o.patch,
                                  outcome=C.RC_UPDATED, etype="task_amended")

    def _op_rerun(self, session_id, request_id, o, oid, idx):
        t = self.task_row(session_id, o.target_task_id)
        if t is None:
            return self._reject(session_id, request_id, o, oid, idx,
                                reason="unknown_task",
                                task_id=o.target_task_id)
        if t["current_revision"] != o.expected_revision:
            return self._reject(session_id, request_id, o, oid, idx,
                                reason="stale_revision",
                                task_id=o.target_task_id,
                                current=t["execution_status"])
        return self._new_revision(session_id, request_id, o, oid, idx, t,
                                  patch=None, outcome=C.RC_ACCEPTED,
                                  etype="task_rerun",
                                  reason=o.refresh_reason)

    def _new_revision(self, session_id, request_id, o, oid, idx, t, patch,
                      outcome, etype, reason=""):
        """§9.1 amend 原子行为：新 revision 快照 → 废止旧 run 输出资格 →
        抑制旧待播 delivery → queued/outbox → 提交。"""
        old_rev = t["current_revision"]
        old = self.revision_row(session_id, t["task_id"], old_rev)
        constraints = json.loads(old["constraints_json"])
        if patch:
            for k, v in patch.items():
                if k == "constraints" and isinstance(v, dict):
                    constraints.update(v)
                else:
                    constraints[k] = v
        new_rev = old_rev + 1
        now = self._now()
        # 旧 run：记录停止请求，保留审计状态（不删除）
        if t["active_run_id"]:
            self._conn.execute(
                "UPDATE runs SET state='stop_requested', stop_reason=? "
                "WHERE run_id=? AND state IN ('queued','running')",
                (etype, t["active_run_id"]))
            self._enqueue(session_id, "cancel_run", {
                "run_id": t["active_run_id"], "reason": etype})
        # 抑制旧 revision 的待播 delivery（不把历史执行改成 canceled）
        self._suppress_deliveries(session_id, t["task_id"], old_rev, lte=True)
        self._conn.execute(
            "INSERT INTO task_revisions(session_id, task_id, revision, goal, "
            "constraints_json, depends_json, source_request_id, user_text, "
            "created_at) VALUES(?,?,?,?,?,?,?,?,?)",
            (session_id, t["task_id"], new_rev, old["goal"],
             json.dumps(constraints, ensure_ascii=False),
             old["depends_json"], request_id,
             getattr(o, "evidence", ""), now))
        self._conn.execute(
            "UPDATE tasks SET current_revision=?, execution_status=?, "
            "active_run_id=NULL, cancel_intent=0, pending_question_id='', "
            "pending_question='', updated_at=? WHERE "
            "session_id=? AND task_id=?",
            (new_rev, C.EXEC_QUEUED, now, session_id, t["task_id"]))
        self._append_event(session_id, etype, task_id=t["task_id"],
                           revision=new_rev,
                           payload={"from_revision": old_rev,
                                    "patch": patch or {},
                                    "reason": reason})
        self._enqueue(session_id, "schedule", {
            "task_id": t["task_id"], "revision": new_rev,
            "operation_id": oid})
        r = C.Receipt(operation_id=oid, op=o.op, task_id=t["task_id"],
                      revision=new_rev, outcome=outcome)
        self._record_op(session_id, request_id, o, oid, idx, r,
                        task_id=t["task_id"])
        return r

    # -- cancel ------------------------------------------------------------

    def _op_cancel(self, session_id, request_id, o, oid, idx):
        targets = o.targets
        if o.scope == "all_active":
            targets = []
            for t in self.task_view(session_id):
                if t["execution_status"] not in (C.EXEC_CANCELED,
                                                 C.EXEC_SUPERSEDED):
                    targets.append({"task_id": t["task_id"],
                                    "expected_revision": t["revision"]})
        if not targets:
            return self._reject(session_id, request_id, o, oid, idx,
                                reason="unknown_task")
        outcomes = []
        for tgt in targets:
            t = self.task_row(session_id, tgt["task_id"])
            if t is None:
                outcomes.append("unknown_task")
                continue
            if t["current_revision"] != tgt["expected_revision"]:
                outcomes.append("stale_revision")
                continue
            st = t["execution_status"]
            if st in (C.EXEC_COMPLETED, C.EXEC_FAILED):
                # §9.2 第三层：执行是事实，只抑制播报。cancel_intent=1
                # 让交付泵的信封校验也拒掉这条结果——2026-11-06 湖人纠错
                # 事故：抑制只扫了 deliveries 表，而交付行要到 can_deliver
                # 通过后才创建（delivery.py:158），结果还躺在 outbox 里，
                # 照播不误。amend/rerun（_new_revision）与 get_task 重播
                # （_redeliver_unheard）都会把标记清回来。
                self._conn.execute(
                    "UPDATE tasks SET cancel_intent=1, updated_at=? WHERE "
                    "session_id=? AND task_id=?",
                    (self._now(), session_id, tgt["task_id"]))
                self._suppress_deliveries(session_id, tgt["task_id"],
                                          t["current_revision"])
                self._append_event(session_id, "delivery_suppressed",
                                   task_id=tgt["task_id"],
                                   revision=t["current_revision"],
                                   payload={"operation_id": oid})
                outcomes.append(C.RC_ALREADY_COMPLETED)
                continue
            has_run = False
            if t["active_run_id"]:
                run = self._conn.execute(
                    "SELECT state FROM runs WHERE run_id=?",
                    (t["active_run_id"],)).fetchone()
                has_run = run is not None and run["state"] in (
                    "queued", "running", "stop_requested")
            now = self._now()
            if st == C.EXEC_QUEUED and not has_run:
                # 确认尚无 worker/工具在运行：原子撤销调度直接 canceled
                self._conn.execute(
                    "UPDATE tasks SET execution_status=?, updated_at=? "
                    "WHERE session_id=? AND task_id=?",
                    (C.EXEC_CANCELED, now, session_id, tgt["task_id"]))
                self._suppress_deliveries(session_id, tgt["task_id"],
                                          t["current_revision"])
                self._append_event(session_id, "task_canceled",
                                   task_id=tgt["task_id"],
                                   revision=t["current_revision"],
                                   payload={"operation_id": oid})
                outcomes.append(C.RC_CANCELED)
            else:
                # running/waiting_input：先写 cancel_requested，等停止确认
                self._conn.execute(
                    "UPDATE tasks SET cancel_intent=1, execution_status=?, "
                    "updated_at=? WHERE session_id=? AND task_id=?",
                    (C.EXEC_CANCEL_REQUESTED, now, session_id, tgt["task_id"]))
                if t["active_run_id"]:
                    self._conn.execute(
                        "UPDATE runs SET state='stop_requested' WHERE "
                        "run_id=? AND state IN ('queued','running')",
                        (t["active_run_id"],))
                    self._enqueue(session_id, "cancel_run", {
                        "run_id": t["active_run_id"], "reason": "user_cancel",
                        "operation_id": oid})
                self._append_event(session_id, "cancel_requested",
                                   task_id=tgt["task_id"],
                                   revision=t["current_revision"],
                                   payload={"operation_id": oid})
                outcomes.append(C.RC_CANCEL_REQUESTED)
        # 回执取最保守的综合结果（逐 target 结果都在 events/operations 里）
        if all(x == C.RC_CANCELED for x in outcomes):
            agg = C.RC_CANCELED
        elif all(x == C.RC_ALREADY_COMPLETED for x in outcomes):
            agg = C.RC_ALREADY_COMPLETED
        elif outcomes and all(x in (C.RC_CANCELED,
                                    C.RC_ALREADY_COMPLETED) for x in outcomes):
            agg = C.RC_ALREADY_COMPLETED
        elif any(x in (C.RC_CANCEL_REQUESTED, "unknown_task",
                       "stale_revision") for x in outcomes):
            agg = C.RC_CANCEL_REQUESTED if C.RC_CANCEL_REQUESTED in outcomes \
                else C.RC_REJECTED
        else:
            agg = C.RC_REJECTED
        reason = ""
        if "unknown_task" in outcomes:
            reason = "unknown_task"
        elif "stale_revision" in outcomes:
            reason = "stale_revision"
        r = C.Receipt(operation_id=oid, op=o.op,
                      task_id=targets[0]["task_id"] if len(targets) == 1
                      else None,
                      outcome=agg, reason_code=reason)
        self._record_op(session_id, request_id, o, oid, idx, r,
                        task_id=targets[0]["task_id"] if len(targets) == 1
                        else None)
        return r

    # -- get / answer --------------------------------------------------------

    def _op_get(self, session_id, request_id, o, oid, idx):
        if o.target_task_id:
            t = self.task_row(session_id, o.target_task_id)
            if t is None:
                return self._reject(session_id, request_id, o, oid, idx,
                                    reason="unknown_task",
                                    task_id=o.target_task_id)
            rev = self.revision_row(session_id, o.target_task_id,
                                    t["current_revision"])
            r = C.Receipt(operation_id=oid, op=o.op,
                          task_id=o.target_task_id,
                          revision=t["current_revision"], outcome=C.RC_INFO,
                          current_state=t["execution_status"])
            # 重播（2026-11-06 待办 → 2026-11-07 单一出口重构）：结果已在
            # → 显式追问重排 deliver outbox 并重开交付行（含已完整听到的
            # played_confirmed——闲聊路不再代播结果，"再讲一遍"只能由交付
            # 泵兑现），回执改说"再说一遍"。在途交付不双排。用户显式追问
            # ≠ 自动重试，不违反"打断是终态"。
            if (t["execution_status"] in (C.EXEC_COMPLETED,
                                          C.EXEC_WAITING_INPUT)
                    and self._redeliver_unheard(session_id, t)):
                r.reason_code = "redelivering"
        else:
            tasks = self.task_view(session_id)
            if o.list_scope == "active":
                tasks = [t for t in tasks if t["execution_status"] in (
                    C.EXEC_QUEUED, C.EXEC_RUNNING, C.EXEC_WAITING_INPUT,
                    C.EXEC_CANCEL_REQUESTED)]
            r = C.Receipt(operation_id=oid, op=o.op, outcome=C.RC_INFO)
        self._record_op(session_id, request_id, o, oid, idx, r)
        return r

    def _op_answer(self, session_id, request_id, o, oid, idx):
        t = self.task_row(session_id, o.target_task_id)
        if t is None:
            return self._reject(session_id, request_id, o, oid, idx,
                                reason="unknown_task",
                                task_id=o.target_task_id)
        if t["current_revision"] != o.expected_revision:
            return self._reject(session_id, request_id, o, oid, idx,
                                reason="stale_revision",
                                task_id=o.target_task_id)
        if t["execution_status"] != C.EXEC_WAITING_INPUT:
            return self._reject(session_id, request_id, o, oid, idx,
                                reason="not_waiting_input",
                                task_id=o.target_task_id,
                                current=t["execution_status"])
        if o.question_id != t["pending_question_id"]:
            return self._reject(session_id, request_id, o, oid, idx,
                                reason="unknown_question",
                                task_id=o.target_task_id,
                                current=t["execution_status"])
        return self._new_revision(
            session_id, request_id, o, oid, idx, t,
            patch={"answered_question": {
                "question_id": o.question_id, "answer": o.answer}},
            outcome=C.RC_ANSWERED, etype="question_answered")

    # -- 内部小工具 ----------------------------------------------------------

    def _reject(self, session_id, request_id, o, oid, idx, reason,
                task_id=None, current=""):
        r = C.Receipt(operation_id=oid, op=o.op, task_id=task_id,
                      outcome=C.RC_REJECTED, reason_code=reason,
                      current_state=current)
        self._record_op(session_id, request_id, o, oid, idx, r,
                        task_id=task_id)
        return r

    def _record_op(self, session_id, request_id, o, oid, idx, receipt,
                   task_id=None):
        self._conn.execute(
            "INSERT INTO operations(operation_id, request_id, op_index, "
            "op_type, args_json, outcome, reason_code, task_id, receipt_json, "
            "committed_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (oid, request_id or "", idx, o.op,
             json.dumps(getattr(o, "__dict__", {}), ensure_ascii=False,
                        default=str),
             receipt.outcome, receipt.reason_code, task_id,
             json.dumps(receipt.to_dict(), ensure_ascii=False), self._now()))

    def _enqueue(self, session_id, kind, payload):
        self._conn.execute(
            "INSERT INTO outbox(session_id, kind, payload_json, created_at) "
            "VALUES(?,?,?,?)",
            (session_id, kind,
             json.dumps(payload, ensure_ascii=False), self._now()))

    # ------------------------------------------------------------- deliveries

    def _suppress_deliveries(self, session_id, task_id, revision,
                              lte=False):
        """必须已在 _tx 内。抑制该 task/revision 的未播 delivery；
        lte=True 时抑制所有 ≤ revision 的历史待播（amend/rerun 用）。"""
        op = "<=?" if lte else "=?"
        self._conn.execute(
            "UPDATE deliveries SET state=?, updated_at=? WHERE session_id=? "
            "AND task_id=? AND revision" + op + " AND state IN ('pending',"
            "'rendering','sending')",
            (C.DLV_SUPPRESSED, self._now(), session_id, task_id, revision))

    def _reopen_delivery(self, delivery_id):
        """必须已在 _tx 内。把已完成交付重置为 pending 并清零重试计数
        （get_task 显式追问重播）。原子 WHERE 守卫：状态不在重开
        白名单（在途状态）时静默不动。"""
        self._conn.execute(
            "UPDATE deliveries SET state=?, retries=0, updated_at=? "
            "WHERE delivery_id=? AND state IN (?,?,?,?,?,?)",
            (C.DLV_PENDING, self._now(), delivery_id,
             C.DLV_INTERRUPTED_PARTIAL, C.DLV_INTERRUPTED_BEFORE_AUDIO,
             C.DLV_RETRY_EXHAUSTED, C.DLV_SUPPRESSED,
             C.DLV_SENT_UNCONFIRMED, C.DLV_PLAYED_CONFIRMED))

    def _redeliver_unheard(self, session_id, t):
        """必须已在 _tx 内。任务已有结果 → 显式追问（get_task）重排一条
        deliver outbox（复用最近一条已完成 deliver 的原文 payload）并
        重开交付行。返回 True 表示已重排（回执应说"再说一遍"）。
        在途交付（outbox 未消费 / delivery pending-rendering-sending）会让
        结果自然到达，不重排；已听到（played_confirmed /
        sent_unconfirmed）也重排——结果播报的唯一出口是交付泵，用户
        "再讲一遍"只能由它兑现（2026-11-07 单一出口重构）。"""
        task_id = t["task_id"]
        for row in self._conn.execute(
                "SELECT payload_json FROM outbox WHERE session_id=? AND "
                "kind='deliver' AND state IN ('pending','leased')",
                (session_id,)).fetchall():
            try:
                if json.loads(row["payload_json"]).get("task_id") == task_id:
                    return False
            except (ValueError, TypeError):
                continue
        dlv = self._conn.execute(
            "SELECT delivery_id, state FROM deliveries WHERE session_id=? "
            "AND kind='result' AND task_id=? ORDER BY rowid DESC LIMIT 1",
            (session_id, task_id)).fetchone()
        if dlv is not None and dlv["state"] not in _DLV_REOPEN_FROM:
            return False
        src = None
        for row in self._conn.execute(
                "SELECT payload_json FROM outbox WHERE session_id=? AND "
                "kind='deliver' AND state='done' ORDER BY outbox_id DESC",
                (session_id,)).fetchall():
            try:
                p = json.loads(row["payload_json"])
            except (ValueError, TypeError):
                continue
            if p.get("task_id") == task_id:
                src = p
                break
        if src is None:
            return False   # 无可重播的结果记录（如测试直改任务状态）
        if dlv is not None:
            self._reopen_delivery(dlv["delivery_id"])
        # 用户显式追问 = 重新要这个结果：清掉改口标记，否则交付泵的
        # 信封校验会把重排的 deliver 又拒掉（2026-11-06 湖人纠错事故的
        # 对称面——"结果先不重复说，你需要的话我再讲"要能兑现）
        self._conn.execute(
            "UPDATE tasks SET cancel_intent=0, updated_at=? WHERE "
            "session_id=? AND task_id=?",
            (self._now(), session_id, task_id))
        self._enqueue(session_id, "deliver", src)
        return True

    def record_delivery(self, session_id, kind, task_id=None, revision=None,
                        run_id=None, operation_id=None, speech_id=None,
                        generation_id=None, dependency_versions=None):
        """登记一个逻辑交付身份；重复登记（重试）返回原 delivery_id。"""
        delivery_id = "dlv-" + uuid.uuid4().hex[:12]
        with self._tx():
            try:
                self._conn.execute(
                    "INSERT INTO deliveries(delivery_id, session_id, kind, "
                    "task_id, revision, run_id, operation_id, speech_id, "
                    "generation_id, dependency_versions_json, updated_at) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (delivery_id, session_id, kind, task_id, revision, run_id,
                     operation_id, speech_id, generation_id,
                     json.dumps(dependency_versions or {},
                                ensure_ascii=False), self._now()))
            except sqlite3.IntegrityError:
                row = self._conn.execute(
                    "SELECT delivery_id FROM deliveries WHERE session_id=? "
                    "AND kind=? AND ifnull(task_id,'')=? AND "
                    "ifnull(operation_id,'')=? AND ifnull(run_id,'')=?",
                    (session_id, kind, task_id or "", operation_id or "",
                     run_id or "")).fetchone()
                if row is None:
                    raise
                return row["delivery_id"]
            return delivery_id

    _ALL_DLV_STATES = frozenset(_DLV_TRANSITIONS) | {
        C.DLV_PLAYED_CONFIRMED, C.DLV_INTERRUPTED_PARTIAL,
        C.DLV_INTERRUPTED_BEFORE_AUDIO, C.DLV_SUPPRESSED,
        C.DLV_RETRY_EXHAUSTED}

    def set_delivery_state(self, delivery_id, state):
        if state not in self._ALL_DLV_STATES:
            raise StoreError(f"unknown delivery state {state}")
        with self._tx():
            row = self._conn.execute(
                "SELECT state FROM deliveries WHERE delivery_id=?",
                (delivery_id,)).fetchone()
            if row is None:
                raise StoreError(f"delivery {delivery_id} not found")
            cur = row["state"]
            allowed = _DLV_TRANSITIONS.get(cur, set())
            if state not in allowed:
                raise StoreError(
                    f"illegal delivery transition {cur} -> {state}")
            self._conn.execute(
                "UPDATE deliveries SET state=?, updated_at=? WHERE "
                "delivery_id=?", (state, self._now(), delivery_id))

    def bump_delivery_retry(self, delivery_id):
        with self._tx():
            self._conn.execute(
                "UPDATE deliveries SET retries=retries+1, updated_at=? WHERE "
                "delivery_id=?", (self._now(), delivery_id))

    def delivery_row(self, delivery_id):
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM deliveries WHERE delivery_id=?",
                (delivery_id,)).fetchone()

    # ----------------------------------------------------------------- runs

    def create_run(self, session_id, task_id, revision, snapshot=None,
                   attempt=1):
        run_id = f"run-{uuid.uuid4().hex[:10]}"
        with self._tx():
            t = self.task_row(session_id, task_id)
            if t is None or t["current_revision"] != revision:
                raise StoreError(
                    f"create_run: {task_id}@{revision} 不再是当前 revision")
            self._conn.execute(
                "INSERT INTO runs(run_id, session_id, task_id, revision, "
                "attempt, snapshot_json) VALUES(?,?,?,?,?,?)",
                (run_id, session_id, task_id, revision, attempt,
                 json.dumps(snapshot or {}, ensure_ascii=False,
                            default=str)))
            self._conn.execute(
                "UPDATE tasks SET active_run_id=?, execution_status=?, "
                "updated_at=? WHERE session_id=? AND task_id=?",
                (run_id, C.EXEC_RUNNING, self._now(), session_id, task_id))
            self._append_event(session_id, "run_started", task_id=task_id,
                               revision=revision, run_id=run_id)
            return run_id

    def lease_run(self, run_id, ttl=RUN_LEASE_TTL_S):
        """条件更新：同一时刻只有一个有效 lease（§5.2）。"""
        token = uuid.uuid4().hex
        with self._tx():
            cur = self._conn.execute(
                "SELECT state, lease_token, lease_expires_at FROM runs "
                "WHERE run_id=?", (run_id,)).fetchone()
            if cur is None or cur["state"] not in ("queued", "running"):
                return None
            if (cur["lease_token"] and cur["lease_expires_at"]
                    and cur["lease_expires_at"] > time.time()):
                return None
            self._conn.execute(
                "UPDATE runs SET lease_token=?, lease_expires_at=?, "
                "state='running', started_at=COALESCE(started_at,?) WHERE "
                "run_id=?", (token, time.time() + ttl, self._now(), run_id))
            return token

    def heartbeat_run(self, run_id, token, ttl=RUN_LEASE_TTL_S):
        with self._tx():
            cur = self._conn.execute(
                "SELECT lease_token FROM runs WHERE run_id=?",
                (run_id,)).fetchone()
            if cur is None or cur["lease_token"] != token:
                return False
            self._conn.execute(
                "UPDATE runs SET lease_expires_at=? WHERE run_id=?",
                (time.time() + ttl, run_id))
            return True

    def run_is_current(self, session_id, task_id, run_id, revision):
        """围栏判据：run 必须仍是该任务当前 revision 的 active run。"""
        with self._lock:
            t = self.task_row(session_id, task_id)
            if t is None or t["current_revision"] != revision:
                return False
            return t["active_run_id"] == run_id

    def finish_run(self, session_id, task_id, revision, run_id, token,
                   state, result_summary="", payload=None):
        """结束 run。带围栏：若 run 已不是当前 revision 的活跃 run
        （被 amend/cancel 抢先），结果只写 runs/events 审计，绝不更新
        tasks 当前状态（§9.3 过期 run 事件只进历史）。"""
        if state not in (C.EXEC_COMPLETED, C.EXEC_FAILED,
                         C.EXEC_WAITING_INPUT, C.EXEC_CANCELED,
                         "interrupted"):
            raise StoreError(f"invalid finish state {state}")
        with self._tx():
            cur = self._conn.execute(
                "SELECT lease_token, state FROM runs WHERE run_id=?",
                (run_id,)).fetchone()
            if cur is None:
                raise StoreError(f"run {run_id} not found")
            if cur["lease_token"] != token:
                raise StoreError("run lease lost; 结果只能丢弃")
            fenced = not self.run_is_current(session_id, task_id, run_id,
                                             revision)
            now = self._now()
            self._conn.execute(
                "UPDATE runs SET state=?, ended_at=?, stop_reason=? WHERE "
                "run_id=?",
                ("canceled" if state == C.EXEC_CANCELED else state, now,
                 "fenced_stale_run" if fenced else "", run_id))
            if fenced:
                # 审计事件，不触碰任务当前状态
                self._append_event(session_id, "stale_run_finished",
                                   task_id=task_id, revision=revision,
                                   run_id=run_id,
                                   payload={"state": state})
                return "fenced"
            pq_id = (payload or {}).get("question_id", "")
            pq = (payload or {}).get("question", "")
            if state == C.EXEC_WAITING_INPUT:
                self._conn.execute(
                    "UPDATE tasks SET execution_status=?, active_run_id=NULL, "
                    "pending_question_id=?, pending_question=?, updated_at=? "
                    "WHERE session_id=? AND task_id=?",
                    (state, str(pq_id), str(pq), now, session_id, task_id))
            else:
                self._conn.execute(
                    "UPDATE tasks SET execution_status=?, active_run_id=NULL, "
                    "pending_question_id='', pending_question='', updated_at=? "
                    "WHERE session_id=? AND task_id=?",
                    (state, now, session_id, task_id))
            self._append_event(
                session_id, state if state != "interrupted" else "run_stopped",
                task_id=task_id, revision=revision, run_id=run_id,
                payload={"summary": result_summary,
                         "detail": payload or {}})
            return state

    # ---------------------------------------------------------------- outbox

    def claim_outbox(self, limit=1, lease_ttl=OUTBOX_LEASE_TTL_S,
                     session_id=None):
        """领取待处理条目（at-least-once；消费方自幂等）。"""
        token = uuid.uuid4().hex
        claimed = []
        with self._tx():
            now = time.time()
            q = ("SELECT outbox_id FROM outbox WHERE (state='pending' OR "
                 "(state='leased' AND lease_expires_at<=?)) "
                 "AND (lease_expires_at IS NULL OR lease_expires_at<=?)")
            args = [now, now]
            if session_id:
                q += " AND session_id=?"
                args.append(session_id)
            q += " ORDER BY outbox_id LIMIT ?"
            args.append(limit)
            rows = self._conn.execute(q, args).fetchall()
            for r in rows:
                self._conn.execute(
                    "UPDATE outbox SET state='leased', lease_token=?, "
                    "lease_expires_at=?, attempts=attempts+1 WHERE "
                    "outbox_id=? AND (state='pending' OR "
                    "(state='leased' AND lease_expires_at<=?))",
                    (token, now + lease_ttl, r["outbox_id"], now))
                claimed.append((r["outbox_id"], token))
        out = []
        for oid, tok in claimed:
            row = self._conn.execute(
                "SELECT * FROM outbox WHERE outbox_id=?", (oid,)).fetchone()
            out.append({"outbox_id": oid, "token": tok,
                        "kind": row["kind"],
                        "payload": json.loads(row["payload_json"]),
                        "attempts": row["attempts"]})
        return out

    def complete_outbox(self, outbox_id, token, state="done"):
        with self._tx():
            cur = self._conn.execute(
                "SELECT lease_token FROM outbox WHERE outbox_id=?",
                (outbox_id,)).fetchone()
            if cur is None or cur["lease_token"] != token:
                return False
            state = state if state in ("done", "failed", "pending") else "done"
            if state == "pending":
                # 归还：必须清 lease，否则未过期的 lease_expires_at
                # 会挡住下一次 claim（state='pending' 但 lease 未失效）
                self._conn.execute(
                    "UPDATE outbox SET state='pending', lease_token=NULL, "
                    "lease_expires_at=NULL WHERE outbox_id=?", (outbox_id,))
            else:
                self._conn.execute(
                    "UPDATE outbox SET state=? WHERE outbox_id=?",
                    (state, outbox_id))
            return True

    # -------------------------------------------------------------- 路由日志

    def log_route(self, session_id, request_id, mode, snapshot=None,
                  plan_json=None, parse_ok=None, error=None, latency_ms=None):
        with self._tx():
            self._conn.execute(
                "INSERT INTO route_log(session_id, request_id, mode, "
                "snapshot_json, plan_json, parse_ok, error, latency_ms, "
                "created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (session_id, request_id, mode,
                 json.dumps(snapshot or {}, ensure_ascii=False, default=str),
                 plan_json, None if parse_ok is None else int(parse_ok),
                 error, latency_ms, self._now()))


    # ------------------------------------------------- 调度器辅助（P1d 加法）

    def count_active_runs(self, session_id):
        """该会话仍在执行的 run 数（running/stop_requested 且有有效 lease）。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT lease_expires_at FROM runs WHERE session_id=? AND "
                "state IN ('running','stop_requested')",
                (session_id,)).fetchall()
            now = time.time()
            return sum(1 for r in rows
                       if r["lease_expires_at"] and r["lease_expires_at"] > now)

    def set_task_blocked(self, session_id, task_id, reason):
        """依赖失败/取消 → 下游 blocked_dependency（只允许从 queued 迁移）。"""
        with self._tx():
            t = self.task_row(session_id, task_id)
            if t is None or t["execution_status"] != C.EXEC_QUEUED:
                return False
            self._conn.execute(
                "UPDATE tasks SET execution_status=?, updated_at=? WHERE "
                "session_id=? AND task_id=?",
                (C.EXEC_BLOCKED_DEPENDENCY, self._now(), session_id, task_id))
            self._append_event(session_id, "dependency_blocked",
                               task_id=task_id, payload={"reason": reason})
            return True

    def latest_event(self, session_id, etype, task_id=None, revision=None):
        with self._lock:
            q = ("SELECT * FROM events WHERE session_id=? AND type=?")
            args = [session_id, etype]
            if task_id:
                q += " AND task_id=?"
                args.append(task_id)
            if revision is not None:
                q += " AND revision=?"
                args.append(revision)
            q += " ORDER BY seq DESC LIMIT 1"
            row = self._conn.execute(q, args).fetchone()
            return dict(row) if row else None

    # ------------------------------------------------- 锁外公共入口（P1d 加法）

    def append_event(self, session_id, etype, task_id=None, revision=None,
                     run_id=None, payload=None):
        """scheduler/delivery 在事务外加事件的公共入口。"""
        with self._tx():
            return self._append_event(session_id, etype, task_id=task_id,
                                      revision=revision, run_id=run_id,
                                      payload=payload)

    def enqueue_outbox(self, session_id, kind, payload):
        """scheduler/delivery 在事务外投递 outbox 条目的公共入口。"""
        with self._tx():
            self._enqueue(session_id, kind, payload)
