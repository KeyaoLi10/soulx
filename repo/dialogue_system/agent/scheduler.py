# -*- coding: utf-8 -*-
"""agent/scheduler.py — 有界队列、run lease、依赖与取消（手册 §8/§10/§14.2）。

职责：消费 outbox 的 schedule / cancel_run 条目，为到期的任务创建 run、
在容量允许时启动 worker 线程执行慢脑 runner。模型与工具调用全部发生在
锁外（worker 线程），状态变更全部走 store 的短事务。

runner 接口（P4 接真实慢脑；P1d 用假 runner 离线测试）：
    runner(snapshot: dict, cancel_event: threading.Event,
           progress_cb: callable) -> dict
返回 {"type": "completed"|"failed"|"needs_input", "payload": {...}}，
payload 遵循手册 §10 事件信封（completed 带 summary/facts/sources/outcome；
needs_input 带 question_id/question/missing_fields）。

取消：cancel_run 条目 → 找到该 run 的 worker cancel_event 并置位；
worker 在模型调用前/工具前后/发布结果前检查（§9.3 的 runner 侧义务）。
"""
from __future__ import annotations

import json
import logging
import threading
import time
from typing import Callable, Optional

from agent import contracts as C
from agent import store as S

logger = logging.getLogger(__name__)

POLL_INTERVAL_S = 0.5
MAX_RUNNING_PER_SESSION = 2
GLOBAL_WORKER_CAP = 4


class TaskScheduler:
    def __init__(self, store: S.AgentStore, runner: Callable,
                 max_running_per_session: int = MAX_RUNNING_PER_SESSION,
                 global_worker_cap: int = GLOBAL_WORKER_CAP,
                 poll_interval_s: float = POLL_INTERVAL_S):
        self.store = store
        self.runner = runner
        self.max_running = max_running_per_session
        self.global_cap = global_worker_cap
        self.poll_interval = poll_interval_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._workers: dict = {}          # run_id -> {"thread","cancel_event"}
        self._workers_lock = threading.Lock()
        # 测试可注入的确定性时钟/钩子
        self.on_run_started = None        # callable(run_id, snapshot)
        self.on_run_finished = None       # callable(run_id, result)

    # ------------------------------------------------------------------ 生命周期

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="agent-scheduler")
        self._thread.start()

    def stop(self, join_timeout=5.0):
        self._stop.set()
        if self._thread:
            self._thread.join(join_timeout)
        with self._workers_lock:
            for w in self._workers.values():
                w["cancel_event"].set()
        deadline = time.monotonic() + join_timeout
        while time.monotonic() < deadline:
            with self._workers_lock:
                alive = any(w["thread"].is_alive()
                            for w in self._workers.values())
            if not alive:
                break
            time.sleep(0.05)

    # ------------------------------------------------------------------ 主循环

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.pump()
            except Exception:  # noqa: BLE001 — 调度循环绝不能死
                import traceback
                traceback.print_exc()
            self._stop.wait(self.poll_interval)

    def pump(self):
        """单轮调度：处理取消请求，再尝试启动到期的任务。"""
        self._pump_cancels()
        self._pump_schedules()

    def _pump_cancels(self):
        # 只取 cancel_run（claim_outbox 不区分 kind，内存过滤后归还其余）
        for item in self._claim_kind("cancel_run"):
            run_id = item["payload"].get("run_id")
            with self._workers_lock:
                w = self._workers.get(run_id)
                if w:
                    w["cancel_event"].set()
            self.store.complete_outbox(item["outbox_id"], item["token"])

    def _claim_kind(self, kind):
        """领取指定 kind 的 outbox 条目（claim 不区分 kind，这里多领后
        把不匹配的按原状态归还）。"""
        claimed = self.store.claim_outbox(limit=8)
        keep = [i for i in claimed if i["kind"] == kind]
        for i in claimed:
            if i["kind"] != kind:
                # 归还：写回 pending（attempts 已 +1，可接受——at-least-once）
                self.store.complete_outbox(i["outbox_id"], i["token"],
                                           state="pending")
        return keep

    def _pump_schedules(self):
        for item in self._claim_kind("schedule"):
            if self._stop.is_set():
                self.store.complete_outbox(item["outbox_id"], item["token"],
                                           state="pending")
                return
            self._try_start(item)

    def _try_start(self, item):
        payload = item["payload"]
        session_id = None
        # outbox 行带 session_id，payload 里也冗余存了 task 上下文
        row = self.store._conn.execute(
            "SELECT session_id FROM outbox WHERE outbox_id=?",
            (item["outbox_id"],)).fetchone()
        session_id = row["session_id"] if row else None
        task_id = payload.get("task_id")
        revision = payload.get("revision")
        t = self.store.task_row(session_id, task_id) if session_id else None
        # 条目过期：任务已改版/取消/已跑 → 丢弃
        if (t is None or t["current_revision"] != revision
                or t["execution_status"] not in (C.EXEC_QUEUED,)):
            self.store.complete_outbox(item["outbox_id"], item["token"])
            return
        # 依赖未满足 → 归还待办
        dep_status = self._dependency_status(session_id, task_id, revision)
        if dep_status == "pending":
            self.store.complete_outbox(item["outbox_id"], item["token"],
                                       state="pending")
            return
        if dep_status == "blocked":
            self.store.set_task_blocked(session_id, task_id,
                                        reason="dependency_failed")
            self.store.complete_outbox(item["outbox_id"], item["token"])
            return
        # 容量：每会话运行数 + 全局 worker 数
        if (self.store.count_active_runs(session_id) >= self.max_running
                or self._active_worker_count() >= self.global_cap):
            self.store.complete_outbox(item["outbox_id"], item["token"],
                                       state="pending")
            return
        try:
            run_id = self.store.create_run(session_id, task_id, revision,
                                           snapshot=self._snapshot(
                                               session_id, task_id, revision))
        except S.StoreError:
            # create_run 的 revision 复验失败——条目过期
            self.store.complete_outbox(item["outbox_id"], item["token"])
            return
        token = self.store.lease_run(run_id)
        if not token:
            self.store.complete_outbox(item["outbox_id"], item["token"],
                                       state="pending")
            return
        cancel_event = threading.Event()
        th = threading.Thread(
            target=self._worker, daemon=True,
            name=f"agent-run-{run_id}",
            args=(session_id, task_id, revision, run_id, token, cancel_event))
        with self._workers_lock:
            self._workers[run_id] = {"thread": th, "cancel_event": cancel_event}
        th.start()
        if self.on_run_started:
            try:
                self.on_run_started(run_id, self._snapshot(
                    session_id, task_id, revision))
            except Exception:  # noqa: BLE001
                pass
        self.store.complete_outbox(item["outbox_id"], item["token"])

    def _active_worker_count(self):
        with self._workers_lock:
            return sum(1 for w in self._workers.values()
                       if w["thread"].is_alive())

    # ------------------------------------------------------------------ 依赖

    def _dependency_status(self, session_id, task_id, revision):
        """all_done | pending | blocked。上游失败/取消 → blocked。"""
        rev = self.store.revision_row(session_id, task_id, revision)
        if rev is None:
            return "blocked"
        deps = json.loads(rev["depends_json"])
        if not deps:
            return "all_done"
        for d in deps:
            t = self.store.task_row(session_id, d["task_id"])
            if t is None:
                return "blocked"
            if t["current_revision"] != d["revision"]:
                return "blocked"      # 上游被改版 → 下游绑定版本失效
            st = t["execution_status"]
            if st in (C.EXEC_COMPLETED,):
                continue
            if st in (C.EXEC_FAILED, C.EXEC_CANCELED,
                      C.EXEC_BLOCKED_DEPENDENCY):
                return "blocked"
            return "pending"
        return "all_done"

    def _dep_results(self, session_id, task_id, revision):
        rev = self.store.revision_row(session_id, task_id, revision)
        deps = json.loads(rev["depends_json"]) if rev else []
        out = []
        for d in deps:
            ev = self.store.latest_event(session_id, "completed",
                                         task_id=d["task_id"],
                                         revision=d["revision"])
            out.append({
                "task_id": d["task_id"], "revision": d["revision"],
                "result": json.loads(ev["payload_json"]) if ev else None,
            })
        return out

    # ------------------------------------------------------------------ 快照

    def _snapshot(self, session_id, task_id, revision):
        rev = self.store.revision_row(session_id, task_id, revision)
        sess = self.store.session_row(session_id)
        snap = C.TaskExecutionSnapshot(
            session_id=session_id, task_id=task_id, revision=revision,
            run_id="", goal=rev["goal"] if rev else "",
            constraints=json.loads(rev["constraints_json"]) if rev else {},
            context="", user_text=rev["user_text"] if rev else "",
            timezone=sess["timezone"] if sess else "Asia/Shanghai",
            depends=self._dep_results(session_id, task_id, revision))
        return snap.to_dict()

    # ------------------------------------------------------------------ worker

    def _worker(self, session_id, task_id, revision, run_id, token,
                cancel_event):
        try:
            self._worker_inner(session_id, task_id, revision, run_id, token,
                               cancel_event)
        finally:
            with self._workers_lock:
                self._workers.pop(run_id, None)

    def _worker_inner(self, session_id, task_id, revision, run_id, token,
                      cancel_event):
        snapshot = self._snapshot(session_id, task_id, revision)
        snapshot["run_id"] = run_id
        progress_seq = [0]

        def progress_cb(etype, payload):
            """慢脑进度事件：只在 run 仍是当前时对外可见（§9.3）。"""
            if cancel_event.is_set():
                return
            if not self.store.run_is_current(session_id, task_id, run_id,
                                             revision):
                return
            progress_seq[0] += 1
            self.store.append_event(
                session_id, etype, task_id=task_id, revision=revision,
                run_id=run_id, payload=payload or {})

        def heartbeat():
            return self.store.heartbeat_run(run_id, token)

        result = self.runner(snapshot, cancel_event, progress_cb)
        # 心跳已失（lease 被夺）：结果丢弃，只留审计
        if not heartbeat():
            self.store.append_event(
                session_id, "run_lease_lost", task_id=task_id,
                revision=revision, run_id=run_id,
                payload={"state": result.get("type", "?")})
            return
        rtype = result.get("type", "failed")
        payload = result.get("payload", {}) or {}
        if cancel_event.is_set() and rtype == "canceled":
            state = C.EXEC_CANCELED
        elif rtype == "needs_input":
            state = C.EXEC_WAITING_INPUT
        elif rtype == "completed":
            state = C.EXEC_COMPLETED
        else:
            state = C.EXEC_FAILED
        try:
            out = self.store.finish_run(
                session_id, task_id, revision, run_id, token, state,
                result_summary=payload.get("summary", ""),
                payload=payload)
        except S.StoreError:
            return
        if self.on_run_finished:
            try:
                self.on_run_finished(run_id, result)
            except Exception:  # noqa: BLE001
                pass
        if out == "fenced":
            return  # 审计已写，不再安排交付
        # 窄版 voiced-by-lane（2026-11-09，三条件详见 store 块注释）：
        # 创建轮闲聊路亲口说过 + 本 run 零工具调用 + 首个 run 的完成
        # 结果是慢脑自产内容（笑话/寒暄类），闲聊路面板可见即等同已
        # 播报——泵不再入队（事故：T4 笑话挂"待播报"1 分 54 秒逼出
        # "是不是重复啦"幻觉）。抑制在入队时刻而非泵领取时刻：显式
        # 重播（get_task 重开）路径不受影响。失败/待输入照常走泵。
        if state == C.EXEC_COMPLETED and self.store.result_voiced_by_lane(
                session_id, task_id, run_id):
            dlv = self.store.record_delivery(
                session_id, "result", task_id=task_id, revision=revision,
                run_id=run_id)
            self.store.set_delivery_state(dlv, C.DLV_VOICED_BY_LANE)
            logger.info(
                "[%s] result voiced by lane: %s (creation turn voiced, "
                "run used no tools) — delivery pump skipped",
                session_id, task_id)
            return
        # 结果交付入 outbox（delivery.py 消费）
        if state in (C.EXEC_COMPLETED, C.EXEC_FAILED,
                     C.EXEC_WAITING_INPUT):
            self.store.enqueue_outbox(session_id, "deliver", {
                "kind": "result", "task_id": task_id, "revision": revision,
                "run_id": run_id, "outcome": state,
                "summary": payload.get("summary", ""),
                "payload": payload})
