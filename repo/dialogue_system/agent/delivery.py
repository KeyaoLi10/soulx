# -*- coding: utf-8 -*-
"""agent/delivery.py — 议程、发言 owner 与交付状态机（手册 §13）。

三态语义（与 legacy `_run_speech_pipeline` 的返回约定逐条对齐——这是
2026-11-05 投递三连修的等价复刻，语音前端本身不动）：
  speak_fn(envelope, text, stop_event) -> True  完整播出
                                        False 播出后被打断（有播放迹象）
                                        None  静默掐死（一个字都没播）
  - None  → 重试（≤ silent_delivery_retries），耗尽 → retry_exhausted
    （不写 delivered；结果仍可查询）
  - False → interrupted_partial，终态，不自动重播
  - True  → sent_unconfirmed → played_confirmed（当前为服务端估计，
    手册 §13.4 estimated 兼容默认）
有效性：每次开口前按 SpeechEnvelope + 任务快照核验（改口/取消的旧结果
不再播）；发言 owner 按 session 串行，一次只有一个声音。

can_deliver_fn(session_id) -> bool：读现有话权（映射 _agent_can_deliver），
不可送达时等待有界时间后归还 outbox 重试，不硬抢话轮。
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Optional

from agent import contracts as C
from agent import store as S

SILENT_DELIVERY_RETRIES = 3
DELIVERY_WAIT_S = 2.0        # 话权不可用时的有界等待
POLL_INTERVAL_S = 0.25


class DeliveryManager:
    def __init__(self, store: S.AgentStore, speak_fn: Callable,
                 can_deliver_fn: Optional[Callable] = None,
                 silent_retries: int = SILENT_DELIVERY_RETRIES,
                 poll_interval_s: float = POLL_INTERVAL_S,
                 delivery_wait_s: float = DELIVERY_WAIT_S):
        self.store = store
        self.speak_fn = speak_fn
        self.can_deliver_fn = can_deliver_fn or (lambda sid: True)
        self.silent_retries = silent_retries
        self.poll_interval = poll_interval_s
        self.delivery_wait_s = delivery_wait_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._speaking: dict = {}          # session_id -> {envelope, stop_event}
        self._speaking_lock = threading.Lock()
        # 测试钩子
        self.on_spoken = None              # callable(session_id, envelope, rc)

    # ------------------------------------------------------------------ 生命周期

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="agent-delivery")
        self._thread.start()

    def stop(self, join_timeout=5.0):
        self._stop.set()
        if self._thread:
            self._thread.join(join_timeout)

    # ------------------------------------------------------------------ 主循环

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.pump()
            except Exception:  # noqa: BLE001
                import traceback
                traceback.print_exc()
            self._stop.wait(self.poll_interval)

    def pump(self):
        """处理 stop_speech（先于新交付）与一条待播结果。"""
        self._pump_stop_speech()
        self._pump_one_delivery()

    def _claim_kind(self, kind):
        claimed = self.store.claim_outbox(limit=4)
        keep = [i for i in claimed if i["kind"] == kind]
        for i in claimed:
            if i["kind"] != kind:
                self.store.complete_outbox(i["outbox_id"], i["token"],
                                           state="pending")
        return keep

    def _pump_stop_speech(self):
        for item in self._claim_kind("stop_speech"):
            sid = self._session_of(item["outbox_id"])
            with self._speaking_lock:
                cur = self._speaking.get(sid)
                if cur:
                    # 只停当前发言 owner 的声音（§9.2：不用全局 interrupt
                    # 杀别的任务的发言——按 session 串行下即本会话）
                    cur["stop_event"].set()
            self.store.complete_outbox(item["outbox_id"], item["token"])

    def _session_of(self, outbox_id):
        with self.store._lock:
            row = self.store._conn.execute(
                "SELECT session_id FROM outbox WHERE outbox_id=?",
                (outbox_id,)).fetchone()
            return row["session_id"] if row else None

    def _pump_one_delivery(self):
        items = self._claim_kind("deliver")
        if not items:
            return
        item = items[0]
        for other in items[1:]:
            self.store.complete_outbox(other["outbox_id"], other["token"],
                                       state="pending")
        session_id = self._session_of(item["outbox_id"])
        payload = item["payload"]
        task_id = payload.get("task_id")
        revision = payload.get("revision")
        # 同会话一次只有一个声音：正在播 → 归还
        with self._speaking_lock:
            if session_id in self._speaking:
                self.store.complete_outbox(item["outbox_id"], item["token"],
                                           state="pending")
                return
        # 有效性核验（§13.2）：改口/取消后的旧结果不播
        env = C.SpeechEnvelope(
            speech_id=f"sp-{payload.get('run_id', '')}",
            session_id=session_id,
            connection_epoch=self._connection_epoch(session_id),
            kind=payload.get("kind", "result"),
            task_id=task_id, revision=revision or 0,
            run_id=payload.get("run_id"))
        if task_id and not env.valid_for(self._task_rev_map(session_id)):
            self.store.complete_outbox(item["outbox_id"], item["token"])
            return
        # 话权：等有界时间，拿不到归还重试
        deadline = time.monotonic() + self.delivery_wait_s
        while time.monotonic() < deadline and not self._stop.is_set():
            if self.can_deliver_fn(session_id):
                break
            time.sleep(0.1)
        else:
            self.store.complete_outbox(item["outbox_id"], item["token"],
                                       state="pending")
            return
        if not self.can_deliver_fn(session_id):
            self.store.complete_outbox(item["outbox_id"], item["token"],
                                       state="pending")
            return
        text = self._render_text(payload)
        if not text:
            self.store.complete_outbox(item["outbox_id"], item["token"])
            return
        delivery_id = self.store.record_delivery(
            session_id, payload.get("kind", "result"), task_id=task_id,
            revision=revision, run_id=payload.get("run_id"))
        self.store.set_delivery_state(delivery_id, C.DLV_RENDERING)
        stop_event = threading.Event()
        with self._speaking_lock:
            self._speaking[session_id] = {"envelope": env,
                                          "stop_event": stop_event}
        try:
            rc = self.speak_fn(env, text, stop_event)
        finally:
            with self._speaking_lock:
                self._speaking.pop(session_id, None)
        # 开口后再核一次：播放期间任务可能已被改口/取消
        if task_id and not env.valid_for(self._task_rev_map(session_id)):
            self.store.set_delivery_state(delivery_id, C.DLV_SUPPRESSED)
            self.store.complete_outbox(item["outbox_id"], item["token"])
            return
        row = self.store.delivery_row(delivery_id)
        retries = row["retries"] if row else 0
        if rc is None:
            # 静默掐死：重试（预算内），不当作成功
            if retries + 1 >= self.silent_retries:
                self.store.bump_delivery_retry(delivery_id)
                self.store.set_delivery_state(delivery_id,
                                              C.DLV_RETRY_EXHAUSTED)
                self.store.complete_outbox(item["outbox_id"], item["token"])
            else:
                self.store.bump_delivery_retry(delivery_id)
                self.store.set_delivery_state(delivery_id, C.DLV_PENDING)
                self.store.complete_outbox(item["outbox_id"], item["token"],
                                           state="pending")
        elif rc is False:
            # 播出后被打断：终态，不自动重播（打断≠取消计算≠任务失败）
            # 有播放迹象 → 已进入 sending/sent_unconfirmed 再落 interrupted
            self.store.set_delivery_state(delivery_id, C.DLV_SENDING)
            self.store.set_delivery_state(delivery_id,
                                          C.DLV_SENT_UNCONFIRMED)
            self.store.set_delivery_state(delivery_id,
                                          C.DLV_INTERRUPTED_PARTIAL)
            self.store.complete_outbox(item["outbox_id"], item["token"])
        else:
            self.store.set_delivery_state(delivery_id, C.DLV_SENDING)
            self.store.set_delivery_state(delivery_id,
                                          C.DLV_SENT_UNCONFIRMED)
            # 服务端估计（§13.4）：ACK 通道具备后才可 confirmed
            self.store.set_delivery_state(delivery_id,
                                          C.DLV_PLAYED_CONFIRMED)
            self.store.complete_outbox(item["outbox_id"], item["token"])
        if self.on_spoken:
            try:
                self.on_spoken(session_id, env, rc)
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------ 辅助

    def _connection_epoch(self, session_id):
        sess = self.store.session_row(session_id)
        return sess["connection_epoch"] if sess else 0

    def _task_rev_map(self, session_id):
        return {t["task_id"]: t for t in self.store.task_view(session_id)}

    def _render_text(self, payload):
        """结果话术：completed 用 payload.summary；failed 用固定受控话术
        （空结果不得包装成 completed——手册 §10）。"""
        outcome = payload.get("outcome")
        label = payload.get("label") or ""
        if outcome == C.EXEC_WAITING_INPUT:
            q = (payload.get("payload") or {}).get("question") or ""
            return q or "我需要再确认一个信息才能继续。"
        if outcome == C.EXEC_FAILED:
            lim = (payload.get("payload") or {}).get("limitations") or []
            extra = f"{'；'.join(lim[:2])}。" if lim else ""
            return f"这项任务没有完成。{extra}需要的话我可以换个方式再试。"
        summary = (payload.get("summary")
                   or (payload.get("payload") or {}).get("summary") or "")
        return str(summary).strip()
