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
# deliver 项的领取租约：必须覆盖最长结果播报（长结果 TTS+播放可达
# 1-2 分钟）。默认 OUTBOX_LEASE_TTL_S=30s 会在播报中途过期——其他
# 消费者（scheduler 的跨 kind 领取）把过期项归还 pending 会让本泵
# 的 complete_outbox token 静默失效，项被自己重领后对已 played_confirmed
# 的交付行再写 RENDERING → StoreError 崩溃循环（2026-11-09 e2e 事故，
# attempts=58）。泵线程若真死，300s 后仍可自愈（at-least-once）。
_DELIVER_LEASE_TTL_S = 300.0

# 已终态处置的交付状态：重复 deliver 项见到即收尾，不重渲染不重播。
# sending 也算——它只在开口之后才写入（rc True/False 两分支都是先播
# 完才落 sending），遗留 sending 必属"上一 attempt 语音已出"的崩溃残留。
# voiced_by_lane 同理（2026-11-09 窄版 voiced-by-lane）：创建轮闲聊路
# 已亲口说过、泵被跳过的结果，重复 deliver 项直接收尾。
_DLV_ALREADY_HANDLED = frozenset({
    C.DLV_SENDING, C.DLV_SENT_UNCONFIRMED, C.DLV_PLAYED_CONFIRMED,
    C.DLV_SUPPRESSED, C.DLV_INTERRUPTED_PARTIAL,
    C.DLV_INTERRUPTED_BEFORE_AUDIO, C.DLV_RETRY_EXHAUSTED,
    C.DLV_VOICED_BY_LANE})


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

    # ------------------------------------------------------------------ 掐音口

    def stop_speaking_for_task(self, session_id, task_id):
        """掐断该任务正在播的交付语音（2026-11-09 e2e 取消竞态）。

        取消落在"已完成、结果正在播"的窗口里时，_op_cancel 只压状态
        （cancel_intent + suppressed），在途语音会完整播完——用户明确
        说了"算了"，却把整段已取消的结果听完。泵线程被 speak_fn 占住，
        队列里的 stop_speech 自己永远轮不到，必须由提交路径跨线程把
        stop_event 置位（speak_fn 内部轮询它，rc 返回 False=被打断，
        泵的开口后复核会把交付落成 suppressed 终态）。"""
        with self._speaking_lock:
            cur = self._speaking.get(session_id)
            if cur and cur["envelope"].task_id == task_id:
                cur["stop_event"].set()

    def stop_speaking(self, session_id):
        """掐断该会话当前在播的交付语音（all_active 取消的兜底口）。"""
        with self._speaking_lock:
            cur = self._speaking.get(session_id)
            if cur:
                cur["stop_event"].set()

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
        # deliver 的租约必须覆盖最长播报（见 _DELIVER_LEASE_TTL_S 注释）；
        # 其余 kind 用 store 默认 30s
        ttl = _DELIVER_LEASE_TTL_S if kind == "deliver" \
            else S.OUTBOX_LEASE_TTL_S
        claimed = self.store.claim_outbox(limit=4, lease_ttl=ttl)
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
        row = self.store.delivery_row(delivery_id)
        if row and row["state"] in _DLV_ALREADY_HANDLED:
            # 重复 deliver 项（2026-11-09 e2e 事故）：这条结果已有终态
            # 处置——已播出（played/sent）、被压制、被打断或重试耗尽。
            # 重渲染会撞终态迁移（StoreError 崩溃循环），把已听过的
            # 结果再播一遍本身就是重复播报。收尾走人。
            # （合法重播不受影响：_redeliver_unheard 在重排前必先把
            # 交付行重置回 pending。）
            self.store.complete_outbox(item["outbox_id"], item["token"])
            return
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
