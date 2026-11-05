# -*- coding: utf-8 -*-
"""agent/frontend_adapter.py — app.py 与 AgentRuntimeV2 之间的窄接入层
（手册 §6）。

所有语音前端能力通过回调注入，本模块不 import app/clients —— 语音前端
（轮次判别、grace、ASR、TTS、播放）是第一公民，adapter 只读它们的结论：

  grace_s_fn()        -> float   当前 grace 阈值（与前端共享同一个
                                turn_grace_ms 设置，绝不复制第二套计时）
  last_voice_ts_fn()  -> float   前端记录的最后语音时间戳
  speak_fn(text, stop_event) -> True/False/None
                                与 _run_speech_pipeline 完全一致的三态：
                                True=完整播出 False=播出后打断 None=静默掐死
  can_deliver_fn()    -> bool    现有话权结论（_agent_can_deliver）
  generate_fn(stop_event) -> segment 生成器
                                表达轮（direct/mixed 的 direct_question）
  history_fn()        -> [{role, content}]   最近对话（喂路由快照）

三种模式（§19 发布开关）：
  legacy  —— adapter 完全不参与（app 侧根本不创建）
  shadow  —— observe()：只跑路由并 log_route，不影响任何语音输出
  enforce —— handle_utterance()：V2 控制器接管控制轮——提交计划（grace
             门后）、模板回执上口、表达轮走 generate_fn

grace 提交门（§6.1）：计划在 grace 窗口内只挂起；窗口内续说 → 旧计划
作废（discard_pending），新输入重新路由。adapter 侧轮询 last_voice_ts，
阈值取前端同一个 grace_s，不重做第二套计时器。
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# 提交门轮询间隔（秒）。只读时间戳，无锁竞争。
GRACE_POLL_S = 0.05
# 路由失败时的固定澄清话术（§6.2：超时/失败必须有终态并释放队列）
ROUTER_FAIL_TEXT = "这个我一时没理清要做什么，你能再说一遍吗？"


class AgentFrontendAdapter:
    def __init__(self, session_id: str, runtime, router,
                 snapshot_builder: Callable,
                 grace_s_fn: Callable[[], float],
                 last_voice_ts_fn: Callable[[], float],
                 speak_fn: Callable,
                 can_deliver_fn: Callable[[], bool],
                 generate_fn: Optional[Callable] = None,
                 history_fn: Optional[Callable] = None,
                 mode: str = "shadow",
                 grace_poll_s: float = GRACE_POLL_S,
                 shadow_idle_wait_s: float = 30.0,
                 on_event: Optional[Callable] = None):
        """runtime: AgentRuntimeV2；router: PlanRouter；snapshot_builder:
        (user_text, request_id, history) -> snapshot dict（agent.context）。
        on_event(event, data)：前端任务面板等 UI 通知（可选）。"""
        self.session_id = session_id
        self.runtime = runtime
        self.router = router
        self.snapshot_builder = snapshot_builder
        self.grace_s_fn = grace_s_fn
        self.last_voice_ts_fn = last_voice_ts_fn
        self.speak_fn = speak_fn
        self.can_deliver_fn = can_deliver_fn
        self.generate_fn = generate_fn
        self.history_fn = history_fn or (lambda: [])
        self.mode = mode
        self.grace_poll_s = grace_poll_s
        self.shadow_idle_wait_s = shadow_idle_wait_s
        self.on_event = on_event
        self._utterance_lock = threading.Lock()   # 串行化 utterance 处理
        self._stop = threading.Event()
        # 当前挂起的控制轮线程（一次只有一个；新话轮到来时旧线程自行退出）
        self._active_turn: dict = {}

    # ------------------------------------------------------------------ 生命周期

    def close(self):
        """连接关闭（§14.1）：停线程、关会话。先落 closed 标记再关
        store（stop_background 会关连接，顺序反了标记就永远写不上）。"""
        self._stop.set()
        try:
            self.runtime.store.close_session(self.session_id,
                                             reason="connection_closed")
        except Exception:  # noqa: BLE001
            pass
        self.runtime.stop_background()

    # ------------------------------------------------------------------ shadow

    def observe(self, text: str, turn_trigger: str = "",
                source_commit_id: Optional[str] = None):
        """shadow 模式：后台跑一次路由，只写 log_route，不碰任何输出。
        语音路径与 legacy 完全一致。"""

        def _run():
            try:
                # GEN_LOCK 与语音生成共享：等快脑真正空闲（无生成、无播放、
                # 用户静默）再路由，绝不抢占正在服务的语音轮；有界等待，
                # 等不到就放弃本次观察（观察可丢，语音不可扰）
                idle_deadline = time.time() + self.shadow_idle_wait_s
                while time.time() < idle_deadline and not self._stop.is_set():
                    try:
                        if self.can_deliver_fn():
                            break
                    except Exception:  # noqa: BLE001
                        # 读不到话权状态就无法保证不扰语音——放弃观察
                        return
                    time.sleep(0.2)
                else:
                    logger.info("[%s][shadow] fast brain never idle; "
                                "observation skipped", self.session_id[:8])
                    return
                snap = self.snapshot_builder(text, None, self.history_fn())
                rr = self.router.route(snap, text,
                                       history=self.history_fn())
                # log_route 由 runtime.submit_request 之外的观察通道写；
                # 这里直接用 store 的路由日志（mode=shadow）
                self.runtime.store.log_route(
                    self.session_id, None, "shadow",
                    snapshot={"user_text": text},
                    plan_json=rr.plan_json if rr.ok else None,
                    error=rr.error)
                logger.info(
                    "[%s][shadow] route %s: %s", self.session_id[:8],
                    "ok" if rr.ok else "fail",
                    rr.to_log() if rr.ok else rr.error)
            except Exception:  # noqa: BLE001 — 观察绝不能影响主路径
                logger.exception("[%s][shadow] route observation failed",
                                 self.session_id[:8])

        threading.Thread(target=_run, daemon=True,
                         name=f"agent-v2-shadow-{self.session_id[:8]}").start()

    # ------------------------------------------------------------------ enforce

    def handle_utterance(self, text: str, turn_trigger: str = "",
                         source_commit_id: Optional[str] = None):
        """enforce 模式：V2 控制器接管本轮控制决策。app 在 ASR 文本确定、
        自回声过滤与话轮门之后调用；不阻塞收音线程（路由在后台跑）。

        流程（§6/§7）：
          1. 上一话轮的计划还没过 grace 门 → 作废（续说场景，§6.2）
          2. submit_request（路由，3-6s）→ 计划挂起
          3. 等 grace 门开（共享阈值，轮询 last_voice_ts）→ commit
          4. 回执模板话术上口（speak_fn 三态）；direct/mixed 追加表达轮
        """
        with self._utterance_lock:
            prev = self._active_turn.get("thread")
            if prev is not None and prev.is_alive():
                # 上一计划未提交：续说作废（§6.2 旧计划作废，重新路由）
                self.runtime.discard_pending(self.session_id,
                                             reason="continuation")
                self._active_turn["abandoned"] = True

        th = threading.Thread(
            target=self._turn_worker, daemon=True,
            name=f"agent-v2-turn-{self.session_id[:8]}",
            args=(text, turn_trigger, source_commit_id))
        self._active_turn = {"thread": th, "abandoned": False}
        th.start()

    def _turn_worker(self, text, turn_trigger, source_commit_id):
        try:
            self._turn_worker_inner(text, turn_trigger, source_commit_id)
        except Exception:  # noqa: BLE001 — 控制轮失败必须落到固定澄清话术
            logger.exception("[%s][v2] turn worker crashed",
                             self.session_id[:8])
            self._speak_safe(ROUTER_FAIL_TEXT)

    def _turn_worker_inner(self, text, turn_trigger, source_commit_id):
        res = self.runtime.submit_request(
            self.session_id, text, source_commit_id=source_commit_id)
        if not res.get("ok"):
            # 路由失败/超时：终态 + 固定澄清（§6.2），队列已释放
            logger.info("[%s][v2] route failed: %s", self.session_id[:8],
                        res.get("error"))
            self._speak_safe(ROUTER_FAIL_TEXT)
            return
        # grace 提交门（§6.1）：窗口内只挂起；复用前端同一阈值
        grace_s = float(self.grace_s_fn() or 0.0)
        if grace_s > 0:
            deadline = time.time() + grace_s + 5.0   # 门最多等 grace+5s
            while time.time() < deadline:
                if self._stop.is_set():
                    return
                if self._active_turn.get("abandoned"):
                    return   # 续说：旧计划已作废，新话轮线程接管
                if not self.runtime.pending_exists(self.session_id):
                    return  # 已被作废/提交
                if time.time() - self.last_voice_ts_fn() >= grace_s:
                    break
                time.sleep(self.grace_poll_s)
        if (self._stop.is_set() or self._active_turn.get("abandoned")
                or not self.runtime.pending_exists(self.session_id)):
            return
        out = self.runtime.commit_pending(self.session_id)
        if out is None:
            return  # 门又关上（续说抢先）——计划仍挂起，交由下一轮处理
        # 回执话术：程序模板，不让模型自由解释（§13.1）
        for receipt_text in out.get("texts") or []:
            if not self._speak_safe(receipt_text):
                return   # 打断即终态，本轮后续回执不再上口
        # 表达轮：direct/mixed 的 direct_question 走正常生成（§7.3：不能
        # 因为本轮有一个控制操作就吞掉另一个问题）
        plan = out.get("plan")
        if (self.generate_fn is not None and plan is not None
                and plan.reply_mode in ("direct", "mixed")
                and (plan.direct_question or plan.reply_mode == "direct")):
            stop_event = threading.Event()
            try:
                gen = self.generate_fn(stop_event)
                rc = self.speak_fn(gen, stop_event)
                logger.info("[%s][v2] expression round rc=%s",
                            self.session_id[:8], rc)
            except Exception:  # noqa: BLE001
                logger.exception("[%s][v2] expression round failed",
                                 self.session_id[:8])

    def _speak_safe(self, text) -> bool:
        """单条固定话术上口。返回是否完整播出（False/None 都不算）。"""
        stop_event = threading.Event()
        try:
            rc = self.speak_fn(text, stop_event)
        except Exception:  # noqa: BLE001
            logger.exception("[%s][v2] speak failed", self.session_id[:8])
            return False
        return rc is True
