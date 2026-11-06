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
  chitchat_fn(stop_event) -> segment 生成器
                                闲聊路（voice lane）：任务面板注入的
                                generate_with_history，与本轮路由并行，
                                负责本轮全部常规口播（2026-11-06 双路
                                并行设计，方案 B）
  generate_fn(stop_event) -> segment 生成器
                                表达轮兜底（方案 C）：仅闲聊路失败/缺席
                                时 direct/mixed 才走旧串行路径
  history_fn()        -> [{role, content}]   最近对话（喂路由快照）
  release_gen_fn(gen_sn) -> None  释放收音循环 commit 时置位的
                                gen_in_progress（enforce 无声轮兜底，
                                gen_serial 已变时为 no-op）

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
import re
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# 纯应答/附和：整句只由应答词构成（"嗯""好的好的""对"……）。这类话永远
# 不承载控制意图。吸收条件 = 上一话轮还活着（路由在飞/计划挂起）或面板
# 上有任务（用户大概率在回应任务交付）——两种情况下都不作废挂起计划、
# 不再起一路路由、不另起闲聊路。2026-11-06 实测事故：天气请求路由 6.3s
# 期间用户一句"嗯，好的好的"把成果作废重来，连环重路由四次，任务 25s
# 后才创建；2026-11-07 实测事故：任务完成后每句"啊好吧/嗯好的"都触发
# 一次闲聊路全盘复述天气+湖人，循环播报停不下来。注意"停"绝不能入表
# ——"你停停停"是控制意图，必须照常走路由。
_ACK_RE = re.compile(
    r"^[嗯呃啊哦噢诶哎唉呀哟呦哈嘿好的对是行呢吧的了嘛呗"
    r"\s，。？！、,.?!…]+$")

# 提交门轮询间隔（秒）。只读时间戳，无锁竞争。
GRACE_POLL_S = 0.05
# 路由失败时的固定澄清话术（§6.2：超时/失败必须有终态并释放队列）
ROUTER_FAIL_TEXT = "这个我一时没理清要做什么，你能再说一遍吗？"
# 无声兜底（2026-11-06 listening 卡死事故家族）：receipts 模式提交了计划
# 却一个字都没上口（模板空洞，如列表查询）——控制轮绝不能无声结束。
SILENT_TURN_ACK_TEXT = "好的。"
# UI 泵轮询间隔（秒）：V2 任务面板（task_update）只读 store，不进控制平面
UI_POLL_S = 1.0
# V2 execution_status -> 前端 task_update 的 status 词汇（未知值透传，
# 前端 TASK_STATUS_LABEL 兜底显示原文）
_UI_STATUS_MAP = {
    "queued": "queued", "running": "running", "completed": "completed",
    "failed": "failed", "canceled": "canceled", "superseded": "canceled",
    "cancel_requested": "running", "blocked_dependency": "queued",
}


class AgentFrontendAdapter:
    def __init__(self, session_id: str, runtime, router,
                 snapshot_builder: Callable,
                 grace_s_fn: Callable[[], float],
                 last_voice_ts_fn: Callable[[], float],
                 speak_fn: Callable,
                 can_deliver_fn: Callable[[], bool],
                 chitchat_fn: Optional[Callable] = None,
                 generate_fn: Optional[Callable] = None,
                 history_fn: Optional[Callable] = None,
                 mode: str = "shadow",
                 grace_poll_s: float = GRACE_POLL_S,
                 shadow_idle_wait_s: float = 30.0,
                 on_event: Optional[Callable] = None,
                 release_gen_fn: Optional[Callable[[int], None]] = None,
                 ui_poll_s: float = UI_POLL_S):
        """runtime: AgentRuntimeV2；router: PlanRouter；snapshot_builder:
        (user_text, request_id, history) -> snapshot dict（agent.context）。
        on_event(event, data)：前端任务面板等 UI 通知——传入即启动 UI 泵
        （轮询 store.task_panel_rows，diff 后推 task_update，与 legacy
        面板同一消息契约；2026-11-06 enforce 实测反馈：V2 任务在面板上
        不可见）。
        release_gen_fn(gen_sn)：释放收音循环在 turn commit 时置位的
        gen_in_progress（app._end_generation，gen_serial 不匹配时为 no-op）。
        enforce 轮的生命周期跨越路由(~5s)到上口结束，期间标志由本 worker
        持有；worker 结束（含无声轮）必须释放，否则 gen_in_progress 永久
        True，后台交付被 _agent_can_deliver 永久拒绝（2026-11-06 事故）。
        """
        self.session_id = session_id
        self.runtime = runtime
        self.router = router
        self.snapshot_builder = snapshot_builder
        self.grace_s_fn = grace_s_fn
        self.last_voice_ts_fn = last_voice_ts_fn
        self.speak_fn = speak_fn
        self.can_deliver_fn = can_deliver_fn
        self.chitchat_fn = chitchat_fn
        self.generate_fn = generate_fn
        self.history_fn = history_fn or (lambda: [])
        self.mode = mode
        self.grace_poll_s = grace_poll_s
        self.shadow_idle_wait_s = shadow_idle_wait_s
        self.on_event = on_event
        self.release_gen_fn = release_gen_fn
        self.ui_poll_s = ui_poll_s
        self._utterance_lock = threading.Lock()   # 串行化 utterance 处理
        self._stop = threading.Event()
        # 当前挂起的控制轮线程（一次只有一个；新话轮到来时旧线程自行退出）。
        # turn dict 由 worker 自己持有引用：self._active_turn 会被新话轮重绑，
        # 旧 worker 若读它就永远读到新 dict——2026-11-06 湖人双任务事故里
        # abandoned 标志因此是死代码（置了没人看见）。
        self._active_turn: dict = {}
        self._turn_seq = 0
        if on_event is not None:
            threading.Thread(
                target=self._ui_pump, daemon=True,
                name=f"agent-v2-ui-{self.session_id[:8]}").start()

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

    # ------------------------------------------------------------------ UI

    def _ui_pump(self):
        """V2 任务面板泵：轮询 store.task_panel_rows，diff 后经 on_event 推
        task_update（与 legacy 面板同一消息契约 {task_id, goal, status,
        progress, result}）。只读，绝不影响控制平面；store 关闭后静默退出。"""
        seen = {}
        while not self._stop.wait(self.ui_poll_s):
            try:
                rows = self.runtime.store.task_panel_rows(self.session_id)
            except Exception:  # noqa: BLE001 — store 可能已随会话关闭
                continue
            for r in rows:
                status = _UI_STATUS_MAP.get(
                    r["execution_status"], r["execution_status"])
                if status == "completed" and r.get("delivered"):
                    status = "delivered"
                key = (status, r.get("progress", ""), r.get("result", ""))
                if seen.get(r["task_id"]) == key:
                    continue
                seen[r["task_id"]] = key
                try:
                    self.on_event("task_update", {
                        "task_id": r["task_id"], "goal": r.get("goal", ""),
                        "status": status, "progress": r.get("progress", ""),
                        "result": r.get("result", "")})
                except Exception:  # noqa: BLE001 — UI 绝不能拖垮 adapter
                    logger.exception("[%s][v2] task_update emit failed",
                                     self.session_id[:8])

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
                         source_commit_id: Optional[str] = None,
                         gen_sn: Optional[int] = None):
        """enforce 模式：V2 控制器接管本轮控制决策。app 在 ASR 文本确定、
        自回声过滤与话轮门之后调用；不阻塞收音线程（路由在后台跑）。

        gen_sn：收音循环在 commit 时 _begin_generation 返回的序号——
        本 worker 全权负责它的释放（上口会 bump 序号自行清理，无声轮
        在 finally 里释放；被更新的话轮取代时 gen_serial 已变，释放
        自然成为 no-op）。

        流程（§6/§7，2026-11-06 双路并行）：
          1. 上一话轮的计划还没过 grace 门 → 作废（续说场景，§6.2）
          2. 闲聊路（chitchat_fn）与路由并行启动：闲聊路过 grace 门后
             生成+流式上口，负责本轮常规口播
          3. submit_request（路由，3-6s）→ 计划挂起 → 等 grace 门开
             → commit（闲聊路活着时受理/状态类回执静音；结果重播不跳——交付泵单一出口）
          4. 纠错/失败类回执上口；闲聊路失败时 clarify/表达轮兜底
        """
        with self._utterance_lock:
            self._turn_seq += 1
            turn_seq = self._turn_seq
            prev_turn = self._active_turn
            prev = prev_turn.get("thread")
            prev_alive = prev is not None and prev.is_alive()
            # 纯应答吸收（_ACK_RE 注释）：上一轮活着或面板有任务时一句
            # "嗯/好的"既不作废其挂起计划也不重路由、不另起闲聊路。
            # 完全空闲（无活跃轮、无任务）时照常走完整路径——孤零零
            # 一句应答也该有人接话。
            if (text.strip() and _ACK_RE.match(text.strip())
                    and (prev_alive
                         or self.runtime.pending_exists(self.session_id)
                         or self._panel_has_tasks())):
                logger.info("[%s][v2] pure ack absorbed: %s",
                            self.session_id[:8], text)
                return
            if (prev is not None and prev.is_alive()
                    and not prev_turn.get("committed")):
                # 上一计划未提交：续说作废（§6.2 旧计划作废，重新路由）。
                # 已提交轮不再 discard——pending 早已弹出，只会留下一条空的
                # "discarded" route_log 噪声行。abandoned 置在 prev_turn
                # （worker 自己持有的 dict）上才真正生效。
                self.runtime.discard_pending(self.session_id,
                                             reason="continuation")
                prev_turn["abandoned"] = True
                # 闲聊路一并作废：停生成（已发射的音频由新话轮的上口
                # interrupt 掐掉，与回执让路同一套语义）
                prev_stop = prev_turn.get("chitchat_stop")
                if prev_stop is not None:
                    prev_stop.set()
                # 续说合并：被作废片段的文本并入新句一起重路由（镜像 LLM
                # 侧按句 groupby 的合并语义——碎片单独路由时快脑只能对着
                # 半句话脑补 goal，2026-11-06 湖人任务就是这么造出来的）。
                prev_text = (prev_turn.get("text") or "").strip()
                if prev_text:
                    text = prev_text + "\n" + text

        turn = {"thread": None, "abandoned": False, "committed": False,
                "text": text, "seq": turn_seq,
                # 闲聊路（voice lane）状态：stop 掐生成；rc 三态与 speak_fn
                # 一致；failed=异常（仲裁回退旧路径）；done=线程收尾
                "chitchat_stop": threading.Event(),
                "chitchat_rc": None, "chitchat_failed": False,
                "chitchat_started": False,
                "chitchat_done": threading.Event()}
        th = threading.Thread(
            target=self._turn_worker, daemon=True,
            name=f"agent-v2-turn-{self.session_id[:8]}",
            args=(turn, text, turn_trigger, source_commit_id, gen_sn))
        turn["thread"] = th
        self._active_turn = turn
        th.start()

    def user_stop(self):
        """用户手动停止（duplex_stop）：掐当前闲聊路、作废旧轮、作废挂起
        计划、停后台交付泵。不动 store——会话还活着，用户下一句话照常
        接管（handle_utterance 每 turn 都会 start_background 幂等补启动）。
        2026-11-06 实测事故：手动停止后闲聊路仍把生成完的整段播完、交付
        泵还在等下一个空隙补播结果——用户听感是"停不下来"。"""
        with self._utterance_lock:
            turn = self._active_turn
            turn["abandoned"] = True
            stop = turn.get("chitchat_stop")
            if stop is not None:
                stop.set()
        try:
            self.runtime.discard_pending(self.session_id, reason="user_stop")
        except Exception:  # noqa: BLE001
            logger.exception("[%s][v2] user_stop discard failed",
                             self.session_id[:8])
        try:
            # 只停泵不关 store（stop_background 会 store.close，会话还
            # 要继续用）；下一话轮 start_background 幂等重启
            self.runtime.scheduler.stop()
            self.runtime.delivery.stop()
        except Exception:  # noqa: BLE001
            logger.exception("[%s][v2] user_stop pump stop failed",
                             self.session_id[:8])
        logger.info("[%s][v2] user stop: chitchat cut, pending discarded, "
                    "delivery pump stopped", self.session_id[:8])

    def _release_gen(self, gen_sn):
        """释放 commit 时置位的 gen_in_progress（无声轮/被取代轮）。
        绝不抛出——释放失败也不能拖垮 worker 收尾。"""
        if gen_sn is None or self.release_gen_fn is None:
            return
        try:
            self.release_gen_fn(gen_sn)
        except Exception:  # noqa: BLE001
            logger.exception("[%s][v2] gen flag release failed",
                             self.session_id[:8])

    def _panel_has_tasks(self) -> bool:
        """面板上还有任务（含已完成）→ 用户的应答大概率是在回应任务
        交付，吸收之（2026-11-07 实测：每句"嗯好的"都触发一次闲聊路
        全盘复述，循环播报停不下来）。store 已关等异常按无任务处理。"""
        try:
            return bool(self.runtime.store.task_panel_rows(self.session_id))
        except Exception:  # noqa: BLE001
            return False

    def _turn_worker(self, turn, text, turn_trigger, source_commit_id, gen_sn):
        try:
            self._turn_worker_inner(turn, text, turn_trigger,
                                    source_commit_id)
        except Exception:  # noqa: BLE001 — 控制轮失败必须落到固定澄清话术
            logger.exception("[%s][v2] turn worker crashed",
                             self.session_id[:8])
            self._speak_safe(ROUTER_FAIL_TEXT)
        finally:
            # commit 时收音循环置位的 gen_in_progress 由本 worker 释放：
            # 上口轮的 _speak 已 bump gen_serial 自行清理（此处 no-op），
            # 无声轮若不释放则标志永久 True → 后台交付被永久拒绝
            # （2026-11-06 listening 卡死事故的致命一环）。
            self._release_gen(gen_sn)

    def _chitchat_worker(self, turn):
        """闲聊路（voice lane）：grace 门开后生成 + 流式上口，与路由路
        并行。rc 三态与 speak_fn 一致；异常记 chitchat_failed（仲裁时
        回退旧路径——回执上口/表达轮兜底，方案 C）。"""
        rc = None
        try:
            # 与路由路同一个 grace 门（§6.1：不复制第二套计时阈值）——
            # 续说窗口内绝不开口，否则续说会被半句闲聊打断
            grace_s = float(self.grace_s_fn() or 0.0)
            if grace_s > 0:
                while True:
                    if self._stop.is_set() or turn.get("abandoned"):
                        return
                    if time.time() - self.last_voice_ts_fn() >= grace_s:
                        break
                    time.sleep(self.grace_poll_s)
            if self._stop.is_set() or turn.get("abandoned"):
                return
            stop_event = turn["chitchat_stop"]
            gen = self.chitchat_fn(stop_event)
            rc = self.speak_fn(gen, stop_event)
        except Exception:  # noqa: BLE001 — 闲聊路失败回退回执路径
            logger.exception("[%s][v2] chitchat lane failed",
                             self.session_id[:8])
            turn["chitchat_failed"] = True
            rc = None
        finally:
            turn["chitchat_rc"] = rc
            turn["chitchat_done"].set()

    @staticmethod
    def _chitchat_alive(turn) -> bool:
        """仲裁用：闲聊路会开口/已开口。running 或已播出（True/False，
        含播出后被掐）都算——受理类回执的口播职责已移交闲聊路。"""
        if not turn.get("chitchat_started") or turn.get("chitchat_failed"):
            return False
        if turn["chitchat_done"].is_set():
            return turn.get("chitchat_rc") in (True, False)
        return True

    def _turn_worker_inner(self, turn, text, turn_trigger, source_commit_id):
        # 闲聊路先行启动（与路由并行——提速的全部来源）：grace 门后生成，
        # 首句 ~1s 即流式进 TTS，不等路由裁决
        if self.chitchat_fn is not None:
            turn["chitchat_started"] = True
            threading.Thread(
                target=self._chitchat_worker, daemon=True,
                name=f"agent-v2-chat-{self.session_id[:8]}",
                args=(turn,)).start()
        res = self.runtime.submit_request(
            self.session_id, text, source_commit_id=source_commit_id,
            request_seq=turn.get("seq"))
        if not res.get("ok"):
            # 路由失败/超时：终态 + 固定澄清（§6.2），队列已释放。
            # 闲聊路活着时不再说——本轮口播已由它负责（纯闲聊轮路由失败
            # 完全无感才是对的）；代价是任务轮的"我去查"可能没有后台任务
            # 兑现（罕见：传输故障本来就已是降级态，用户重问即恢复）。
            logger.info("[%s][v2] route failed: %s", self.session_id[:8],
                        res.get("error"))
            if not self._chitchat_alive(turn):
                self._speak_safe(ROUTER_FAIL_TEXT)
            return
        # grace 提交门（§6.1）：窗口内只挂起；复用前端同一阈值
        grace_s = float(self.grace_s_fn() or 0.0)
        if grace_s > 0:
            deadline = time.time() + grace_s + 5.0   # 门最多等 grace+5s
            while time.time() < deadline:
                if self._stop.is_set():
                    return
                if turn.get("abandoned"):
                    return   # 续说：旧计划已作废，新话轮线程接管
                if not self.runtime.pending_exists(self.session_id):
                    return  # 已被作废/提交
                if time.time() - self.last_voice_ts_fn() >= grace_s:
                    break
                time.sleep(self.grace_poll_s)
        if (self._stop.is_set() or turn.get("abandoned")
                or not self.runtime.pending_exists(self.session_id)):
            return
        chitchat_alive = self._chitchat_alive(turn)
        out = self.runtime.commit_pending(self.session_id,
                                          chitchat_spoke=chitchat_alive)
        if out is None:
            return  # 门又关上（续说抢先）——计划仍挂起，交由下一轮处理
        turn["committed"] = True
        plan = out.get("plan")
        ops = [getattr(o, "op", "") for o in
               (plan.operations if plan is not None else [])]
        # stop-only：闲聊路的纪律是"停一下→不出声"，commit 后把生成器
        # 也掐掉（防模型不守规矩多话；已发射音频由 stop_speech 交付泵处理）
        if ops == ["stop_speech"]:
            turn["chitchat_stop"].set()
        # 回执话术：程序模板，不让模型自由解释（§13.1）。闲聊路活着时
        # 受理/状态类已被 commit_pending 过滤（QUIET_IF_VOICED_OUTCOMES），
        # 这里剩下的只剩纠错/失败类——它们必须上口（闲聊路不知道拒绝原因）
        for receipt_text in out.get("texts") or []:
            if not self._speak_safe(receipt_text):
                return   # 打断即终态，本轮后续回执不再上口
        # 澄清话术（clarify 模式）：闲聊路按同一条"没说清要反问"纪律自己
        # 问；只有闲聊路失败/缺席时路由的反问才上口（兜底，方案 C）
        if (plan is not None and plan.reply_mode == "clarify"
                and plan.clarification):
            if not self._chitchat_alive(turn):
                self._speak_safe(plan.clarification)
            return
        # 表达轮兜底（方案 C）：direct/mixed 的 direct_question 只在闲聊路
        # 失败/缺席时走旧串行路径——绝不让闲聊路故障变成无声轮（§7.3：
        # 不能因为本轮有一个控制操作就吞掉另一个问题）
        if (self.generate_fn is not None and plan is not None
                and plan.reply_mode in ("direct", "mixed")
                and (plan.direct_question or plan.reply_mode == "direct")
                and not self._chitchat_alive(turn)):
            stop_event = threading.Event()
            try:
                gen = self.generate_fn(stop_event)
                rc = self.speak_fn(gen, stop_event)
                logger.info("[%s][v2] expression round rc=%s",
                            self.session_id[:8], rc)
            except Exception:  # noqa: BLE001
                logger.exception("[%s][v2] expression round failed",
                                 self.session_id[:8])
            return
        # 无声兜底：计划已提交却一个字都没上口（模板空洞，如列表查询）。
        # 闲聊路还活着/已开口就不补——它（可能还在生成）就是本轮的口播。
        # stop_speech 除外——它的回执就是安静本身，再说一个字就自相矛盾。
        if not (out.get("texts") or []) and not self._chitchat_alive(turn):
            if "stop_speech" not in ops:
                self._speak_safe(SILENT_TURN_ACK_TEXT)

    def _speak_safe(self, text) -> bool:
        """单条固定话术上口。返回是否完整播出（False/None 都不算）。"""
        stop_event = threading.Event()
        try:
            rc = self.speak_fn(text, stop_event)
        except Exception:  # noqa: BLE001
            logger.exception("[%s][v2] speak failed", self.session_id[:8])
            return False
        return rc is True
