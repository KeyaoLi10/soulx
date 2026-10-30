"""Deterministic Agent Runtime: task management, agenda, delivery arbitration.

Per-session component owned by app.py. The two brains never talk to each
other directly — everything flows through here:

  fast brain (app pipeline) --delegate_task/cancel_task--> Runtime
  Runtime --thread--> slow brain (agent/slow_brain.py, :6008 thinking=on)
  slow brain --events--> Runtime task state
  Runtime --agenda + arbiter--> deliver callback (app speaks the result)

The Runtime holds no model of its own: admission rules, task bookkeeping and
delivery timing are plain Python. Speech generation itself is delegated to
the app-provided `deliver` callback (which reuses the normal pipeline's
segmenting + TTS + interruption machinery), so there is exactly one speech
outlet in the whole system.

Delivery timing ("agenda"): a completed result waits until `can_deliver()`
says the user is idle (not generating, playback finished, no voice activity
for a moment) — the user can keep chatting while the slow brain works, and
the result only arrives at a natural gap.
"""
import re
import threading
import time

from agent.slow_brain import SlowBrain
from clients.llm_client import TOOL_QUIET  # [[quiet]] success marker this
# runtime prefixes its success observations with — see the double-
# acceptance fix of 2026-10-30 in llm_client.generate_agent_with_history.

MAX_RUNNING_TASKS = 2
MAX_DELIVERY_ATTEMPTS = 2
ARBITER_POLL_S = 0.5
# after this many seconds a completed-but-undeliverable result is delivered
# anyway at the next gap, even if the task finished long ago (no expiry: a
# late result is still worth hearing — task.md's "改口后旧结果晚到" case is
# handled by cancel, not by silent expiry)

# Tools offered to the FAST brain via /agent_chat (thinking=off). The LLM
# only ever sees these two; slow-brain tools (search/repl/...) are internal
# to the slow loop.
FAST_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "delegate_task",
            "description": (
                "Delegate a complex task to the background agent. Use for "
                "anything that needs web search, careful multi-step "
                "reasoning, or real computation — NOT for casual chat. "
                "Returns immediately with a task id; the result will be "
                "reported to the user later. After calling this, briefly "
                "tell the user you are working on it. Never invent the "
                "result yourself."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "goal": {
                        "type": "string",
                        "description": (
                            "self-contained task description in the user's "
                            "language, including all details needed to "
                            "execute it without further context"
                        ),
                    },
                    "context": {
                        "type": "string",
                        "description": (
                            "optional conversation context the task needs "
                            "(what was discussed, constraints the user gave)"
                        ),
                    },
                },
                "required": ["goal"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_task",
            "description": (
                "Cancel background task(s). Use when the user changes their "
                "mind or no longer wants the answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "task id like T3; omit to cancel ALL tasks",
                    },
                },
                "required": [],
            },
        },
    },
]

# appended to the persona system prompt when the agent is enabled
AGENT_SYSTEM_ADDON = (
    "你有一个后台任务助手（慢脑），可以把复杂任务委派给它后台执行：\n"
    "- 【实时性问题必须委派，严禁凭记忆作答】以下几类问题你的内在知识必然过时，"
    "无论你多确定都必须调用 delegate_task 委派查询，绝不能直接回答："
    "今天几号/星期几/现在几点、天气、股价/指数/汇率/油价/物价、新闻时事、"
    "体育比赛结果，以及一切带“今天/现在/最新/最近/刚刚”字样的时效性问题。"
    "拿不准算不算实时问题时一律委派——委派只慢几秒，凭记忆硬答会把过时的错误"
    "数字直接说给用户听。\n"
    "- 需要联网搜索、查资料、查天气/新闻/价格、仔细计算或多步推理的请求，必须调用 "
    "delegate_task 委派，goal 里写全所有细节（自包含，后台看不到你们的对话）。\n"
    "- 【最重要】凡是你对用户说了“我去查/我查一下/稍等”这类话，同一次回复里必须已经"
    "调用了 delegate_task。绝不允许口头答应去查却不委派——那样用户永远等不到结果。\n"
    "- 委派后如果本轮还没告知用户，用一两句话自然地告诉用户你在处理、"
    "稍后主动汇报；如果调用前已经说过受理话术，就不要再重复一遍。"
    "绝不编造结果。\n"
    "- 【重要】后台任务的结果交付之前，你完全不知道结果内容：此时绝不许说出任何"
    "具体数据或事实（温度、数字、日期、排名等），也绝不许声称自己有定位、传感器、"
    "本地数据等能力——只能说“还在查，稍后告诉你”。用户追问时如实说明后台还在处理。\n"
    "- 【同样重要】用户改口说“算了/不用查了/取消吧”这类话时，同一次回复里必须已经"
    "调用了 cancel_task（不填 task_id 即取消全部任务），绝不允许只口头说取消却不调用"
    "——那样任务会在后台继续空跑。\n"
    "- 简单闲聊、问候、常识性回答不要委派，直接回答。\n"
    "- 【闲聊不汇报】后台任务运行期间，用户聊别的话题时，自然地陪聊，"
    "绝不要主动提及后台任务或其进度（“还在查”“快好了”这类话也不要说）。"
    "只有当用户主动问到任务、进度或结果（如“查得怎么样了”“好了吗”）时，"
    "系统才会把任务状态提示给你，此时如实简短汇报。\n"
    "- 汇报结果时必须依据系统提示里的真实结果，关键数字和日期不得更改。"
)


class AgentTask:
    __slots__ = (
        "id", "goal", "status", "created_ts", "finished_ts",
        "progress", "result", "error", "cancel_event", "attempts",
    )

    def __init__(self, task_id, goal):
        self.id = task_id
        self.goal = goal
        self.status = "queued"        # queued|running|completed|failed|canceled
        self.created_ts = time.time()
        self.finished_ts = None
        self.progress = ""            # last progress description
        self.result = None
        self.error = None
        self.cancel_event = threading.Event()
        self.attempts = 0             # delivery attempts so far

    def view(self):
        return {
            "task_id": self.id, "goal": self.goal, "status": self.status,
            "progress": self.progress, "result": self.result,
        }


class AgentRuntime:
    """Owns the session's background tasks and the delivery agenda."""

    def __init__(self, client_id, emit, can_deliver, deliver):
        """
        emit(client_id_fixed, "task_update", {...})  — WS push to frontend
        can_deliver() -> bool  — app-side "user idle, safe to speak"
        deliver(entry) -> bool — app-side speech generation; True = fully
                                spoken, False = interrupted (re-enqueued)
        """
        self.client_id = client_id
        self.emit = emit
        self.can_deliver = can_deliver
        self.deliver = deliver
        self.slow = SlowBrain()
        self.lock = threading.Lock()
        self.tasks = {}                # id -> AgentTask
        self.agenda = []               # completed entries awaiting speech
        self._delivering = False
        self._next_id = 0
        self._stop = threading.Event()
        self._arbiter = None

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def start(self):
        if self._arbiter and self._arbiter.is_alive():
            return
        self._arbiter = threading.Thread(
            target=self._arbiter_loop, daemon=True, name=f"agent-arbiter-{self.client_id[:8]}"
        )
        self._arbiter.start()

    def shutdown(self):
        self._stop.set()
        with self.lock:
            tasks = list(self.tasks.values())
            self.agenda.clear()
        for t in tasks:
            t.cancel_event.set()

    # ------------------------------------------------------------------
    # fast-brain tool entry point (synchronous, runs in the pipeline thread)
    # ------------------------------------------------------------------
    def handle_tool(self, name, args):
        args = args or {}
        if name == "delegate_task":
            return self.delegate(args.get("goal", ""), args.get("context", ""))
        if name == "cancel_task":
            return self.cancel(args.get("task_id") or None)
        return f"unknown tool: {name}"

    # ------------------------------------------------------------------
    def delegate(self, goal, context=""):
        goal = (goal or "").strip()
        if not goal:
            return "任务创建失败：goal 为空。请向用户澄清需要做什么。"
        if not self.slow.available():
            return ("后台任务助手当前不可用（慢脑服务未运行）。"
                    "请直接告诉用户暂时无法处理这个复杂任务，可以先用你自己的知识简答。")
        with self.lock:
            running = [t for t in self.tasks.values() if t.status in ("queued", "running")]
            if len(running) >= MAX_RUNNING_TASKS:
                ids = "、".join(t.id for t in running)
                return (f"后台已有 {len(running)} 个任务在运行（{ids}），本次未能创建新任务。"
                        "请告诉用户稍等，或问是否要取消现有任务。")
            self._next_id += 1
            task = AgentTask(f"T{self._next_id}", goal)
            self.tasks[task.id] = task
        threading.Thread(
            target=self._run_task, args=(task, context or ""),
            daemon=True, name=f"slow-brain-{task.id}",
        ).start()
        return (
            TOOL_QUIET
            + f"任务已创建：{task.id}，正在后台执行。"
            "如果本次回复尚未对用户说过话，请用一两句话自然地告诉用户你在"
            "处理这件事、稍后会主动汇报；如果已经说过受理话术（“我去查/"
            "正在查/稍后告诉你”一类），不要再重复，直接结束本次回复。"
            "任何时候都不要编造结果。"
        )

    def cancel(self, task_id=None):
        with self.lock:
            if task_id:
                targets = [self.tasks.get(task_id)]
                targets = [t for t in targets if t]
                if not targets:
                    return f"没有找到任务 {task_id}。当前任务：{self._active_summary() or '无'}。"
            else:
                targets = [t for t in self.tasks.values() if t.status in ("queued", "running")]
            for t in targets:
                if t.status in ("queued", "running"):
                    t.status = "canceled"
                    t.finished_ts = time.time()
                    t.cancel_event.set()
            # a completed-but-undelivered task is also "no longer wanted"
            if not task_id:
                self.agenda = [e for e in self.agenda if e["task_id"] not in
                               {t.id for t in targets}]
        for t in targets:
            self._emit_task(t)
        if not targets:
            return "没有正在运行的后台任务。"
        return (
            TOOL_QUIET + "已取消任务：" + "、".join(t.id for t in targets)
            + "。如果本次回复尚未对用户说过话，请自然地告知用户；已经说过"
            "取消确认的话就不要再重复。"
        )

    def task_view(self):
        """Compact task view injected into the fast brain's context (None
        when there is nothing worth mentioning). Delivered/failed tasks are
        excluded: their results already reached the conversation history as
        the delivery note, so repeating them here is noise."""
        with self.lock:
            active = [t for t in self.tasks.values()
                      if t.status in ("queued", "running")]
        if not active:
            return None
        lines = []
        for t in active:
            age = int(time.time() - t.created_ts)
            lines.append(f"- {t.id}（已运行 {age} 秒）：{t.goal}")
        return "当前后台任务：\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    # deterministic cancel-intent detection
    # ------------------------------------------------------------------
    # Measured failure mode (scripts/cancel_repro.py, 2026-10-30): the fast
    # brain UNDERSTANDS "算了不用查了" perfectly but does not CALL
    # cancel_task — it narrates the cancellation as already done (0/4 with a
    # strengthened prompt that fixes the identical delegate-side failure).
    # Per this system's own design (deterministic Runtime arbitrates, the
    # brains never act on each other's behalf implicitly), the Runtime
    # detects the retraction itself, cancels BEFORE generation, and tells
    # the fast brain what happened so its confirmation is grounded in fact.
    _CANCEL_NEG_RE = re.compile(r"别取消|不要取消|不准取消|不能取消|继续查|接着查|还是查|先别")
    _CANCEL_RE = re.compile(r"算了|不用查|不查了|不要查了|别查了|取消|不用了|不需要了")

    def auto_cancel(self, user_text):
        """If the user's words clearly retract a running background task,
        cancel it now. Returns a system note for the fast brain's context
        (to be injected alongside the task view), or None."""
        text = (user_text or "").strip()
        if not text or self._CANCEL_NEG_RE.search(text):
            return None
        with self.lock:
            active = [t for t in self.tasks.values()
                      if t.status in ("queued", "running")]
        if not active or not self._CANCEL_RE.search(text):
            return None
        ids = "、".join(t.id for t in active)
        self.cancel()  # cancels all active tasks, emits task_update
        return (f"[系统提示：已根据用户的话自动取消后台任务 {ids}。"
                "请用一两句话自然地向用户确认，不要编造其他结果。]")

    # ------------------------------------------------------------------
    # deterministic real-time-question detection
    # ------------------------------------------------------------------
    # Measured failure mode (2026-10-30 live session): asked "今天是几号",
    # the fast brain answered a confident but WRONG date from parametric
    # memory ("2024年10月12日") without ever calling delegate_task — the
    # addon prompt alone doesn't reliably trigger the call. Same design as
    # auto_cancel: the Runtime detects the pattern itself and injects a
    # grounded note; the fast brain still makes the actual tool call.
    # Time words alone are not enough ("今天心情不错" is small talk) — the
    # pattern requires a realtime noun (weather/quotes/news/date...) near a
    # time marker, or the noun on its own.
    _REALTIME_RE = re.compile(
        r"(今天|现在|目前|最新|最近|刚刚).{0,10}"
        r"(几号|星期|礼拜|几点|天气|温度|气温|股价|指数|汇率|油价|金价|新闻|价格|行情)"
        r"|(天气|预报|气温|股价|指数|汇率|油价|金价|新闻|时事|赛果|比分|几号|星期几|礼拜几|现在几点)"
        r"|(查|问)[^，。？！]{0,6}(今天|日期)"
    )

    def realtime_note(self, user_text):
        """If the user asks a real-time question and nothing is running yet,
        return a note telling the fast brain it MUST delegate this one."""
        text = (user_text or "").strip()
        if not text or not self._REALTIME_RE.search(text):
            return None
        with self.lock:
            active = [t for t in self.tasks.values()
                      if t.status in ("queued", "running")]
        if active:
            # already working on something — the mention-gated task view
            # covers progress questions; don't force a second delegation
            return None
        return ("[系统提示：用户的问题涉及实时性信息（日期/天气/行情/新闻等），"
                "你的内在知识必然过时。本次回复必须调用 delegate_task 委派后台查询，"
                "结果交付之前不要向用户说出任何具体数字或事实，先简短说明你去查一下。]")

    # ------------------------------------------------------------------
    # deterministic delegate safety net (auto_delegate)
    # ------------------------------------------------------------------
    # Measured failure mode (2026-10-30 14:51, session ec822c0a): the user
    # asked "再查查今天的天气", realtime_note injected the MUST-delegate
    # instruction, and the fast brain STILL only said "好哒，我去看看今天
    # 天气怎么样。" without ever calling delegate_task — no task existed,
    # the panel stayed empty, and a minute later the model fabricated a
    # weather answer from parametric memory. Third confirmation (after
    # auto_cancel and the double-acceptance probe) that behavior
    # constraints cannot live in the prompt alone. Mirror of auto_cancel:
    # the Runtime creates the task AFTER generation when the spoken reply
    # committed to checking but no delegate actually ran — the acceptance
    # speech is already out, so the task makes it true instead of leaving
    # it a lie. task_ids_before is the pre-generation task-id snapshot the
    # caller takes, distinguishing "delegate ran this turn" from "task
    # table unchanged".
    _ACCEPT_RE = re.compile(
        r"我去查|我去看看|我去问|这就去|马上(帮)?你?查|帮你查|帮你看看"
        r"|正在查|稍后(告诉|汇报|给你)|等(一下|我)查"
    )
    # sentence-level question guard: the acceptance commitment must live
    # in a NON-question sentence. Measured 2026-10-30 15:07 (session
    # 164595f0): "好的，我先帮你查一下今天的信息～今天几号了？" — the
    # model committed, then asked the user the date (nonsense, but the
    # commitment stands); a whole-reply "？" guard let that case slip
    # through the net. Conversely "我可以帮你查，你想查哪里的呀？" is one
    # question sentence offering, not committing — still skipped.
    _SENT_SPLIT_RE = re.compile(r"(?<=[。！？!?～\n])")

    def _accepts_in_speech(self, reply):
        non_q = "".join(
            s for s in self._SENT_SPLIT_RE.split(reply)
            if not s.rstrip().endswith(("？", "?"))
        )
        return bool(self._ACCEPT_RE.search(non_q))

    def auto_delegate(self, user_text, reply_text, task_ids_before):
        """If the fast brain's spoken reply committed to checking
        ("我去查…") but no delegate_task ran this turn, create the task
        now. Returns the new task id, or None."""
        text = (user_text or "").strip()
        reply = (reply_text or "").strip()
        if not text or not reply:
            return None
        if not self._accepts_in_speech(reply):
            return None
        if not re.search(r"查|搜|帮我|算", text):
            return None
        with self.lock:
            if set(self.tasks.keys()) != set(task_ids_before):
                return None  # a delegate DID run this turn
            active = [t for t in self.tasks.values()
                      if t.status in ("queued", "running")]
        if active:
            return None
        before = set(self.tasks.keys())
        self.delegate(text, "")
        with self.lock:
            new = sorted(set(self.tasks.keys()) - before)
        return new[-1] if new else None

    # ------------------------------------------------------------------
    # slow brain side
    # ------------------------------------------------------------------
    def _run_task(self, task, context):
        def on_event(evt):
            kind = evt.get("type")
            with self.lock:
                if kind == "started":
                    if task.status == "queued":
                        task.status = "running"
                elif kind == "progress":
                    task.progress = evt.get("desc", "")
                elif kind == "completed":
                    if task.status in ("queued", "running"):
                        task.status = "completed"
                        task.result = evt.get("result")
                        task.finished_ts = time.time()
                        self.agenda.append({
                            "task_id": task.id, "goal": task.goal,
                            "result": task.result, "attempts": 0,
                        })
                elif kind == "failed":
                    if task.status in ("queued", "running"):
                        task.status = "failed"
                        task.error = evt.get("error")
                        task.finished_ts = time.time()
                        # a failed task must still be spoken: the user was
                        # promised a result, and silence would leave them
                        # waiting for one that is never coming
                        self.agenda.append({
                            "task_id": task.id, "goal": task.goal,
                            "result": None, "error": task.error, "attempts": 0,
                        })
                elif kind == "canceled":
                    if task.status in ("queued", "running"):
                        task.status = "canceled"
                        task.finished_ts = time.time()
            self._emit_task(task)

        self.slow.run_task(task.id, task.goal, context,
                           event_cb=on_event, cancel_event=task.cancel_event)

    def _emit_task(self, task):
        try:
            self.emit("task_update", task.view())
        except Exception:
            pass

    # ------------------------------------------------------------------
    # delivery arbiter
    # ------------------------------------------------------------------
    def _arbiter_loop(self):
        while not self._stop.is_set():
            time.sleep(ARBITER_POLL_S)
            if self._delivering:
                continue
            with self.lock:
                if not self.agenda:
                    continue
                entry = self.agenda[0]
            if not self.can_deliver():
                continue
            with self.lock:
                # re-check: agenda may have changed while we polled
                if not self.agenda or self._delivering:
                    continue
                entry = self.agenda.pop(0)
                entry["attempts"] += 1
                self._delivering = True
            try:
                fully_spoken = self.deliver(entry)
            except Exception as e:
                fully_spoken = True  # don't crash the arbiter on app errors
                import logging
                logging.getLogger().error(f"agent delivery failed: {e}", exc_info=True)
            finally:
                with self.lock:
                    self._delivering = False
            if not fully_spoken and entry["attempts"] < MAX_DELIVERY_ATTEMPTS:
                # interrupted mid-report (user started talking): try again at
                # the next natural gap, but only once more
                with self.lock:
                    self.agenda.insert(0, entry)
            else:
                with self.lock:
                    t = self.tasks.get(entry["task_id"])
                    # both outcomes count as "spoken to the user"
                    if t and t.status in ("completed", "failed"):
                        t.status = "delivered"
                # notify the frontend in every terminal path — the status
                # flip to "delivered" is otherwise never pushed
                with self.lock:
                    t = self.tasks.get(entry["task_id"])
                if t:
                    self._emit_task(t)

    # ------------------------------------------------------------------
    def _active_summary(self):
        act = [f"{t.id}:{t.goal[:20]}" for t in self.tasks.values()
               if t.status in ("queued", "running")]
        return "; ".join(act)
