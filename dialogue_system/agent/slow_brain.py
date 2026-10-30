"""Slow brain: background agent loop for complex, multi-step tasks.

Runs on the dedicated slow LLM instance (:6008) via /agent_chat. Thinking is
off by default (demo latency, see SLOW_THINKING) — set SLOW_BRAIN_THINKING=1
for deep-reasoning mode.
One run_task() = one agent loop:

    build task prompt (goal + context + current time)
      -> stream generation (reasoning is discarded, never spoken)
      -> tool calls parsed out by the server (T chunks) -> execute -> append
         observation -> loop (max MAX_STEPS rounds)
      -> a step with no tool calls is the final answer

Events (via event_cb(task_id, event_dict)) drive the Runtime's task view and
the frontend task panel:
    {"type": "started"}                       task loop began
    {"type": "progress", "desc": str}         one tool call / observation
    {"type": "completed", "result": str}      final answer text
    {"type": "failed", "error": str}
    {"type": "canceled"}

run_task() blocks — the Agent Runtime calls it on a worker thread. Cancel by
setting the cancel_event: checked between stream chunks (the HTTP connection
is dropped, which also stops server-side generation via the disconnect
handler) and before each tool execution.
"""
import datetime
import json
import logging
import os
import threading
import time

import requests

from agent import tools as agent_tools
from clients.llm_client import _LEAD_ROLE_JUNK, strip_role_junk  # same
# occasional role-marker hallucination the fast brain strips (see
# llm_client) — the slow brain's answer is spoken too, so it needs the
# same guard (lead + mid-answer standalone marker lines)

logger = logging.getLogger(__name__)

SLOW_LLM_BASE = "http://127.0.0.1:6008"

# Tool-round budget (2026-10-30, 用户要求"3 轮调用就一定要出结果"): the loop
# runs MAX_STEPS rounds, the LAST of which has tools stripped and forces a
# conclusion — so MAX_STEPS = 4 means 3 tool-calling rounds + 1 forced answer.
# The old 8 was measured dragging simple lookups through speculative
# fetch_page chains; 3 rounds is enough for search→read→calculate tasks, and
# the in-band "已达上限" note makes the model wrap up with what it has.
MAX_STEPS = 4
TASK_TIMEOUT_S = 480          # hard wall-clock budget for one task
MAX_CALLS_PER_ROUND = 4       # hard cap on tool calls in one generation
STREAM_READ_TIMEOUT_S = 120   # no token for this long -> connection error
CONNECT_TIMEOUT_S = 10

# Thinking mode + temperature, tuned 2026-10-30 for demo latency ("工具调用
# 总是很慢"): thinking=on rounds have a heavy-tailed duration (measured 5s to
# 100+s per round on identical tasks — the model sometimes reasons for the
# whole token budget), which is what made delegated tasks take minutes.
# thinking=off + temperature 0.3 measured 8-15s end-to-end on lookup tasks
# with equivalent answer quality. Flip SLOW_BRAIN_THINKING=1 to restore deep
# reasoning for harder task mixes (then per-round budget is 4096 so long
# chains of thought are not truncated into empty answers — a 2048 cap with
# thinking=on produced exactly that: thinking ate the whole budget, no
# answer text came out, the task "completed" with the apology placeholder).
SLOW_THINKING = os.environ.get("SLOW_BRAIN_THINKING", "0") == "1"
TEMPERATURE = 0.3
# generation cap per round. 4096 (old default) let a runaway round stream
# for ~180s; 2048 bounds it at ~90s while staying far above what
# thinking=off rounds use (~300 tokens)
MAX_TOKENS_PER_ROUND = 4096 if SLOW_THINKING else 2048

SLOW_SYSTEM_PROMPT = (
    "你是一个语音对话系统的后台任务代理（慢脑），负责在后台完成前台对话助手委派给你的复杂任务。"
    "你可以多步调用工具（web_search 搜索网页、fetch_page 抓取网页、python_repl 运行 Python 代码、"
    "get_datetime 获取时间），逐步收集信息并推理。规则：\n"
    "1. 每一步要么调用工具继续收集信息，要么直接给出最终答案，不要空转。"
    "思考务必精炼：每轮几十个字内做出决策，不要反复推演。"
    "一轮最多调用两三个工具，优先消化已有结果而不是不断抓取新网页。\n"
    "2. 【最重要】你最多只有 3 轮工具调用机会，用完后必须基于已有信息给出最终答案——"
    "查到的信息已经足够回答任务时，立即在下一轮给出最终答案，"
    "不要再调用工具、不要反复验证。对查天气、查事实这类任务，通常一次搜索的摘要就足够了。\n"
    "3. 最终答案将被转成语音播报给用户：使用与任务描述相同的语言；先给结论，再给关键事实、数据或依据；"
    "字数控制在两百字以内（除非任务明确要求更详细）；不要使用 markdown 格式、列表符号、标题或表情。\n"
    "4. 严格基于工具返回的真实结果作答，工具结果里没有的信息要么继续查证，要么明确说明未找到，"
    "绝不能编造。\n"
    "5. 任何数值计算（算术、日期推算、单位换算）都必须先用 python_repl 算出"
    "结果再作答，不要心算——心算错了用户会直接听到错误数字。\n"
    "6. 如果尝试多步后仍无法完成，给出你能确认的部分结果并说明哪些部分没有查到。"
)


class SlowBrain:
    """Thin client around the slow LLM's /agent_chat — holds no task state."""

    def __init__(self, base_url=SLOW_LLM_BASE):
        self.base_url = base_url.rstrip("/")

    def available(self):
        try:
            r = requests.get(f"{self.base_url}/health", timeout=3)
            return r.ok and r.json().get("loaded")
        except requests.RequestException:
            return False

    # ------------------------------------------------------------------
    def run_task(self, task_id, goal, context="", event_cb=None,
                 cancel_event=None):
        """Blocking agent loop; returns the final answer text or None.

        Never raises — every failure becomes a failed event so the Runtime
        can tell the user in speech instead of dropping the task silently.
        """
        cancel_event = cancel_event or threading.Event()

        def emit(evt):
            evt = dict(evt, task_id=task_id, ts=time.time())
            if event_cb:
                try:
                    event_cb(evt)
                except Exception:
                    pass

        deadline = time.monotonic() + TASK_TIMEOUT_S

        # current time goes in-band: most "what/when is..." tasks need it and
        # it saves a get_datetime round trip
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        user_msg = f"当前时间：{now}\n\n任务：{goal}"
        if context:
            user_msg += f"\n\n相关对话背景：\n{context}"

        messages = [
            {"role": "system", "content": SLOW_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        emit({"type": "started"})
        t_start = time.monotonic()
        logger.info("[slowbrain %s] start: %s", task_id, goal[:80])

        for step in range(MAX_STEPS):
            if cancel_event.is_set():
                emit({"type": "canceled"})
                return None
            if time.monotonic() > deadline:
                emit({"type": "failed", "error": "task timeout"})
                return None

            # last allowed round: no tools, force a conclusion
            last = step == MAX_STEPS - 1
            if last and any(m["role"] == "tool" for m in messages):
                messages.append({
                    "role": "user",
                    "content": "工具调用次数已达上限。请立即基于已有信息给出最终答案。",
                })
            try:
                t_gen = time.monotonic()
                speech, tool_calls = self._generate(
                    messages, tools=None if last else agent_tools.TOOL_SCHEMAS,
                    cancel_event=cancel_event, deadline=deadline,
                )
                logger.info(
                    "[slowbrain %s] round %d/%d gen %.1fs: %d tool call(s), "
                    "speech %d chars, ctx %d chars",
                    task_id, step + 1, MAX_STEPS, time.monotonic() - t_gen,
                    len(tool_calls), len(speech),
                    sum(len(m.get("content") or "") for m in messages),
                )
            except _Canceled:
                emit({"type": "canceled"})
                return None
            except Exception as e:
                logger.info("[slowbrain %s] failed after %.1fs: llm error: %s",
                            task_id, time.monotonic() - t_start, e)
                emit({"type": "failed", "error": f"llm error: {e}"})
                return None

            if not tool_calls:
                result = strip_role_junk(_LEAD_ROLE_JUNK.sub("", speech)).strip()
                if not result:
                    result = "抱歉，这个任务我没能完成。"
                logger.info("[slowbrain %s] completed in %.1fs (%d rounds): %s",
                            task_id, time.monotonic() - t_start, step + 1,
                            result[:80])
                emit({"type": "completed", "result": result})
                return result

            # append the assistant turn exactly as the model produced it
            # (speakable text + its own tool-call XML), then one tool
            # observation per call
            assistant_content = speech + "\n" if speech.strip() else ""
            for name, args in tool_calls:
                assistant_content += _render_tool_xml(name, args)
            messages.append({"role": "assistant", "content": assistant_content.rstrip("\n")})

            # hard cap: a thinking=on round that fires a dozen fetch_page
            # calls bloats the context (4k chars each) and slows every later
            # round — execute the first few, tell the model the rest were cut
            dropped = len(tool_calls) - MAX_CALLS_PER_ROUND
            if dropped > 0:
                tool_calls = tool_calls[:MAX_CALLS_PER_ROUND]

            round_cache = {}
            for name, args in tool_calls:
                if cancel_event.is_set():
                    emit({"type": "canceled"})
                    return None
                if time.monotonic() > deadline:
                    emit({"type": "failed", "error": "task timeout"})
                    return None
                # duplicate calls within one round (measured: the model
                # sometimes fires the identical web_search/fetch_page twice)
                # execute once and reuse the observation — a duplicated slow
                # fetch otherwise burns its full timeout a second time
                cache_key = (name, json.dumps(args, sort_keys=True,
                                              ensure_ascii=False))
                if cache_key in round_cache:
                    observation = round_cache[cache_key]
                else:
                    t_tool = time.monotonic()
                    emit({"type": "progress", "desc": _describe_call(name, args)})
                    observation = agent_tools.execute_tool(name, args)
                    logger.info("[slowbrain %s] tool %s(%.40s) %.1fs -> %d chars",
                                task_id, name,
                                args.get("query") or args.get("url")
                                or args.get("code") or "",
                                time.monotonic() - t_tool, len(observation))
                    round_cache[cache_key] = observation
                    emit({"type": "progress",
                          "desc": f"{_describe_call(name, args)} -> "
                                  f"{len(observation)} 字符结果"})
                messages.append({"role": "tool", "content": observation})
            if dropped > 0:
                messages.append({
                    "role": "tool",
                    "content": (f"本轮只执行了前 {MAX_CALLS_PER_ROUND} 个工具调用，"
                                f"其余 {dropped} 个被忽略。请先消化已有结果。"),
                })

        # unreachable: the last round runs without tools and returns above
        emit({"type": "failed", "error": "max steps exceeded"})
        return None

    # ------------------------------------------------------------------
    def _generate(self, messages, tools, cancel_event, deadline):
        """One /agent_chat round. Returns (speakable_text, [(name, args)]).

        Raises _Canceled if the cancel event fired mid-stream.
        """
        body = {
            "messages": messages,
            "thinking": SLOW_THINKING,
            "max_tokens": MAX_TOKENS_PER_ROUND,
            "temperature": TEMPERATURE,
        }
        if tools:
            body["tools"] = tools
        r = requests.post(
            f"{self.base_url}/agent_chat",
            json=body,
            stream=True,
            timeout=(CONNECT_TIMEOUT_S, STREAM_READ_TIMEOUT_S),
        )
        r.raise_for_status()
        speech_parts, tool_calls = [], []
        try:
            for chunk in r.iter_content(chunk_size=None):
                if cancel_event.is_set():
                    raise _Canceled()
                if time.monotonic() > deadline:
                    raise TimeoutError("task timeout")
                if not chunk:
                    continue
                tag, payload = chunk[:1], chunk[1:].decode("utf-8", errors="replace")
                if tag == b"S":
                    speech_parts.append(payload)
                elif tag == b"T":
                    call = json.loads(payload)
                    tool_calls.append((call["name"], call.get("arguments") or {}))
                # "R" reasoning deltas are intentionally dropped here — the
                # slow brain's thinking is never spoken and never persisted
        finally:
            if cancel_event.is_set():
                r.close()
        return "".join(speech_parts), tool_calls


class _Canceled(Exception):
    pass


def _render_tool_xml(name, args):
    inner = "".join(
        f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in args.items()
    )
    return f"<tool_call>\n<function={name}>\n{inner}</function>\n</tool_call>\n"


def _describe_call(name, args):
    """One short human-readable line for progress events / the task panel."""
    if name == "web_search":
        return f"搜索：{args.get('query', '')}"
    if name == "fetch_page":
        url = str(args.get("url", ""))
        return f"打开网页：{url[:60]}"
    if name == "python_repl":
        return "运行 Python 代码"
    if name == "get_datetime":
        return "查询当前时间"
    return f"调用工具 {name}"
