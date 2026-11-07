import json
import os
import re
import requests
from itertools import groupby
from omegaconf import OmegaConf
from modules.utils.text_utils import split_cn_en

# Wire contract between the agent runtime's tool observations and this
# generation loop: an observation starting with this marker means "this
# tool call succeeded and needs no further spoken follow-up" — when the
# model already spoke this turn (e.g. its acceptance preceded the
# delegate_task call), the loop ends deterministically instead of running
# another round that would just re-accept. agent/runtime.py prefixes its
# success observations with it; failures never carry it.
TOOL_QUIET = "[[quiet]]"

# occasional role-marker hallucination at the start of an agent round (the
# model literally writes "user\n\n\n" before its answer) — stripped from the
# first bytes of each round so TTS never speaks it
_LEAD_ROLE_JUNK = re.compile(r"^\s*(?:assistant|user|system|tool)\s*\n+")

# mid-answer variant (measured 2026-10-30 14:50:27, session ec822c0a): in
# tool-continuation rounds the model sometimes writes a chat-transcript
# boundary — "好的好的。\nassistant\n好滴，…" — and TTS reads the bare marker
# aloud ("assistant"). _LEAD_ROLE_JUNK only covers the answer start, so
# standalone role lines are stripped anywhere in the stream. During
# streaming only \n-terminated lines match (a partial "assista" tail must
# not be touched); the tail variant catches an unterminated final one.
_ROLE_LINE_JUNK = re.compile(r"(?m)^[ \t]*(?:assistant|user|system|tool)[ \t]*\r?\n")
_ROLE_TAIL_JUNK = re.compile(r"(?m)^[ \t]*(?:assistant|user|system|tool)[ \t]*$")


def strip_role_junk(text):
    """Remove standalone role-marker lines anywhere in spoken text."""
    return _ROLE_TAIL_JUNK.sub("", _ROLE_LINE_JUNK.sub("", text))


class QwenLLM_stream:
    # ASCII . and , must NOT split inside a number: "风速是 5.6 米每秒" used
    # to split at the dot into "风速是 5." | "6 米每秒" — two independent TTS
    # clips ("五." 拖长 + 割裂的 "六米每秒", session 1dfcf9ed 2026-11-05).
    # Lookarounds keep every other dot/comma behavior identical; a decimal
    # dot adjacent to the buffer tail simply waits for the rest of the
    # number (final flush at stream end yields it whole).
    PUNCT = r"(?<!\d)\.(?!\d)|[!?;:。！？；：\n]"
    MORE_PUNCT = r"(?<!\d)\.(?!\d)|(?<!\d),(?!\d)|[、，。！？；：\n!?;:]"
    MIN_LEN_FOR_SEG = 10
    MAX_BUFFER_LEN = 30
    MAX_BUFFER_SESSION = 5

    # keep in sync with frontend/script.js DEFAULT_PERSONA and
    # services/qwen35_fused_server.py SYSTEM_PROMPT. Chinese-first on
    # purpose: a one-line English persona loses to the model's Chinese
    # assistant prior ("我是小爱同学" incident, 2026-11-05 session fb4870f6
    # — asked 你先介绍一下你自己, the model answered as Xiaomi's assistant
    # despite the Elva prompt). The identity block anchors the name with
    # explicit negatives.
    SYSTEM_PROMPT = (
        "你是 Elva，一个温柔、自然的语音对话助手，正在通过语音和用户实时聊天。"
        "请用简短、口语化的自然中文回答，就像真人说话一样；不要输出多余解释，"
        "不要用列表或 Markdown 格式。"
        "用户表示同意、附和或只是简单回应时，顺着上文继续说即可。"
        "说话要像真人：应承、附和、告别这些高频话不要每次都用一模一样的句子，"
        "开口前看一眼自己刚才说过什么，刚用过的句式就换一个说法——"
        "重复的固定句式听起来非常机械。"
        "用户让你停下时，就什么都不要说。"
        "身份必须始终如一：你的名字是 Elva。被问到“你是谁”时，"
        "用一两句口语化的中文介绍自己是 Elva。"
        "你不是小爱同学、不是 Siri、不是小度，也不是任何手机或其他公司自带的"
        "语音助手——永远不要用这些身份自称或介绍自己。"
    )

    def __init__(self, api_url: str = "http://localhost:6007/chat"):
        self.api_url = api_url
        self.agent_api_url = "http://localhost:6007/agent_chat"
        self.sessions = {}

    def get_session(self, client_id: int):
        if client_id not in self.sessions:
            self.sessions[client_id] = []
        return self.sessions[client_id]

    def add_message(self, client_id: int, role: str, content: str):
        session = self.get_session(client_id)
        session.append({"role": role, "content": content})

        merged = []
        for role_key, group in groupby(session, key=lambda x: x["role"]):
            contents = [msg["content"] for msg in group]
            merged.append({"role": role_key, "content": "\n".join(contents)})

        self.sessions[client_id] = merged[-self.MAX_BUFFER_SESSION :]

    def pop_message(self, client_id: int, role: str, content: str) -> bool:
        """Retract one message (by exact role+content) from the TAIL of a
        session. add_message merges consecutive same-role messages with
        "\\n", so a message appended next to `content` may share one entry
        with it — in that case only the `content` portion is stripped.
        Returns True if anything was removed.

        Used by the delivery path to retract a result note whose delivery
        was killed before any audio was audible: while the note sits in
        history, the user's next ordinary reply reads it and speaks the
        result, and the re-enqueued retry then says it AGAIN (measured
        2026-11-05 17:36: full report spoken by the user-turn pipeline,
        then re-delivered in full — the user heard it twice)."""
        session = self.get_session(client_id)
        if not session:
            return False
        msg = session[-1]
        if msg["role"] != role:
            return False
        c = msg["content"]
        if c == content:
            del session[-1]
            return True
        if c.startswith(content + "\n"):
            msg["content"] = c[len(content) + 1:]
            return True
        if c.endswith("\n" + content):
            msg["content"] = c[: len(c) - len(content) - 1]
            return True
        if ("\n" + content + "\n") in c:
            msg["content"] = c.replace("\n" + content + "\n", "\n", 1)
            return True
        return False

    def pop_segment(self, buffer: str, min_len: int = None):
        if min_len is None:
            min_len = self.MIN_LEN_FOR_SEG
        if len(split_cn_en(buffer)) < min_len:
            return None, buffer

        matches = list(re.finditer(self.PUNCT, buffer))
        if matches and len(buffer):
            idx = matches[-1].end()
            seg = buffer[:idx]
            rest = buffer[idx:]
            if len(split_cn_en(seg)) < min_len:
                return None, buffer
            return seg, rest

        # buffer too long, force cut
        if len(split_cn_en(buffer)) >= self.MAX_BUFFER_LEN:
            matches = list(re.finditer(self.MORE_PUNCT, buffer))
            if matches and len(buffer):
                idx = matches[-1].end()
                seg = buffer[:idx]
                rest = buffer[idx:]
                return seg, rest
            return buffer, ""

        return None, buffer

    def generate_with_history(
        self, client_id: int, stop_event=None, system_prompt=None,
        max_history=None, min_seg_len=None, extra_context=None,
    ):
        """Stream the reply for a session.

        system_prompt: per-session persona override (None = class default).
        max_history: how many past turns to include (None = class default).
        min_seg_len: minimum characters before a segment is yielded for TTS
            (None = class default). Smaller = audio starts sooner.
        extra_context: optional text injected as a final user-role note for
            this generation only (agent-runtime V2 表达轮的任务语境；默认
            None 时行为与原来完全一致).
        """
        messages = self.get_session(client_id)
        if max_history:
            messages = messages[-int(max_history):]
        conversation = [
            {"role": "system", "content": system_prompt or self.SYSTEM_PROMPT}
        ] + messages
        if extra_context:
            conversation.append({"role": "user", "content": extra_context})

        response = None
        try:
            response = requests.post(
                self.api_url, json={"messages": conversation}, stream=True, timeout=60
            )

            buffer = ""
            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                if stop_event and stop_event.is_set():
                    break
                if not chunk:
                    continue

                buffer += chunk

                while True:
                    seg, buffer = self.pop_segment(buffer, min_len=min_seg_len)
                    if seg:
                        yield seg
                    else:
                        break

            if buffer.strip() and not (stop_event and stop_event.is_set()):
                yield buffer.strip()

        except Exception as e:
            print(f"Qwen API call failed: {e}")
            yield "Sorry, I cannot answer right now."
        finally:
            # close the socket on EVERY path (normal end, stop, timeout) —
            # an unclosed abandoned response kept the connection alive so
            # the server never saw the disconnect and kept generating
            if response is not None:
                response.close()

    @staticmethod
    def _render_tool_xml(name, args):
        inner = "".join(
            f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in args.items()
        )
        return f"<tool_call>\n<function={name}>\n{inner}</function>\n</tool_call>\n"

    def generate_agent_with_history(
        self, client_id: int, stop_event=None, system_prompt=None,
        max_history=None, min_seg_len=None, extra_context=None,
        tools=None, tool_handler=None, max_rounds=3,
    ):
        """Fast-brain generation with native tool calling (/agent_chat).

        Streams speakable text segments exactly like generate_with_history;
        complete tool calls from the stream are handed to `tool_handler`
        (name, args) -> observation string, and the loop continues with the
        assistant tool-call + tool observation appended as TRANSIENT
        messages. Tool rounds are never written into the persistent session
        history — the caller commits only the final spoken text.

        extra_context: optional text injected as a final user-role note for
        this generation only (e.g. the Agent Runtime's task view).
        """
        import json as _json

        messages = self.get_session(client_id)
        if max_history:
            messages = messages[-int(max_history):]
        base = [{"role": "system", "content": system_prompt or self.SYSTEM_PROMPT}] + messages
        if extra_context:
            base.append({"role": "user", "content": extra_context})

        transient = []
        try:
            for _round in range(max_rounds):
                response = None
                buffer = ""
                full_text = ""
                tool_calls = []
                sanitized = False
                try:
                    response = requests.post(
                        self.agent_api_url,
                        json={"messages": base + transient, "tools": tools,
                              "thinking": False},
                        stream=True, timeout=90,
                    )
                    for chunk in response.iter_content(chunk_size=None):
                        if stop_event and stop_event.is_set():
                            break
                        if not chunk:
                            continue
                        tag, payload = chunk[:1], chunk[1:].decode("utf-8", errors="replace")
                        if tag == b"S":
                            buffer += payload
                            full_text += payload
                            if not sanitized:
                                stripped = _LEAD_ROLE_JUNK.sub("", buffer)
                                if stripped != buffer:
                                    junk_len = len(buffer) - len(stripped)
                                    buffer = stripped
                                    full_text = full_text[junk_len:]
                                    sanitized = True
                                elif len(buffer) > 24:
                                    sanitized = True
                            # standalone role-marker lines mid-answer
                            # (see _ROLE_LINE_JUNK) — before popping, so a
                            # marker line can never ride inside a segment
                            buffer = _ROLE_LINE_JUNK.sub("", buffer)
                            while True:
                                seg, buffer = self.pop_segment(buffer, min_len=min_seg_len)
                                if seg:
                                    yield seg
                                else:
                                    break
                        elif tag == b"T":
                            call = _json.loads(payload)
                            tool_calls.append((call["name"], call.get("arguments") or {}))
                except Exception as e:
                    print(f"Qwen agent API call failed: {e}")
                    yield "Sorry, I cannot answer right now."
                    return
                finally:
                    if response is not None:
                        response.close()

                if buffer.strip() and not (stop_event and stop_event.is_set()):
                    buffer = strip_role_junk(buffer)
                    if buffer.strip():
                        yield buffer.strip()

                if not tool_calls or tool_handler is None:
                    return

                # transient continuation: the model's own words + its tool
                # calls, then one observation per call. strip_role_junk so
                # the model doesn't see (and re-learn) its own marker
                # hallucination in the transcript
                content = strip_role_junk(full_text)
                content = (content + "\n" if content.strip() else "")
                for name, args in tool_calls:
                    content += self._render_tool_xml(name, args)
                transient.append({"role": "assistant", "content": content.rstrip("\n")})
                # Quiet-stop (deterministic, 2026-10-30 double-acceptance
                # fix): every tool succeeded with the [[quiet]] marker AND
                # the model already spoke this generation (e.g. its
                # acceptance preceded the delegate_task call) -> executing
                # the tools is the whole point of this round; another
                # generation round would only re-accept ("好的,我正在查
                # …" 4s after "好的,我马上去查…"). Prompt-level "don't
                # repeat" instructions measured 2/4 failures. Silent tool
                # calls (no speech yet) and failure observations (no
                # marker) still get their continuation round.
                all_quiet = True
                for name, args in tool_calls:
                    try:
                        obs = tool_handler(name, args)
                    except Exception as e:
                        obs = f"tool execution error: {e}"
                        all_quiet = False
                    if obs.startswith(TOOL_QUIET):
                        obs = obs[len(TOOL_QUIET):]
                    else:
                        all_quiet = False
                    transient.append({"role": "tool", "content": str(obs)})
                if all_quiet and full_text.strip():
                    return
        finally:
            pass


class ControlStreamError(Exception):
    """/agent_chat_v2 协议违规（帧超限/非 JSON 行/非 v2 帧）。"""


class ControlResult:
    """一次控制轮调用的完整结果（手册 §12）。

    ok 只有在收到 finish_reason=="stop" 的 done 帧且无 error 帧时才为
    True——异常 EOF、length 截断、缺 done、连接错误都不提交控制批次。
    """

    def __init__(self, text="", reasoning="", tool_calls=None,
                 finish_reason=None, error=None, interrupted=False):
        self.text = text                      # text_delta 拼接
        self.reasoning = reasoning            # reasoning_delta 拼接
        self.tool_calls = tool_calls or []    # [{call_id, name, arguments}]
        self.finish_reason = finish_reason    # None = 没收到 done 帧
        self.error = error
        self.interrupted = interrupted

    @property
    def ok(self):
        return (self.finish_reason == "stop" and not self.error
                and not self.interrupted)

    def __repr__(self):
        return (f"ControlResult(ok={self.ok}, finish={self.finish_reason!r}, "
                f"error={self.error!r}, calls={len(self.tool_calls)}, "
                f"text={len(self.text)}ch)")


class AgentControlClient:
    """/agent_chat_v2 的 NDJSON 客户端（手册 §12）。

    逐字节缓冲、按 \\n 组帧、单帧大小上限；多字节 UTF-8 不会被换行切分
    （\\n 不是多字节序列的一部分），逐行 decode 安全。不假设 HTTP chunk
    边界与应用帧边界有任何对应关系。
    """

    MAX_FRAME_BYTES = 1 << 20                # 单帧 1MB 上限

    # 路由专用实例走 AGENT_CONTROL_ENDPOINT（默认回退主 :6007）
    DEFAULT_ENDPOINT = os.environ.get(
        "AGENT_CONTROL_ENDPOINT", "http://localhost:6007/agent_chat_v2")

    def __init__(self, endpoint=None, timeout_s=30.0):
        self.endpoint = endpoint or self.DEFAULT_ENDPOINT
        self.timeout_s = timeout_s

    def _iter_frames(self, response, stop_event=None):
        """把字节流组成 NDJSON 帧并逐帧 yield。协议违规抛
        ControlStreamError，stop_event 置位则静默停止（缺 done 由调用方
        通过 finish_reason is None 判定）。"""
        buf = b""
        for chunk in response.iter_content(chunk_size=None):
            if stop_event is not None and stop_event.is_set():
                return
            if not chunk:
                continue
            buf += chunk
            while True:
                i = buf.find(b"\n")
                if i < 0:
                    if len(buf) > self.MAX_FRAME_BYTES:
                        raise ControlStreamError(
                            f"frame exceeds {self.MAX_FRAME_BYTES} bytes "
                            "without newline")
                    break
                line, buf = buf[:i], buf[i + 1:]
                if not line.strip():
                    continue
                if len(line) > self.MAX_FRAME_BYTES:
                    raise ControlStreamError("frame exceeds size cap")
                try:
                    f = json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, ValueError) as e:
                    raise ControlStreamError(f"bad NDJSON line: {e}") from None
                if not isinstance(f, dict) or f.get("v") != 2:
                    raise ControlStreamError("frame is not a v2 object")
                yield f

    def run(self, messages, tools=None, thinking=False, stop_event=None,
            temperature=None, max_tokens=None):
        """执行一次控制轮并收集全部帧。任何网络/协议错误都折叠进
        ControlResult.error，不抛出。"""
        body = {"messages": messages, "thinking": thinking}
        if tools:
            body["tools"] = tools
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        response = None
        text_parts, reason_parts, calls = [], [], []
        finish, error = None, None
        try:
            response = requests.post(
                self.endpoint, json=body, stream=True, timeout=self.timeout_s)
            response.raise_for_status()
            for f in self._iter_frames(response, stop_event):
                t = f.get("type")
                if t == "text_delta":
                    text_parts.append(f.get("text") or "")
                elif t == "reasoning_delta":
                    reason_parts.append(f.get("text") or "")
                elif t == "tool_call":
                    calls.append({
                        "call_id": f.get("call_id"),
                        "name": f.get("name"),
                        "arguments": f.get("arguments") or {},
                    })
                elif t == "error":
                    error = f.get("error")
                elif t == "done":
                    finish = f.get("finish_reason")
        except requests.RequestException as e:
            error = f"http: {e}"
        except ControlStreamError as e:
            error = f"protocol: {e}"
        finally:
            # 与 legacy 客户端同一教训：每条路径都关 socket，否则服务端
            # 看不到断连、继续生成占住 GEN_LOCK
            if response is not None:
                response.close()
        return ControlResult(
            text="".join(text_parts), reasoning="".join(reason_parts),
            tool_calls=calls, finish_reason=finish, error=error,
            interrupted=bool(stop_event is not None and stop_event.is_set()))


class QwenLLM_IndexTTS_stream:
    MAX_BUFFER_SESSION = 5
    # keep in sync with frontend/script.js DEFAULT_PERSONA and
    # services/qwen35_fused_server.py SYSTEM_PROMPT. Chinese-first on
    # purpose: a one-line English persona loses to the model's Chinese
    # assistant prior ("我是小爱同学" incident, 2026-11-05 session fb4870f6
    # — asked 你先介绍一下你自己, the model answered as Xiaomi's assistant
    # despite the Elva prompt). The identity block anchors the name with
    # explicit negatives.
    SYSTEM_PROMPT = (
        "你是 Elva，一个温柔、自然的语音对话助手，正在通过语音和用户实时聊天。"
        "请用简短、口语化的自然中文回答，就像真人说话一样；不要输出多余解释，"
        "不要用列表或 Markdown 格式。"
        "用户表示同意、附和或只是简单回应时，顺着上文继续说即可。"
        "说话要像真人：应承、附和、告别这些高频话不要每次都用一模一样的句子，"
        "开口前看一眼自己刚才说过什么，刚用过的句式就换一个说法——"
        "重复的固定句式听起来非常机械。"
        "用户让你停下时，就什么都不要说。"
        "身份必须始终如一：你的名字是 Elva。被问到“你是谁”时，"
        "用一两句口语化的中文介绍自己是 Elva。"
        "你不是小爱同学、不是 Siri、不是小度，也不是任何手机或其他公司自带的"
        "语音助手——永远不要用这些身份自称或介绍自己。"
    )

    def __init__(self, api_url="http://localhost:6007/chat_indextts"):
        self.api_url = api_url
        self.sessions = {}

    def get_session(self, client_id):
        if client_id not in self.sessions:
            self.sessions[client_id] = []
        return self.sessions[client_id]

    def add_message(self, client_id, role, content):
        session = self.get_session(client_id)
        session.append({"role": role, "content": content})

        merged = []
        for role_key, group in groupby(session, key=lambda x: x["role"]):
            contents = [msg["content"] for msg in group]
            merged.append({"role": role_key, "content": "\n".join(contents)})

        self.sessions[client_id] = merged[-self.MAX_BUFFER_SESSION :]

    def generate_with_history(self, client_id, stop_event=None):
        messages = self.get_session(client_id)
        conversation = [{"role": "system", "content": self.SYSTEM_PROMPT}] + messages

        response = None
        try:
            response = requests.post(
                self.api_url, json={"messages": conversation}, stream=True, timeout=60
            )

            seg = ""

            for chunk in response.iter_content(chunk_size=None, decode_unicode=False):
                if stop_event and stop_event.is_set():
                    break

                if not chunk:
                    continue

                tag = chunk[:1]
                payload = chunk[1:]

                if tag == b"B":
                    tmp = seg.strip()
                    seg = ""
                    yield {"text": tmp, "wav": payload}
                else:
                    seg += payload.decode("utf-8")

        except Exception as e:
            print(f"Qwen or IndexTTS API call failed: {e}")
            yield {"text": "Sorry, I cannot answer right now.", "wav": b""}
        finally:
            # close the socket on EVERY path (normal end, stop, timeout) —
            # an unclosed abandoned response kept the connection alive so
            # the server never saw the disconnect and kept generating
            if response is not None:
                response.close()


class QwenLLM_Cosyvoice_stream:
    MAX_BUFFER_SESSION = 5
    # keep in sync with frontend/script.js DEFAULT_PERSONA and
    # services/qwen35_fused_server.py SYSTEM_PROMPT. Chinese-first on
    # purpose: a one-line English persona loses to the model's Chinese
    # assistant prior ("我是小爱同学" incident, 2026-11-05 session fb4870f6
    # — asked 你先介绍一下你自己, the model answered as Xiaomi's assistant
    # despite the Elva prompt). The identity block anchors the name with
    # explicit negatives.
    SYSTEM_PROMPT = (
        "你是 Elva，一个温柔、自然的语音对话助手，正在通过语音和用户实时聊天。"
        "请用简短、口语化的自然中文回答，就像真人说话一样；不要输出多余解释，"
        "不要用列表或 Markdown 格式。"
        "用户表示同意、附和或只是简单回应时，顺着上文继续说即可。"
        "说话要像真人：应承、附和、告别这些高频话不要每次都用一模一样的句子，"
        "开口前看一眼自己刚才说过什么，刚用过的句式就换一个说法——"
        "重复的固定句式听起来非常机械。"
        "用户让你停下时，就什么都不要说。"
        "身份必须始终如一：你的名字是 Elva。被问到“你是谁”时，"
        "用一两句口语化的中文介绍自己是 Elva。"
        "你不是小爱同学、不是 Siri、不是小度，也不是任何手机或其他公司自带的"
        "语音助手——永远不要用这些身份自称或介绍自己。"
    )

    def __init__(self, api_url="http://localhost:6007/chat_cosyvoice"):
        self.api_url = api_url
        self.sessions = {}

    def get_session(self, client_id):
        if client_id not in self.sessions:
            self.sessions[client_id] = []
        return self.sessions[client_id]

    def add_message(self, client_id, role, content):
        session = self.get_session(client_id)
        session.append({"role": role, "content": content})

        merged = []
        for role_key, group in groupby(session, key=lambda x: x["role"]):
            contents = [msg["content"] for msg in group]
            merged.append({"role": role_key, "content": "\n".join(contents)})

        self.sessions[client_id] = merged[-self.MAX_BUFFER_SESSION :]

    def generate_with_history(self, client_id, stop_event=None):
        messages = self.get_session(client_id)
        conversation = [{"role": "system", "content": self.SYSTEM_PROMPT}] + messages

        try:
            response = requests.post(
                self.api_url, json={"messages": conversation}, stream=True, timeout=60
            )

            seg = ""

            for chunk in response.iter_content(chunk_size=None, decode_unicode=False):
                if stop_event and stop_event.is_set():
                    response.close()
                    break

                if not chunk:
                    continue

                tag = chunk[:1]
                payload = chunk[1:]

                if tag == b"B":
                    tmp = seg.strip()
                    seg = ""
                    yield {"text": tmp, "wav": payload}
                else:
                    seg += payload.decode("utf-8")

        except Exception as e:
            print(f"Qwen or Cosyvoice API call failed: {e}")
            yield {"text": "Sorry, I cannot answer right now.", "wav": b""}


# test
if __name__ == "__main__":
    import time

    llm = QwenLLM_stream(api_url="http://localhost:6007/chat")
    text = "Hello, can you introduce yourself in detail?"
    llm.add_message(0, "user", text)
    start_time = time.time()
    for reply in llm.generate_with_history(0):
        end_time = time.time()
        print(f"Response time: {end_time - start_time} seconds | {reply}")
        start_time = time.time()
