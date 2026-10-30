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
    PUNCT = r"[.!?;:。！？；：\n]"
    MORE_PUNCT = r"[,.!?;:、，。！？；：\n]"
    MIN_LEN_FOR_SEG = 10
    MAX_BUFFER_LEN = 30
    MAX_BUFFER_SESSION = 5

    SYSTEM_PROMPT = (
        "You are a gentle and natural voice conversation assistant."
        "Your name is Elva."
        "You are communicating with the user via speech; please respond in a natural, brief, and colloquial manner."
        "Do not output extra explanations, do not use lists or markdown format."
        "Keep the conversation coherent, just like talking in reality."
        "If the user shows agreement, affirmation or backchannels, continue speaking naturally based on the previous context."
        "If the user asks you to stop, output nothing."
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
        max_history=None, min_seg_len=None,
    ):
        """Stream the reply for a session.

        system_prompt: per-session persona override (None = class default).
        max_history: how many past turns to include (None = class default).
        min_seg_len: minimum characters before a segment is yielded for TTS
            (None = class default). Smaller = audio starts sooner.
        """
        messages = self.get_session(client_id)
        if max_history:
            messages = messages[-int(max_history):]
        conversation = [
            {"role": "system", "content": system_prompt or self.SYSTEM_PROMPT}
        ] + messages

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


class QwenLLM_IndexTTS_stream:
    MAX_BUFFER_SESSION = 5
    SYSTEM_PROMPT = (
        "You are a gentle and natural voice conversation assistant."
        "Your name is Elva."
        "You are communicating with the user via speech; please respond in a natural, brief, and colloquial manner."
        "Do not output extra explanations, do not use lists or markdown format."
        "Keep the conversation coherent, just like talking in reality."
        "If the user shows agreement, affirmation or backchannels, continue speaking naturally based on the previous context."
        "If the user asks you to stop, output nothing."
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
    SYSTEM_PROMPT = (
        "You are a gentle and natural voice conversation assistant."
        "Your name is Elva."
        "You are communicating with the user via speech; please respond in a natural, brief, and colloquial manner."
        "Do not output extra explanations, do not use lists or markdown format."
        "Keep the conversation coherent, just like talking in reality."
        "If the user shows agreement, affirmation or backchannels, continue speaking naturally based on the previous context."
        "If the user asks you to stop, output nothing."
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
