import re
import requests
from itertools import groupby
from omegaconf import OmegaConf
from modules.utils.text_utils import split_cn_en


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
