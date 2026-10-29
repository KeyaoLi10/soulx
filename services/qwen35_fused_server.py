"""Fused Qwen3.5-9B + FireRedTTS server for SoulX-Duplug dialogue system.

Wire-compatible replacement for dialogue_system/modules/qwen_llm/llm_server.py
(the repo's server loads Qwen2.5-7B-Instruct + IndexTTS/CosyVoice, neither of
which is available on this box: HF is unreachable and the LLM the user wants
is the local Qwen3.5-9B). Endpoints match what clients/llm_client.py expects:

  POST /chat          {"messages": [...]}  -> streaming plain-text chunks
  POST /chat_indextts {"messages": [...]}  -> streaming raw bytes, each chunk
                      is b"S"+utf8 text fragment or b"B"+int16 pcm @24k mono

The LLM is /vepfs-cnbjc4e12c6d7e1f/krisl/models/Qwen3.5-9B (same weights and
loading recipe as full_duplex/frc_services/llm/qwen35_chat_server.py); TTS is
the FireRedTTS OpenAI-compatible service on :8081 (24kHz int16 mono WAV
stream, 44-byte header, edge-trimmed chunks).

Run (venv-frc-llm, transformers 5.x):
  CUDA_VISIBLE_DEVICES=6 /home/research/krisl/full_duplex/venv-frc-llm/bin/python \
      qwen35_fused_server.py [--port 6007]
"""
import argparse
import asyncio
import json
import os
import re
import threading

import requests
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

MODEL_PATH = os.environ.get(
    "QWEN35_MODEL_PATH", "/vepfs-cnbjc4e12c6d7e1f/krisl/models/Qwen3.5-9B"
)
TTS_URL = os.environ.get("FIRERED_TTS_URL", "http://127.0.0.1:8081/v1/audio/speech")
TTS_VOICE = os.environ.get("FIRERED_TTS_VOICE", "f531")

# generation params mirror dialogue_system/modules/qwen_llm/config.yaml
TEMP = 0.8
TOP_P = 0.95
MAX_NEW_TOKENS = 512

# same system prompt as the repo's llm_client.py
SYSTEM_PROMPT = (
    "You are a gentle and natural voice conversation assistant."
    "Your name is Elva."
    "You are communicating with the user via speech; please respond in a natural, brief, and colloquial manner."
    "Do not output extra explanations, do not use lists or markdown format."
    "Keep the conversation coherent, just like talking in reality."
    "If the user shows agreement, affirmation or backchannels, continue speaking naturally based on the previous context."
    "If the user asks you to stop, output nothing."
)

# punctuation that ends a TTS-schedulable segment (same set as the repo's
# iterative_generate in modules/qwen_llm/llm_server.py)
SEG_PUNCT = re.compile(r"[，。！？：；,.!?:;\n]")
# token-like units for the minimum-segment-length check (mirrors split_cn_en)
TOKEN_RE = re.compile(r"[一-鿿]|[A-Za-z]+|[0-9]+")
# minimum CJK chars / latin words before a segment is worth synthesizing
MIN_SEG_CHARS = 3

app = FastAPI()
TOKENIZER = None
MODEL = None
GEN_LOCK = threading.Lock()  # serialize generate(); concurrent generate() on
                             # one model time-slices the GPU anyway


def load_model():
    global TOKENIZER, MODEL
    TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH)
    for attn in ("sdpa", "eager"):
        try:
            MODEL = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                dtype=torch.bfloat16,
                attn_implementation=attn,
            ).to("cuda:0")
            print(f"[llm] loaded with attn_implementation={attn}", flush=True)
            break
        except Exception as e:
            print(f"[llm] {attn} failed ({e})", flush=True)
    if MODEL is None:
        raise RuntimeError("could not load model")
    MODEL.eval()
    print("[llm] model loaded", flush=True)


def _build_prompt(messages, tools=None, thinking=False):
    msgs = [{"role": m["role"], "content": m.get("content") or ""} for m in messages]
    # transformers 5.x rejects a system message anywhere but index 0 — if the
    # caller already supplied one, don't prepend a second
    if not msgs or msgs[0]["role"] != "system":
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + msgs
    return TOKENIZER.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
        tools=tools,
    )


def _token_stream(messages, cancel_event, gen_kwargs):
    """Yield text fragments from the LLM; cancel_event aborts generation."""
    prompt = _build_prompt(messages)
    ids = TOKENIZER(prompt, return_tensors="pt").to(MODEL.device)

    from transformers import StoppingCriteria, StoppingCriteriaList

    class _Cancel(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return cancel_event.is_set()

    streamer = TextIteratorStreamer(
        TOKENIZER, skip_prompt=True, skip_special_tokens=True
    )
    kwargs = dict(
        **ids,
        streamer=streamer,
        stopping_criteria=StoppingCriteriaList([_Cancel()]),
        **gen_kwargs,
    )
    thread = threading.Thread(target=MODEL.generate, kwargs=kwargs)
    with GEN_LOCK:
        thread.start()
        try:
            for text in streamer:
                if cancel_event.is_set():
                    break
                if text:
                    yield text
        finally:
            # make sure generate() exits promptly even if the consumer stopped
            cancel_event.set()
            thread.join()


def _token_stream_prompt(prompt, cancel_event, gen_kwargs):
    """Yield decoded deltas for a prebuilt prompt string (agent path, where
    the prompt depends on tools/thinking and is built by the caller)."""
    ids = TOKENIZER(prompt, return_tensors="pt").to(MODEL.device)

    from transformers import StoppingCriteria, StoppingCriteriaList

    class _Cancel(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return cancel_event.is_set()

    streamer = TextIteratorStreamer(
        TOKENIZER, skip_prompt=True, skip_special_tokens=True
    )
    kwargs = dict(
        **ids,
        streamer=streamer,
        stopping_criteria=StoppingCriteriaList([_Cancel()]),
        **gen_kwargs,
    )
    thread = threading.Thread(target=MODEL.generate, kwargs=kwargs)
    with GEN_LOCK:
        thread.start()
        try:
            for text in streamer:
                if cancel_event.is_set():
                    break
                if text:
                    yield text
        finally:
            # make sure generate() exits promptly even if the consumer stopped
            cancel_event.set()
            thread.join()


def _clean_for_tts(text: str) -> str:
    return text.replace("*", "").replace("`", "").replace("#", "").strip()


def _tts_pcm_chunks(text: str):
    """Yield int16 pcm byte chunks for text from the FireRedTTS stream."""
    text = _clean_for_tts(text)
    if not text:
        return
    try:
        resp = requests.post(
            TTS_URL,
            json={"input": text, "voice": TTS_VOICE, "stream": True},
            stream=True,
            timeout=180,
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"[tts] request failed: {e}", flush=True)
        return
    header_skipped = False
    for chunk in resp.iter_content(chunk_size=None):
        if not chunk:
            continue
        if not header_skipped:
            # first bytes are the 44-byte WAV header the tts server prepends
            chunk = chunk[44:]
            header_skipped = True
            if not chunk:
                continue
        yield chunk


async def _stream_tagged(request: Request, conversation, body, tts=True):
    """Yield ("S", utf8-bytes) text chunks and ("B", pcm-bytes) audio chunks.

    Text chunks stream out as the LLM decodes; each complete
    punctuation-delimited segment is sent to FireRedTTS and its pcm yielded
    as soon as it is synthesized. With tts=False (the /chat text endpoint)
    the TTS step is skipped entirely — the web app synthesizes audio itself,
    so doing the (discarded) synthesis here only added latency and, when the
    TTS service was slow, stalled the text stream behind it.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    done = object()
    cancel = threading.Event()
    gen_kwargs = {
        "temperature": float(body.get("temperature") or TEMP),
        "top_p": float(body.get("top_p") or TOP_P),
        "max_new_tokens": int(body.get("max_tokens") or MAX_NEW_TOKENS),
        "do_sample": True,
    }

    def produce():
        try:
            for text in _token_stream(conversation, cancel, gen_kwargs):
                loop.call_soon_threadsafe(queue.put_nowait, text)
        except Exception as e:
            print(f"[llm] stream error: {e}", flush=True)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, done)

    threading.Thread(target=produce, daemon=True).start()

    disconnected = False
    buffer = ""
    try:
        while True:
            item = await queue.get()
            if item is done:
                break
            if await request.is_disconnected():
                cancel.set()
                disconnected = True
                break
            buffer += item
            yield ("S", item.encode("utf-8"))
            if not tts:
                continue
            # flush every complete punctuation-delimited segment to TTS
            while True:
                m = SEG_PUNCT.search(buffer)
                if not m:
                    break
                seg = buffer[: m.end()]
                buffer = buffer[m.end():]
                if len(TOKEN_RE.findall(seg)) < MIN_SEG_CHARS:
                    continue
                seg = seg.strip()
                if not seg:
                    continue
                # synth off the event loop so other requests keep flowing
                pcm_list = await loop.run_in_executor(
                    None, lambda s=seg: list(_tts_pcm_chunks(s))
                )
                for pcm in pcm_list:
                    if await request.is_disconnected():
                        disconnected = True
                        break
                    yield ("B", pcm)
                if disconnected:
                    break
            if disconnected:
                break
    finally:
        # runs on normal completion AND on GeneratorExit (client gone /
        # response cancelled). Without it an interrupted request kept
        # generating to MAX_NEW_TOKENS server-side while holding GEN_LOCK,
        # delaying every subsequent request behind the zombie.
        cancel.set()
    if tts and not disconnected and buffer.strip():
        pcm_list = await loop.run_in_executor(
            None, lambda: list(_tts_pcm_chunks(buffer))
        )
        for pcm in pcm_list:
            yield ("B", pcm)


@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    messages = data.get("messages", [])
    if not messages:
        return StreamingResponse(iter([]), media_type="text/plain")
    # system prompt is injected in _build_prompt (only if the caller didn't
    # send one — transformers 5.x rejects a second system message)
    conversation = messages

    async def stream():
        async for tag, payload in _stream_tagged(request, conversation, data, tts=False):
            if tag == "S":  # text only — "B" pcm has no place in a text stream
                yield payload.decode("utf-8", errors="replace")

    return StreamingResponse(stream(), media_type="text/plain")


@app.post("/chat_indextts")
async def chat_indextts(request: Request):
    data = await request.json()
    messages = data.get("messages", [])
    if not messages:
        return StreamingResponse(iter([]), media_type="text/plain")
    conversation = messages

    async def byte_stream():
        async for tag, payload in _stream_tagged(request, conversation, data):
            yield tag.encode("ascii") + payload

    return StreamingResponse(byte_stream())


@app.get("/health")
def health():
    return {"status": "ok", "loaded": MODEL is not None}


# ---------------------------------------------------------------------------
# /v1/score — first-token probability scoring (no decode), for the EOT
# detector of the fireredchat frontend mode. Ported from
# full_duplex/frc_services/llm/qwen35_chat_server.py: a yes/no judgment
# needs exactly one forward pass this way (~60ms vs a full generation).
#
# Body: {"messages": [...], "candidates": ["是", "否"]}
# -> {"probabilities": {"是": 0.94, "否": 0.06}}
# ---------------------------------------------------------------------------

@app.post("/v1/score")
async def score(request: Request):
    import math

    body = await request.json()
    messages = body.get("messages", [])
    candidates = body.get("candidates") or ["yes", "no"]
    prompt = _build_prompt(messages)

    cand_ids = []
    for c in candidates:
        toks = TOKENIZER(c, add_special_tokens=False)["input_ids"]
        if not toks:
            return JSONResponse(
                {"error": f"candidate {c!r} has no tokens"}, status_code=400
            )
        cand_ids.append(toks[0])
    if len(set(cand_ids)) != len(cand_ids):
        return JSONResponse(
            {"error": "candidates share a first token"}, status_code=400
        )

    def run():
        ids = TOKENIZER(prompt)["input_ids"]
        input_ids = torch.tensor([ids], dtype=torch.long, device=MODEL.device)
        with torch.no_grad():
            logits = MODEL(input_ids=input_ids).logits[0, -1]
        vals = [logits[i].item() for i in cand_ids]
        m = max(vals)
        exps = [math.exp(v - m) for v in vals]
        z = sum(exps)
        return {c: e / z for c, e in zip(candidates, exps)}

    # serialize against generate() (same model, same GPU) but off the event
    # loop so a slow prefill never blocks the streaming endpoints
    def locked_run():
        with GEN_LOCK:
            return run()

    probs = await asyncio.get_running_loop().run_in_executor(None, locked_run)
    return {"probabilities": probs}


# ---------------------------------------------------------------------------
# /agent_chat — tools + thinking-capable streaming for the fast/slow brains
#
# The fast brain (dialogue app) and the slow brain (background agent loop)
# both need native tool calling with low-latency streaming speech text. This
# endpoint streams the model's decoded output split into three channels so
# the caller never has to regex control intent out of natural language:
#
#   b"S" + utf8   speakable content delta  (goes to segmenting/TTS)
#   b"R" + utf8   reasoning delta          (thinking mode; NEVER spoken)
#   b"T" + utf8   one complete tool call as JSON {"name","arguments"},
#                 emitted only after the closing </tool_call> is seen
#
# Qwen3.5 renders tool calls as <tool_call><function=name><parameter=k>v
# </parameter></function></tool_call>; these tags are NOT special tokens
# (verified against the tokenizer), so they surface in the streamer output
# and can be split server-side. A tag may straddle decode chunks, hence the
# buffering state machine below.
# ---------------------------------------------------------------------------

_TOOL_FN_RE = re.compile(r"<function=\s*([\w.-]+)\s*>(.*?)</function>", re.DOTALL)
_TOOL_PARAM_RE = re.compile(r"<parameter=\s*([\w.-]+)\s*>(.*?)</parameter>", re.DOTALL)


def _parse_tool_call(block: str):
    """Parse the inner text of one <tool_call>...</tool_call> block."""
    m = _TOOL_FN_RE.search(block)
    if not m:
        return None
    call = {"name": m.group(1), "arguments": {}}
    for pm in _TOOL_PARAM_RE.finditer(m.group(2)):
        call["arguments"][pm.group(1)] = pm.group(2).strip()
    return call


class _TagSplitter:
    """Incrementally split decoded output into text / <think> / <tool_call>.

    feed(delta) returns finalized (kind, text) pieces; tags split across
    stream chunks are held back until decidable. kind: 'text' | 'think' |
    'tool' (the full inner block, parsed by the caller).
    """

    OPEN_TAGS = ("<think>", "<tool_call>")

    def __init__(self, start_mode="text"):
        self.buf = ""
        self.mode = start_mode  # text | think | tool
        # with enable_thinking=True the chat template ends the generation
        # prompt with "<think>\n", so the stream begins already inside
        # thinking (the opening tag never appears in the decoded output)

    def _partial_tail(self, tags):
        """How many trailing chars could be the start of one of `tags`."""
        n = len(self.buf)
        for k in range(min(n, max(len(t) for t in tags)), 0, -1):
            tail = self.buf[-k:]
            if any(t.startswith(tail) for t in tags):
                return k
        return 0

    def feed(self, delta):
        self.buf += delta
        out = []
        while True:
            if self.mode == "text":
                found = [(self.buf.find(t), t) for t in self.OPEN_TAGS]
                found = [(i, t) for i, t in found if i != -1]
                if not found:
                    keep = self._partial_tail(self.OPEN_TAGS)
                    cut = len(self.buf) - keep
                    if cut > 0:
                        out.append(("text", self.buf[:cut]))
                        self.buf = self.buf[cut:]
                    return out
                i, t = min(found)
                if i:
                    out.append(("text", self.buf[:i]))
                self.buf = self.buf[i + len(t):]
                self.mode = "think" if t == "<think>" else "tool"
            elif self.mode == "think":
                i = self.buf.find("</think>")
                if i == -1:
                    keep = self._partial_tail(self.OPEN_TAGS + ("</think>",))
                    cut = len(self.buf) - keep
                    if cut > 0:
                        out.append(("think", self.buf[:cut]))
                        self.buf = self.buf[cut:]
                    return out
                if i:
                    out.append(("think", self.buf[:i]))
                self.buf = self.buf[i + len("</think>"):]
                self.mode = "text"
            else:  # inside an open <tool_call>: hold until closed
                i = self.buf.find("</tool_call>")
                if i == -1:
                    return out
                block = self.buf[:i]
                self.buf = self.buf[i + len("</tool_call>"):]
                self.mode = "text"
                out.append(("tool", block))

    def flush(self):
        """End of stream: release whatever never got closed."""
        out = []
        if self.buf:
            out.append((self.mode, self.buf))
            self.buf = ""
        self.mode = "text"
        return out


async def _stream_agent(request: Request, conversation, body, tools, thinking):
    """Yield ('S'|'R'|'T', bytes) chunks for one agent generation."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    done = object()
    cancel = threading.Event()
    gen_kwargs = {
        "temperature": float(body.get("temperature") or TEMP),
        "top_p": float(body.get("top_p") or TOP_P),
        "max_new_tokens": int(body.get("max_tokens") or (4096 if thinking else MAX_NEW_TOKENS)),
        "do_sample": True,
    }
    prompt = _build_prompt(conversation, tools=tools, thinking=thinking)

    def produce():
        try:
            for text in _token_stream_prompt(prompt, cancel, gen_kwargs):
                loop.call_soon_threadsafe(queue.put_nowait, text)
        except Exception as e:
            print(f"[llm] agent stream error: {e}", flush=True)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, done)

    threading.Thread(target=produce, daemon=True).start()

    splitter = _TagSplitter(start_mode="think" if thinking else "text")
    try:
        while True:
            item = await queue.get()
            if item is done:
                break
            if await request.is_disconnected():
                cancel.set()
                break
            pieces = splitter.feed(item)
            for kind, text in pieces:
                if kind == "text" and text:
                    yield ("S", text.encode("utf-8"))
                elif kind == "think" and text:
                    yield ("R", text.encode("utf-8"))
                elif kind == "tool":
                    call = _parse_tool_call(text)
                    if call:
                        yield ("T", json.dumps(call, ensure_ascii=False).encode("utf-8"))
    finally:
        cancel.set()
    for kind, text in splitter.flush():
        if kind == "text" and text:
            yield ("S", text.encode("utf-8"))
        elif kind == "tool":
            call = _parse_tool_call(text)
            if call:
                yield ("T", json.dumps(call, ensure_ascii=False).encode("utf-8"))


@app.post("/agent_chat")
async def agent_chat(request: Request):
    data = await request.json()
    messages = data.get("messages", [])
    tools = data.get("tools") or None
    thinking = bool(data.get("thinking"))
    if not messages:
        return StreamingResponse(iter([]), media_type="text/plain")

    async def byte_stream():
        async for tag, payload in _stream_agent(
            request, messages, data, tools, thinking
        ):
            yield tag.encode("ascii") + payload

    return StreamingResponse(byte_stream())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6007)
    args = parser.parse_args()
    load_model()
    uvicorn.run(app, host=args.host, port=args.port)
