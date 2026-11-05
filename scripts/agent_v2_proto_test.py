# -*- coding: utf-8 -*-
"""P2a 离线测试：/agent_chat_v2 NDJSON 帧协议 + _TagSplitter G6 修复 +
v2 贪心生成参数（手册 §12）。

不加载模型（模块导入只建 FastAPI app，load_model 在 __main__），
_stream_agent 用假异步生成器替换，纯进程内验证帧协议。

Run: /home/research/krisl/soulx/venv/bin/python -I scripts/agent_v2_proto_test.py
"""
import asyncio
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "qwen35_fused_server", ROOT / "services" / "qwen35_fused_server.py")
srv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(srv)

FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok  {name}")
    else:
        FAILS.append(name)
        print(f"FAIL  {name}  {detail}")


# ==== A. _TagSplitter G6：未闭合块绝不外泄 ==================================

# A1 普通文本在 feed 阶段正常放出（无未决标签尾巴时不必等 flush）
sp = srv._TagSplitter()
pieces = sp.feed("你好")
check("A1 plain text released by feed, flush empty",
      pieces == [("text", "你好")] and sp.flush() == []
      and not sp.truncated)

# A2 未闭合 <tool_call>：丢弃 + truncated（截断的调用绝不能执行/进 TTS）
sp = srv._TagSplitter()
sp.feed("前面的话<tool_call>{\"name\": \"web_sear")
out = sp.flush()
check("A2 unclosed tool dropped, truncated set",
      out == [] and sp.truncated, str(out))

# A3 未闭合 <think>：丢弃（推理草稿），不标 truncated
sp = srv._TagSplitter()
sp.feed("正文<think>想到一半")
out = sp.flush()
check("A3 unclosed think dropped",
      out == [] and not sp.truncated, str(out))

# A4 闭合 tool 块在 feed 阶段正常产出（flush 修复不影响正常路径）
# 工具调用是 Qwen3.5 原生 <function=…><parameter=…> 格式，不是 JSON
sp = srv._TagSplitter()
pieces = sp.feed('好的<tool_call><function=calculate><parameter=expression>'
                 '1+1</parameter></function></tool_call>尾巴')
flushed = sp.flush()
check("A4 closed tool + trailing text parsed by feed",
      ("tool", '<function=calculate><parameter=expression>'
       '1+1</parameter></function>') in pieces
      and ("text", "尾巴") in pieces and flushed == [],
      str((pieces, flushed)))
tool_block = next(t for k, t in pieces if k == "tool")
call = srv._parse_tool_call(tool_block)
check("A5 _parse_tool_call on closed block",
      call and call.get("name") == "calculate"
      and call.get("arguments", {}).get("expression") == "1+1", str(call))

# A6 thinking 起始模式（chat template 已开 <think>）：草稿流出、正文保留
sp = srv._TagSplitter(start_mode="think")
pieces = sp.feed("推理内容</think>答案正文")
check("A6 think start mode unaffected",
      ("think", "推理内容") in pieces and ("text", "答案正文") in pieces,
      str(pieces))

# ==== B. _agent_gen_kwargs：v2 贪心默认，legacy 行为不变 ====================

gk = srv._agent_gen_kwargs({}, thinking=False)
check("B1 legacy default temp", gk["temperature"] == 0.8 and gk["do_sample"])
gk = srv._agent_gen_kwargs({"temperature": 0}, thinking=False)
check("B2 legacy explicit 0 still swallowed (documented quirk, unchanged)",
      gk["temperature"] == 0.8 and gk["do_sample"])

gk = srv._agent_gen_kwargs({}, thinking=False, v2=True)
check("B3 v2 default greedy", gk["temperature"] == 0.0
      and gk["do_sample"] is False, str(gk))
gk = srv._agent_gen_kwargs({"temperature": 0}, thinking=False, v2=True)
check("B4 v2 explicit 0 respected (the or-bug fix)",
      gk["temperature"] == 0.0 and gk["do_sample"] is False, str(gk))
gk = srv._agent_gen_kwargs({"temperature": 0.7}, thinking=False, v2=True)
check("B5 v2 explicit sampling on", gk["temperature"] == 0.7
      and gk["do_sample"] is True)
gk = srv._agent_gen_kwargs({"max_tokens": 300}, thinking=False, v2=True)
check("B6 v2 max_tokens respected", gk["max_new_tokens"] == 300)

# ==== C. /agent_chat_v2 NDJSON 帧协议（假 _stream_agent）=====================


class _FakeRequest:
    def __init__(self, data):
        self._data = data

    async def json(self):
        return self._data

    async def is_disconnected(self):
        return False


async def _collect(response):
    chunks = []
    async for c in response.body_iterator:
        chunks.append(c)
    return "".join(chunks)


def _run_endpoint(data, fake):
    orig = srv._stream_agent
    srv._stream_agent = fake
    try:
        resp = asyncio.run(srv.agent_chat_v2(_FakeRequest(data)))
        text = asyncio.run(_collect(resp))
    finally:
        srv._stream_agent = orig
    return [json.loads(line) for line in text.splitlines() if line.strip()]


async def _fake_ok(request, conversation, body, tools, thinking,
                   v2=False, state=None):
    yield ("R", "思考片段".encode("utf-8"))
    yield ("S", '{"schema_version": 1, '.encode("utf-8"))
    yield ("S", '"operations": []}'.encode("utf-8"))
    yield ("T", json.dumps({"name": "web_search",
                            "arguments": {"query": "天气"}},
                           ensure_ascii=False).encode("utf-8"))
    if state is not None:
        state["truncated"] = False
        state["disconnected"] = False


frames = _run_endpoint({"messages": [{"role": "user", "content": "hi"}],
                        "thinking": False}, _fake_ok)
check("C1 all frames v2 + monotonic seq",
      all(f.get("v") == 2 for f in frames)
      and [f["seq"] for f in frames] == list(range(1, len(frames) + 1)))
kinds = [f["type"] for f in frames]
check("C2 frame order",
      kinds == ["reasoning_delta", "text_delta", "text_delta",
                "tool_call", "done"], str(kinds))
joined = "".join(f["text"] for f in frames if f["type"] == "text_delta")
check("C3 text deltas concatenate to full plan",
      joined == '{"schema_version": 1, "operations": []}', joined)
tc = next(f for f in frames if f["type"] == "tool_call")
check("C4 tool_call frame shape",
      tc["name"] == "web_search" and tc["arguments"] == {"query": "天气"}
      and tc["call_id"] == f"C{tc['seq']}", str(tc))
check("C5 done frame finish_reason stop",
      frames[-1] == {"v": 2, "seq": len(frames), "type": "done",
                     "finish_reason": "stop"}, str(frames[-1]))


async def _fake_truncated(request, conversation, body, tools, thinking,
                          v2=False, state=None):
    yield ("S", '{"schema_ver'.encode("utf-8"))
    if state is not None:
        state["truncated"] = True
        state["disconnected"] = False


frames = _run_endpoint({"messages": [{"role": "user", "content": "hi"}]},
                       _fake_truncated)
check("C6 truncation marked as length (client must not commit)",
      frames[-1]["finish_reason"] == "length", str(frames[-1]))


async def _fake_boom(request, conversation, body, tools, thinking,
                     v2=False, state=None):
    yield ("S", b"x")
    raise RuntimeError("model exploded")


frames = _run_endpoint({"messages": [{"role": "user", "content": "hi"}]},
                       _fake_boom)
check("C7 stream error → error + done(error), no crash",
      frames[-2]["type"] == "error" and "model exploded" in frames[-2]["error"]
      and frames[-1]["finish_reason"] == "error", str(frames[-2:]))

frames = _run_endpoint({"messages": [], "thinking": False}, _fake_ok)
check("C8 empty messages → error + done(error)",
      [f["type"] for f in frames] == ["error", "done"]
      and frames[-1]["finish_reason"] == "error", str(frames))

# C9: 多行文本（NDJSON 换行必须被 json.dumps 转义，帧边界不被破坏）


async def _fake_multiline(request, conversation, body, tools, thinking,
                          v2=False, state=None):
    yield ("S", "第一行\n第二行".encode("utf-8"))
    if state is not None:
        state["truncated"] = False
        state["disconnected"] = False


frames = _run_endpoint({"messages": [{"role": "user", "content": "hi"}]},
                       _fake_multiline)
check("C9 newlines inside text_delta stay JSON-escaped",
      len(frames) == 2 and frames[0]["text"] == "第一行\n第二行"
      and frames[-1]["finish_reason"] == "stop", str(frames))

print()
if FAILS:
    print(f"agent_v2_proto_test: {len(FAILS)} FAILED: {FAILS}")
    sys.exit(1)
print("agent_v2_proto_test: all passed")
