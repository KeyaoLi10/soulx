# -*- coding: utf-8 -*-
"""P2b 离线测试：v2 NDJSON 控制客户端 + 路由器 + 快照构建（手册 §7/§12）。

客户端测试打真实本地 HTTP 服务（字节级拆包）；路由器测试注入假 transport；
快照测试用内存 SQLite store。无 GPU/无外部服务。

Run: /home/research/krisl/soulx/venv/bin/python -I scripts/agent_router_test.py
"""
import json
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "repo" / "dialogue_system"))

from agent import contracts as C  # noqa: E402
from agent import context as CT  # noqa: E402
from agent.router import (FEW_SHOTS, ROUTER_SYSTEM_PROMPT,  # noqa: E402
                          PlanRouter)
from clients.llm_client import (AgentControlClient,  # noqa: E402
                                ControlResult)
from agent.store import AgentStore  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok  {name}")
    else:
        FAILS.append(name)
        print(f"FAIL  {name}  {detail}")


# ==== 本地 NDJSON 假服务（真实 socket + 字节级拆包）========================


class _NDJSONHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        cfg = self.server.cfg
        body = cfg["body"]
        self.send_response(cfg.get("status", 200))
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        chunk = cfg.get("chunk", 1)
        for i in range(0, len(body), chunk):
            self.wfile.write(body[i:i + chunk])
            self.wfile.flush()
            if cfg.get("chunk_delay"):
                time.sleep(cfg["chunk_delay"])
        if cfg.get("delay_after"):
            time.sleep(cfg["delay_after"])
        if cfg.get("omit_done"):
            return  # 不发 done，直接结束响应


def _serve(cfg):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _NDJSONHandler)
    srv.cfg = cfg
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/agent_chat_v2"


def _frames_body(frames, drop_done=False):
    lines = []
    seq = 0
    for f in frames:
        seq += 1
        f = dict(f)
        f.update({"v": 2, "seq": seq})
        lines.append(json.dumps(f, ensure_ascii=False))
    if drop_done:
        lines = lines[:-1]
    return ("\n".join(lines) + "\n").encode("utf-8")


GOOD_FRAMES = [
    {"type": "reasoning_delta", "text": "先想"},
    {"type": "text_delta", "text": "第一段"},
    {"type": "text_delta", "text": "，第二段"},
    {"type": "tool_call", "call_id": "C4", "name": "route_turn",
     "arguments": {"plan_json": "{\"schema_version\": 1}"}},
    {"type": "done", "finish_reason": "stop"},
]

# ==== A. AgentControlClient ================================================

# A1 逐字节拆包：帧完整重组
srv, url = _serve({"body": _frames_body(GOOD_FRAMES), "chunk": 1})
cli = AgentControlClient(endpoint=url, timeout_s=10)
res = cli.run([{"role": "user", "content": "hi"}])
check("A1 byte-split stream assembles frames",
      res.ok and res.text == "第一段，第二段" and res.reasoning == "先想",
      repr(res))
check("A1b tool_call collected",
      len(res.tool_calls) == 1 and res.tool_calls[0]["call_id"] == "C4"
      and res.tool_calls[0]["arguments"]["plan_json"] == '{"schema_version": 1}',
      repr(res.tool_calls))
srv.shutdown()

# A2 多字节 UTF-8 被网络边界切开（2 字节中文按 1 字节 chunk）
utf8_frames = [
    {"type": "text_delta", "text": "你好世界"},
    {"type": "done", "finish_reason": "stop"},
]
srv, url = _serve({"body": _frames_body(utf8_frames), "chunk": 1})
res = AgentControlClient(endpoint=url, timeout_s=10).run([])
check("A2 multibyte utf-8 across chunk boundary",
      res.ok and res.text == "你好世界", repr(res))
srv.shutdown()

# A3 缺 done：不提交
srv, url = _serve({"body": _frames_body(GOOD_FRAMES, drop_done=True)})
res = AgentControlClient(endpoint=url, timeout_s=10).run([])
check("A3 missing done → not ok, finish None",
      not res.ok and res.finish_reason is None and not res.error, repr(res))
srv.shutdown()

# A4 length 截断：不提交
srv, url = _serve({"body": _frames_body([
    {"type": "text_delta", "text": "半截"},
    {"type": "done", "finish_reason": "length"}])})
res = AgentControlClient(endpoint=url, timeout_s=10).run([])
check("A4 finish length → not ok",
      not res.ok and res.finish_reason == "length", repr(res))
srv.shutdown()

# A5 error 帧 + done(error)
srv, url = _serve({"body": _frames_body([
    {"type": "error", "error": "boom"},
    {"type": "done", "finish_reason": "error"}])})
res = AgentControlClient(endpoint=url, timeout_s=10).run([])
check("A5 error frame surfaced, not ok",
      not res.ok and res.error == "boom", repr(res))
srv.shutdown()

# A6 非 JSON 行 → 协议错误，不抛
srv, url = _serve({"body": b"not-json\n"})
res = AgentControlClient(endpoint=url, timeout_s=10).run([])
check("A6 bad NDJSON line → protocol error folded in",
      not res.ok and res.error and res.error.startswith("protocol"),
      repr(res))
srv.shutdown()

# A7 帧超限（>1MB 无换行）
big = b'{"v":2,"seq":1,"type":"text_delta","text":"' + b"x" * (1 << 20) + b'"'
srv, url = _serve({"body": big, "chunk": 65536})
res = AgentControlClient(endpoint=url, timeout_s=30).run([])
check("A7 frame cap violation → protocol error",
      not res.ok and res.error and "exceeds" in res.error, repr(res))
srv.shutdown()

# A8 连接拒绝 → http 错误，不抛
res = AgentControlClient(endpoint="http://127.0.0.1:1/agent_chat_v2",
                         timeout_s=2).run([])
check("A8 connection refused → http error",
      not res.ok and res.error and res.error.startswith("http"), repr(res))

# A9 stop_event 中断 → interrupted，不 ok
srv, url = _serve({"body": _frames_body(
    [{"type": "text_delta", "text": "慢"}] * 50
    + [{"type": "done", "finish_reason": "stop"}]), "chunk": 1,
    "chunk_delay": 0.02})
stop = threading.Event()


def _killer():
    time.sleep(0.2)
    stop.set()


threading.Thread(target=_killer, daemon=True).start()
res = AgentControlClient(endpoint=url, timeout_s=10).run(
    [], stop_event=stop)
check("A9 stop_event interrupts → not ok, not committed",
      not res.ok and res.interrupted, repr(res))
srv.shutdown()

# ==== B. PlanRouter（注入 transport）=======================================


def _mk_transport(script):
    """script: 每次调用弹出 (ControlResult, extra_messages_out) 项。"""
    calls = []

    def transport(messages, tools, stop_event=None, **kw):
        calls.append([m["role"] for m in messages], )
        calls[-1] = {"n_messages": len(messages), "messages": messages,
                     "tools": tools}
        step = script.pop(0) if script else ControlResult(
            finish_reason="stop")
        if callable(step):
            step = step(calls[-1])
        return step

    return transport, calls


GOOD_PLAN = json.dumps({
    "schema_version": 1,
    "operations": [
        {"op": "create_task", "local_ref": "L1", "label": "查天气",
         "goal": "查北京天气", "evidence": {"quote": "查北京天气"}}],
    "reply_mode": "receipts"}, ensure_ascii=False)


def _tool_call_result(plan_json):
    return ControlResult(
        text="", tool_calls=[{"call_id": "C1", "name": "route_turn",
                              "arguments": {"plan_json": plan_json}}],
        finish_reason="stop")


SNAP = {"snapshot_version": 3, "request_id": "R1", "user_text": "查北京天气",
        "focus_task_id": None, "pending_question": None, "tasks": [],
        "recent_terminal": [], "capabilities": ["web_search"],
        "timezone": "Asia/Shanghai"}

# B1 正常：route_turn 工具调用 → 计划解析成功
tr, calls = _mk_transport([_tool_call_result(GOOD_PLAN)])
rr = PlanRouter(transport=tr).route(SNAP, "查北京天气")
check("B1 tool_call plan routed and parsed",
      rr.ok and rr.plan and rr.plan.operations[0].op == "create_task"
      and rr.plan.operations[0].label == "查天气", rr.to_log())

# B2 消息组装：系统提示在前、few-shot 成对、快照+原文在最后
msg = calls[0]["messages"]
check("B2 system prompt first", msg[0]["role"] == "system"
      and msg[0]["content"] == ROUTER_SYSTEM_PROMPT)
fs_user = [m for m in msg if m["role"] == "user"
           and "查一下北京明天的天气" in m["content"]]
check("B3 few-shots present (multi-task example)",
      bool(fs_user) and len(FEW_SHOTS) >= 24)
last = msg[-1]
check("B4 snapshot + user text in final message",
      last["role"] == "user" and '"task"' not in last["content"][:1]
      and "查北京天气" in last["content"]
      and "snapshot_version" in last["content"], last["content"][:120])
check("B5 route_turn tool advertised",
      calls[0]["tools"][0]["function"]["name"] == "route_turn")

# B6 文本通道兜底（无工具调用，计划以文本给出）
tr, calls = _mk_transport([ControlResult(text=GOOD_PLAN,
                                          finish_reason="stop")])
rr = PlanRouter(transport=tr).route(SNAP, "查北京天气")
check("B6 text-channel plan fallback", rr.ok and rr.attempts == 1,
      rr.to_log())

# B7 格式修复：第一次坏 JSON，第二次好 → attempts=2，修复提示已注入
tr, calls = _mk_transport([
    _tool_call_result('{"schema_version": 1, "operations": ['),
    _tool_call_result(GOOD_PLAN)])
rr = PlanRouter(transport=tr).route(SNAP, "查北京天气")
check("B7 one repair round recovers", rr.ok and rr.attempts == 2, rr.to_log())
repair_msg = calls[1]["messages"][-1]
check("B7b repair note appended",
      repair_msg["role"] == "user" and "plan_json" in repair_msg["content"],
      repair_msg["content"][:80])

# B8 传输故障不修复：直接终态，attempts=1
tr, calls = _mk_transport([ControlResult(error="http: boom")])
rr = PlanRouter(transport=tr).route(SNAP, "查北京天气")
check("B8 transport failure terminal, no repair",
      not rr.ok and rr.attempts == 1 and "http" in rr.error, rr.to_log())

# B9 length 截断可重试一次，重试成功
tr, calls = _mk_transport([
    ControlResult(text="半截", finish_reason="length"),
    _tool_call_result(GOOD_PLAN)])
rr = PlanRouter(transport=tr).route(SNAP, "查北京天气")
check("B9 truncation retried once then ok", rr.ok and rr.attempts == 2,
      rr.to_log())

# B10 两轮都解析失败 → 终态带原因
tr, calls = _mk_transport([_tool_call_result("not json"),
                           _tool_call_result("still not json")])
rr = PlanRouter(transport=tr).route(SNAP, "查北京天气")
check("B10 persistent parse failure terminal",
      not rr.ok and rr.attempts == 2 and "parse" in rr.error, rr.to_log())

# B11 第三轮不再尝试（修复上限 1）
tr, calls = _mk_transport([_tool_call_result("a"), _tool_call_result("b"),
                           _tool_call_result(GOOD_PLAN)])
rr = PlanRouter(transport=tr).route(SNAP, "查北京天气")
check("B11 repair budget is exactly one",
      not rr.ok and rr.attempts == 2, rr.to_log())

# B12 cancel_tasks 缺 target → PlanError → 修复失败终态（G2 在路由层拦截）
bad_cancel = json.dumps({"schema_version": 1, "operations": [
    {"op": "cancel_tasks", "scope": "explicit_targets"}]})
tr, calls = _mk_transport([_tool_call_result(bad_cancel),
                           _tool_call_result(bad_cancel)])
rr = PlanRouter(transport=tr).route(SNAP, "算了别查了")
check("B12 cancel without targets rejected at parse (G2)",
      not rr.ok and "targets" in rr.error, rr.to_log())

# B13 stop_event 置位 → cancelled
stop = threading.Event()
stop.set()
tr, calls = _mk_transport([_tool_call_result(GOOD_PLAN)])
rr = PlanRouter(transport=tr).route(SNAP, "查北京天气", stop_event=stop)
check("B13 stop_event → cancelled", not rr.ok and rr.error == "cancelled",
      rr.to_log())

# B14 会话历史注入（截尾）
tr, calls = _mk_transport([_tool_call_result(GOOD_PLAN)])
hist = [{"role": "user", "content": f"u{i}"} for i in range(10)]
PlanRouter(transport=tr).route(SNAP, "查北京天气", history=hist)
sent_users = [m["content"] for m in calls[0]["messages"]
              if m["role"] == "user"]
check("B14 history trimmed to 6",
      any("u9" in c for c in sent_users)
      and not any("u0" in c and len(c) == 2 for c in sent_users))

# ==== C. context.build_router_snapshot ======================================

tmp = tempfile.mkdtemp(prefix="agent_router_ctx_")
store = AgentStore(path=str(Path(tmp) / "ctx.sqlite3"))
store.ensure_session("s1")
plan = C.parse_plan(json.dumps({
    "schema_version": 1,
    "operations": [
        {"op": "create_task", "local_ref": "L1", "label": "查天气",
         "goal": "查询北京最近一周天气，包括气温、降水和空气质量，"
                 "并给出每日摘要与出行建议等非常详细的信息"},
        {"op": "create_task", "local_ref": "L2", "label": "计算",
         "goal": "算 128*36"}]}))
store.commit_plan("s1", "R1", plan, plan_version=1)

# 把 T1 跑到完成（带结果），T2 保持 queued
run_id = store.create_run("s1", "T1", 1, snapshot={})
token = store.lease_run(run_id)
store.finish_run("s1", "T1", 1, run_id, token, C.EXEC_COMPLETED,
                 result_summary="北京晴",
                 payload={"summary": "北京晴", "facts": [], "sources": []})

# T2 进入 waiting_input 并挂问题
run2 = store.create_run("s1", "T2", 1, snapshot={})
tok2 = store.lease_run(run2)
store.finish_run("s1", "T2", 1, run2, tok2, C.EXEC_WAITING_INPUT,
                 payload={"question": "要查几天的？",
                          "question_id": "Q1", "missing_fields": ["days"]})

snap = CT.build_router_snapshot(store, "s1", "改查五天的",
                                request_id="R2", history=[
                                    {"role": "user", "content": "之前的话"}])
check("C1 active task listed with goal",
      any(t["task_id"] == "T2" and t["execution_status"] == C.EXEC_WAITING_INPUT
          for t in snap["tasks"]), json.dumps(snap["tasks"], ensure_ascii=False))
check("C2 pending question extracted",
      snap["pending_question"] and
      snap["pending_question"]["question"] == "要查几天的？"
      and snap["pending_question"]["task_id"] == "T2",
      str(snap["pending_question"]))
check("C3 completed task in terminal index, goal trimmed",
      any(t["task_id"] == "T1" and t["execution_status"] == C.EXEC_COMPLETED
          and len(t["goal"]) <= CT._TERMINAL_GOAL_CHARS
          for t in snap["recent_terminal"]),
      json.dumps(snap["recent_terminal"], ensure_ascii=False))
check("C4 focus = last active task", snap["focus_task_id"] == "T2",
      snap["focus_task_id"])
check("C5 timezone from session", snap["timezone"] == "Asia/Shanghai")
check("C6 history included", snap.get("recent_history")
      == [{"role": "user", "content": "之前的话"}])
v1 = snap["snapshot_version"]
check("C7 snapshot_version positive", v1 >= 2, v1)
# 再 amend T2 → revision+1 → 版本单调增
plan2 = C.parse_plan(json.dumps({
    "schema_version": 1,
    "operations": [
        {"op": "amend_task", "target_task_id": "T2", "expected_revision": 1,
         "patch": {"query": "五天"}}]}))
store.commit_plan("s1", "R3", plan2, plan_version=1)
snap2 = CT.build_router_snapshot(store, "s1", "嗯")
check("C8 snapshot_version monotonic after amend",
      snap2["snapshot_version"] > v1)

# C9 快照消息渲染
msg = CT.render_snapshot_message(snap, "改查五天的")
check("C9 render message carries snapshot + raw text",
      "snapshot_version" in msg and "改查五天的" in msg
      and "route_turn" in msg)

store.close()

print()
if FAILS:
    print(f"agent_router_test: {len(FAILS)} FAILED: {FAILS}")
    sys.exit(1)
print("agent_router_test: all passed")
