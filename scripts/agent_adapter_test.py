# -*- coding: utf-8 -*-
"""P3/P4 离线测试：frontend_adapter（shadow/enforce、grace 提交门、
续说作废、路由失败澄清、表达轮）+ slow_brain V2 runner（§10 事件信封、
needs_input、空结果不包装、依赖注入、取消）。

无 GPU/无网络（web_search 等工具执行器被打桩）。真 AgentRuntimeV2 +
真 store + 假路由/假语音出口。

Run: /home/research/krisl/soulx/venv/bin/python -I scripts/agent_adapter_test.py
"""
import json
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "repo" / "dialogue_system"))

from agent import contracts as C  # noqa: E402
from agent import slow_brain as SB  # noqa: E402
from agent.frontend_adapter import (ROUTER_FAIL_TEXT,  # noqa: E402
                                    SILENT_TURN_ACK_TEXT,
                                    AgentFrontendAdapter)
from agent.router import PlanRouter  # noqa: E402
from agent.runtime import AgentRuntimeV2  # noqa: E402
from agent.tools import ToolResult  # noqa: E402
from clients.llm_client import ControlResult  # noqa: E402

FAILS = []
rt_ref = [None]


def check(name, cond, detail=""):
    if cond:
        print(f"  ok  {name}")
    else:
        FAILS.append(name)
        print(f"FAIL  {name}  {detail}")


def plan_json(ops, **kw):
    d = {"schema_version": 1, "operations": ops}
    d.update(kw)
    return json.dumps(d, ensure_ascii=False)


CREATE_PLAN = plan_json([
    {"op": "create_task", "local_ref": "L1", "label": "查天气",
     "goal": "查北京天气"}], reply_mode="receipts")

DIRECT_PLAN = plan_json([], reply_mode="direct")

MIXED_PLAN = plan_json([
    {"op": "cancel_tasks", "targets": [
        {"task_id": "T1", "expected_revision": 1}]}],
    reply_mode="mixed", direct_question="圆周率是多少？")


def make_env(tmpdir, router_plans, speak_results=None, grace_open=True,
             can_deliver=True, runner=None):
    """router_plans: 弹出式（字符串=计划 JSON；Exception=抛出）。"""
    state = {
        "speak_calls": [], "gen_calls": [], "router_calls": 0,
        "grace_open": grace_open, "can_deliver": can_deliver,
        "grace_s": 0.0, "last_voice_ts": 0.0, "gen_releases": [],
    }
    plans = list(router_plans)
    lock = threading.Lock()

    def router_fn(sid, text, snap):
        with lock:
            state["router_calls"] += 1
            item = plans.pop(0) if plans else CREATE_PLAN
        if isinstance(item, Exception):
            raise item
        return item

    def transport(messages, tools, stop_event=None, **kw):
        # shadow 观察走 PlanRouter 的真路径（v2 NDJSON 客户端打桩）
        with lock:
            state["router_calls"] += 1
            item = plans.pop(0) if plans else CREATE_PLAN
        if isinstance(item, Exception):
            return ControlResult(error="http: " + str(item))
        return ControlResult(
            tool_calls=[{"call_id": "C1", "name": "route_turn",
                         "arguments": {"plan_json": item}}],
            finish_reason="stop")

    plan_router = PlanRouter(transport=transport)

    def speak_fn(item, stop_event):
        if hasattr(item, "__iter__") and not isinstance(item, (str, bytes)):
            item = "".join(item)          # 表达轮生成器：消费成文本
        state["speak_calls"].append(item)
        rs = speak_results or []
        return rs[len(state["speak_calls"]) - 1] if rs else True

    def deliver_speak(env, text, stop_event):
        state["speak_calls"].append(text)
        return True

    db_path = Path(tmpdir) / f"ad{time.time()}{id(state)}.sqlite3"
    state["db_path"] = str(db_path)
    rt = AgentRuntimeV2(
        db_path=str(db_path),
        router_fn=router_fn, runner=runner or (lambda s, c, p: {
            "type": "completed",
            "payload": {"summary": "北京晴", "facts": [], "sources": [],
                        "outcome": "complete"}}),
        speak_fn=deliver_speak,
        can_deliver_fn=lambda sid: state["can_deliver"],
        grace_gate=lambda sid: state["grace_open"])
    rt.delivery.delivery_wait_s = 0.2
    adapter = AgentFrontendAdapter(
        session_id="s1", runtime=rt, router=plan_router,
        snapshot_builder=lambda text, rid, hist: {"user_text": text},
        grace_s_fn=lambda: state["grace_s"],
        last_voice_ts_fn=lambda: state["last_voice_ts"],
        speak_fn=speak_fn,
        can_deliver_fn=lambda: state["can_deliver"],
        generate_fn=lambda stop_event: iter(["你好呀，我是表达轮。"]),
        history_fn=lambda: [{"role": "user", "content": "hi"}],
        release_gen_fn=lambda sn: state["gen_releases"].append(sn),
        mode="enforce")
    rt_ref[0] = rt
    return rt, adapter, state


def wait(cond, timeout=8.0):
    deadline = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > deadline:
            return False
        rt_ref[0].scheduler.pump()
        rt_ref[0].delivery.pump()
        time.sleep(0.02)
    return True


# ==== 场景 1：shadow 观察——只写日志，绝不上口 ==============================
tmp = tempfile.mkdtemp(prefix="agent_adapter_test_")
rt, ad, st = make_env(tmp, [CREATE_PLAN])
ad.mode = "shadow"
ad.observe("查北京天气")
ok = wait(lambda: st["router_calls"] == 1)
check("1 shadow routes in background", ok)
time.sleep(0.2)
check("1b shadow never speaks", st["speak_calls"] == [],
      str(st["speak_calls"]))
rows = rt.store._conn.execute(
    "SELECT mode FROM route_log ORDER BY rowid DESC LIMIT 1").fetchall()
check("1c shadow logs route_log(mode=shadow)",
      rows and rows[0]["mode"] == "shadow", str(rows))

# ==== 场景 2：shadow 等快脑空闲（不与语音生成抢 GEN_LOCK）==================
rt, ad, st = make_env(tmp, [CREATE_PLAN])
ad.mode = "shadow"
st["can_deliver"] = False
t0 = time.time()
ad.observe("查天气")
time.sleep(0.4)
check("2 router waits while fast brain busy", st["router_calls"] == 0)
st["can_deliver"] = True
ok = wait(lambda: st["router_calls"] == 1)
check("2b router fires once idle", ok and time.time() - t0 > 0.3)

# ==== 场景 3：enforce 端到端：提交→回执→执行→交付 ==========================
rt, ad, st = make_env(tmp, [CREATE_PLAN])
rt_ref[0] = rt
st["grace_open"] = True
ad.handle_utterance("帮我查北京天气")
ok = wait(lambda: any("已经安排好了" in str(x) for x in st["speak_calls"]))
check("3 receipt spoken (program template)",
      ok, str(st["speak_calls"]))
ok = wait(lambda: any("北京晴" == x for x in st["speak_calls"]))
check("3b result delivered end-to-end", ok, str(st["speak_calls"]))
check("3c task completed",
      rt.store.task_row("s1", "T1")["execution_status"] == C.EXEC_COMPLETED)
check("3d no expression round for receipts mode",
      not any("表达轮" in str(x) for x in st["speak_calls"]),
      str(st["speak_calls"]))

# ==== 场景 4：grace 门——窗口内挂起，窗口过后提交 ==========================
rt, ad, st = make_env(tmp, [CREATE_PLAN])
rt_ref[0] = rt
st["grace_s"] = 0.4
st["last_voice_ts"] = time.time()      # 用户刚在说话
st["grace_open"] = False
ad.handle_utterance("帮我查北京天气")
time.sleep(0.3)
check("4 plan held while grace window open",
      rt.pending_exists("s1") or st["router_calls"] == 1)
check("4b nothing spoken during hold", st["speak_calls"] == [])
st["last_voice_ts"] = time.time() - 10  # 窗口过了
st["grace_open"] = True
ok = wait(lambda: any("已经安排好了" in str(x) for x in st["speak_calls"]))
check("4c commit after grace passes", ok, str(st["speak_calls"]))

# ==== 场景 5：续说作废——旧计划丢弃，只提交新计划 ==========================
rt, ad, st = make_env(tmp, [
    plan_json([{"op": "create_task", "local_ref": "L1", "label": "旧",
                "goal": "旧的"}]),
    plan_json([{"op": "create_task", "local_ref": "L1", "label": "新",
                "goal": "新的"}])])
rt_ref[0] = rt
st["grace_s"] = 5.0                     # 门一直关着
st["last_voice_ts"] = time.time()
st["grace_open"] = False
ad.handle_utterance("查个东西")
ok = wait(lambda: rt.pending_exists("s1"))
check("5 first plan pending", ok)
ad.handle_utterance("不对，查另一个")
ok = wait(lambda: st["router_calls"] == 2)
check("5b continuation re-routes", ok)
check("5c old plan discarded (only newest pending)",
      not any(t["label"] == "旧" for t in rt.store.task_view("s1")))
st["grace_open"] = True
st["last_voice_ts"] = 0
ok = wait(lambda: any(t["label"] == "新" for t in rt.store.task_view("s1")))
check("5d only new plan committed", ok,
      str([t["label"] for t in rt.store.task_view("s1")]))

# ==== 场景 6：路由失败 → 固定澄清话术 ======================================
rt, ad, st = make_env(tmp, [RuntimeError := Exception("router timeout")])
rt_ref[0] = rt
ad.handle_utterance("嗯……")
ok = wait(lambda: st["speak_calls"] != [])
check("6 router failure speaks fixed clarification",
      ok and st["speak_calls"][0] == ROUTER_FAIL_TEXT, str(st["speak_calls"]))

# ==== 场景 7：direct 模式 → 表达轮 =========================================
rt, ad, st = make_env(tmp, [DIRECT_PLAN])
rt_ref[0] = rt
ad.handle_utterance("你好呀")
ok = wait(lambda: any("表达轮" in str(x) for x in st["speak_calls"]))
check("7 direct mode runs expression round", ok, str(st["speak_calls"]))

# ==== 场景 8：mixed → 回执 + 表达轮 ========================================
rt, ad, st = make_env(tmp, [MIXED_PLAN])
rt_ref[0] = rt
# 先造一个 T1 供取消
pre = C.parse_plan(plan_json([
    {"op": "create_task", "local_ref": "L1", "label": "检索",
     "goal": "检索"}]))
rt.store.ensure_session("s1")
rt.store.commit_plan("s1", "R0", pre, plan_version=1)
ad.handle_utterance("取消检索，顺便问下圆周率是多少？")
ok = wait(lambda: any("已经取消" in str(x) or "正在停止" in str(x)
                      for x in st["speak_calls"]))
check("8 mixed mode speaks control receipt", ok, str(st["speak_calls"]))
ok = wait(lambda: any("表达轮" in str(x) for x in st["speak_calls"]))
check("8b mixed mode does not swallow the question (expression round)",
      ok, str(st["speak_calls"]))

# ==== 场景 10：get_task 状态回执上口（RC_INFO 模板，2026-11-06 事故）=====
rt, ad, st = make_env(tmp, [
    plan_json([{"op": "get_task", "target_task_id": "T1"}],
              reply_mode="receipts")])
rt_ref[0] = rt
# 先造一个 T1（commit 后处于 queued）
pre = C.parse_plan(CREATE_PLAN)
rt.store.ensure_session("s1")
rt.store.commit_plan("s1", "R0", pre, plan_version=1)
ad.handle_utterance("那个天气查得怎么样了", gen_sn=100)
ok = wait(lambda: any("还在进行中" in str(x) or "已经完成了" in str(x)
                      for x in st["speak_calls"]))
check("10 get_task status receipt spoken (RC_INFO template)",
      ok, str(st["speak_calls"]))
check("10b gen flag released after status turn",
      100 in st["gen_releases"], str(st["gen_releases"]))

# 10c：任务跑完后 get_task → "已经完成了"
ok = wait(lambda: rt.store.task_row("s1", "T1")["execution_status"]
          == C.EXEC_COMPLETED)
check("10c T1 completed after scheduler", ok)
rt, ad, st = make_env(tmp, [
    plan_json([{"op": "get_task", "target_task_id": "T1"}],
              reply_mode="receipts")])
rt_ref[0] = rt
rt.store.ensure_session("s1")
pre = C.parse_plan(CREATE_PLAN)
rt.store.commit_plan("s1", "R0", pre, plan_version=1)
# 手动把 T1 推到 completed（不经 runner）
rt.store._conn.execute(
    "UPDATE tasks SET execution_status='completed' "
    "WHERE session_id='s1' AND task_id='T1'")
rt.store._conn.commit()
ad.handle_utterance("好了吗")
ok = wait(lambda: any("已经完成了" in str(x) for x in st["speak_calls"]))
check("10d get_task on completed speaks done template",
      ok, str(st["speak_calls"]))

# ==== 场景 11：clarify 模式——澄清问题必须上口 ==============================
rt, ad, st = make_env(tmp, [
    plan_json([], reply_mode="clarify", clarification="你想查哪个城市？")])
rt_ref[0] = rt
ad.handle_utterance("帮我查天气")
ok = wait(lambda: any("你想查哪个城市" in str(x) for x in st["speak_calls"]))
check("11 clarify mode speaks the clarification",
      ok, str(st["speak_calls"]))

# ==== 场景 12：无声兜底与 stop_speech 豁免 ==================================
# runtime_query（RC_INFO 无模板）+ receipts → 兜底话术，绝不能无声
rt, ad, st = make_env(tmp, [
    plan_json([{"op": "runtime_query", "query": "datetime"}],
              reply_mode="receipts")])
rt_ref[0] = rt
ad.handle_utterance("现在几点了", gen_sn=200)
ok = wait(lambda: st["speak_calls"] != [])
check("12 template-hole receipts turn speaks fallback ack",
      ok and st["speak_calls"][0] == SILENT_TURN_ACK_TEXT,
      str(st["speak_calls"]))
check("12b gen flag released on silent-fallback turn",
      200 in st["gen_releases"], str(st["gen_releases"]))

# stop_speech-only：回执就是安静本身，兜底不得打破
rt, ad, st = make_env(tmp, [
    plan_json([{"op": "stop_speech"}], reply_mode="receipts")])
rt_ref[0] = rt
ad.handle_utterance("别说了", gen_sn=300)
ok = wait(lambda: 300 in st["gen_releases"])
check("12c stop_speech-only turn releases gen flag", ok,
      str(st["gen_releases"]))
check("12d stop_speech-only turn stays silent (no fallback)",
      st["speak_calls"] == [], str(st["speak_calls"]))

# ==== 场景 13：上口轮也释放 gen_sn（_speak 已 bump 序号，释放为 no-op 语义
# 由 app 侧 gen_serial 保证；adapter 侧只验证 finally 一定回调） ============
rt, ad, st = make_env(tmp, [DIRECT_PLAN])
rt_ref[0] = rt
ad.handle_utterance("你好呀", gen_sn=400)
ok = wait(lambda: any("表达轮" in str(x) for x in st["speak_calls"]))
check("13 speaking turn runs expression round", ok, str(st["speak_calls"]))
ok = wait(lambda: 400 in st["gen_releases"])
check("13b speaking turn releases gen flag in finally", ok,
      str(st["gen_releases"]))

for r in [r for r in [rt] if r]:
    pass

# ==== 场景 9：close 释放 ====================================================
rt, ad, st = make_env(tmp, [])
rt.store.ensure_session("s1")
ad.close()
time.sleep(0.2)
check("9 close stops scheduler thread",
      rt.scheduler._thread is None or not rt.scheduler._thread.is_alive())
import sqlite3 as _sq  # noqa: E402
conn = _sq.connect(st["db_path"])
conn.row_factory = _sq.Row
row = conn.execute(
    "SELECT closed, closed_reason FROM sessions WHERE session_id='s1'"
).fetchone()
conn.close()
check("9b session closed=1 persisted before store shutdown",
      row is not None and row["closed"] == 1
      and row["closed_reason"] == "connection_closed", str(dict(row) if row else None))

# ==== B. slow_brain V2 runner（§10）========================================

def fake_generate_factory(script):
    """script: 每轮弹出 (speech, tool_calls)。记录收到的 messages。"""
    seen = []

    def fake(self, messages, tools, cancel_event, deadline):
        seen.append([dict(m) for m in messages])
        step = script.pop(0) if script else ("", [])
        return step

    return fake, seen


# B1 completed：一轮无工具调用直接给答案
orig = SB.SlowBrain._generate
fake, seen = fake_generate_factory([("北京晴，最高 12 度。", [])])
SB.SlowBrain._generate = fake
runner = SB.make_v2_runner()
snap = {"task_id": "T1", "revision": 1, "goal": "查北京天气",
        "constraints": {"city": "北京"}, "user_text": "北京天气怎么样",
        "timezone": "Asia/Shanghai", "depends": [], "budget_timeout_s": 60}
res = runner(snap, threading.Event(), lambda *a: None)
check("B1 completed payload shape",
      res["type"] == "completed"
      and res["payload"]["summary"] == "北京晴，最高 12 度。"
      and res["payload"]["outcome"] == "complete"
      and res["payload"]["sources"] == [], str(res))
user_msg = seen[0][1]["content"]
check("B1b prompt carries goal/constraints/user_text/time",
      "查北京天气" in user_msg and "北京" in user_msg
      and "北京天气怎么样" in user_msg and "当前时间" in user_msg, user_msg[:120])

# B2 依赖结果注入 prompt
fake, seen = fake_generate_factory([("温差 3 度。", [])])
SB.SlowBrain._generate = fake
snap2 = dict(snap, depends=[
    {"task_id": "T1", "revision": 1,
     "result": {"summary": "北京晴 12 度"}},
    {"task_id": "T2", "revision": 1, "result": None}])
res = runner(snap2, threading.Event(), lambda *a: None)
user_msg = seen[0][1]["content"]
check("B2 depends results in prompt",
      "北京晴 12 度" in user_msg and "没有可用结果" in user_msg,
      user_msg[-200:])

# B3 空结果不得包装成 completed
fake, seen = fake_generate_factory([("  ", [])])
SB.SlowBrain._generate = fake
res = runner(snap, threading.Event(), lambda *a: None)
check("B3 empty result → failed(outcome=empty), never completed",
      res["type"] == "failed"
      and res["payload"]["outcome"] == "empty", str(res))

# B4 needs_input：request_input 工具命中即转 waiting_input
fake, seen = fake_generate_factory([(
    "", [("request_input", {"question": "要查哪一天？",
                            "missing_field": "日期"})])])
SB.SlowBrain._generate = fake
res = runner(snap, threading.Event(), lambda *a: None)
check("B4 needs_input payload",
      res["type"] == "needs_input"
      and res["payload"]["question"] == "要查哪一天？"
      and res["payload"]["question_id"] == "T1-r1-q1"
      and res["payload"]["missing_fields"] == ["日期"], str(res))

# B5 工具轮：execute_tool_v2 真调用（打桩 web_search）+ 进度事件 + 二轮总结
from agent import tools as agent_tools  # noqa: E402
progress_events = []
fake, seen = fake_generate_factory([
    ("", [("web_search", {"query": "北京天气"}),
          ("calculate", {"expression": "12*2"})]),
    ("北京明天晴，24 度。", [])])
SB.SlowBrain._generate = fake
orig_search = agent_tools.NEW_TOOL_EXECUTORS["web_search"]
agent_tools.NEW_TOOL_EXECUTORS["web_search"] = lambda args, call_id="": \
    ToolResult(call_id=call_id, status="ok", data="北京：晴，12 度",
               sources=[{"url": "https://example.com/weather",
                         "retrieved_at": "2026-11-06"}])
try:
    res = runner(snap, threading.Event(),
                 lambda etype, payload: progress_events.append(
                     (etype, payload)))
finally:
    agent_tools.NEW_TOOL_EXECUTORS["web_search"] = orig_search
check("B5 completed after tool round",
      res["type"] == "completed"
      and res["payload"]["summary"] == "北京明天晴，24 度。", str(res))
check("B5b sources collected from ToolResult",
      res["payload"]["sources"]
      and res["payload"]["sources"][0]["id"] == "SRC1"
      and res["payload"]["sources"][0]["url"].endswith("/weather"),
      str(res["payload"]["sources"]))
tool_obs = [m["content"] for m in seen[1] if m["role"] == "tool"]
check("B5c calculate really executed (24 in observation)",
      any("24" in t for t in tool_obs), str(tool_obs))
check("B5d progress events from real steps only",
      any(e == "progress" and "搜索" in p.get("desc", "")
          for e, p in progress_events), str(progress_events[:4]))

# B6 工具失败进 limitations，不伪装成功
fake, seen = fake_generate_factory([
    ("", [("web_search", {"query": "x"})]), ("没查到相关信息。", [])])
SB.SlowBrain._generate = fake
orig_search = agent_tools.NEW_TOOL_EXECUTORS["web_search"]
agent_tools.NEW_TOOL_EXECUTORS["web_search"] = lambda args, call_id="": \
    ToolResult(call_id=call_id, status="unavailable", error_code="no_network")
try:
    res = runner(snap, threading.Event(), lambda *a: None)
finally:
    agent_tools.NEW_TOOL_EXECUTORS["web_search"] = orig_search
check("B6 tool failure recorded in limitations",
      res["type"] == "completed"
      and any("web_search" in l for l in res["payload"]["limitations"]),
      str(res["payload"]["limitations"]))
tool_obs = [m["content"] for m in seen[1] if m["role"] == "tool"]
check("B6b failure observation not faked as success",
      any("[unavailable]" in t for t in tool_obs), str(tool_obs))

# B7 取消贯穿
fake, seen = fake_generate_factory([("", [("web_search", {"query": "x"})]),
                                    ("never", [])])
SB.SlowBrain._generate = fake
cancel = threading.Event()
cancel.set()
res = runner(snap, cancel, lambda *a: None)
check("B7 cancel → canceled", res["type"] == "canceled", str(res))

# B8 恢复被替换的方法
SB.SlowBrain._generate = orig

print()
if FAILS:
    print(f"agent_adapter_test: {len(FAILS)} FAILED: {FAILS}")
    sys.exit(1)
print("agent_adapter_test: all passed")
