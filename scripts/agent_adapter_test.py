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
from agent import context as agent_context  # noqa: E402
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
             can_deliver=True, runner=None, ui=False, speak_impl=None,
             chitchat_impl=None):
    """router_plans: 弹出式（字符串=计划 JSON；Exception=抛出；callable=
    调用取返回值——用于卡住某次路由制造竞态）。speak_impl 可注入自定义
    上口（如门控回执制造"已提交但线程存活"）。chitchat_impl 可注入
    闲聊路（callable(stop_event) -> 生成器，可抛异常/门控阻塞）；
    缺省 None = 不起闲聊路（旧场景验证无闲聊路的仲裁原语义）。
    ui=True 时挂 on_event（UI 泵 0.05s 轮询，事件进 state["ui_events"]）。"""
    state = {
        "speak_calls": [], "gen_calls": [], "router_calls": 0,
        "routed_texts": [],
        "grace_open": grace_open, "can_deliver": can_deliver,
        "grace_s": 0.0, "last_voice_ts": 0.0, "gen_releases": [],
        "ui_events": [],
    }
    plans = list(router_plans)
    lock = threading.Lock()

    def _pop_plan():
        with lock:
            state["router_calls"] += 1
            item = plans.pop(0) if plans else CREATE_PLAN
        if callable(item):
            item = item()
        return item

    def router_fn(sid, text, snap):
        with lock:
            state["routed_texts"].append(text)
        item = _pop_plan()
        if isinstance(item, Exception):
            raise item
        return item

    def transport(messages, tools, stop_event=None, **kw):
        # shadow 观察走 PlanRouter 的真路径（v2 NDJSON 客户端打桩）
        item = _pop_plan()
        if isinstance(item, Exception):
            return ControlResult(error="http: " + str(item))
        return ControlResult(
            tool_calls=[{"call_id": "C1", "name": "route_turn",
                         "arguments": {"plan_json": item}}],
            finish_reason="stop")

    plan_router = PlanRouter(transport=transport)

    def default_speak(item, stop_event):
        if hasattr(item, "__iter__") and not isinstance(item, (str, bytes)):
            item = "".join(item)          # 表达轮生成器：消费成文本
        state["speak_calls"].append(item)
        rs = speak_results or []
        return rs[len(state["speak_calls"]) - 1] if rs else True

    speak_fn = speak_impl or default_speak
    # 闲聊路仅在新场景显式注入（旧场景验证的是无闲聊路的仲裁原语义）
    chitchat_fn = chitchat_impl or None

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
        chitchat_fn=chitchat_fn,
        generate_fn=lambda stop_event: iter(["你好呀，我是表达轮。"]),
        history_fn=lambda: [{"role": "user", "content": "hi"}],
        release_gen_fn=lambda sn: state["gen_releases"].append(sn),
        on_event=(lambda e, d: state["ui_events"].append((e, d)))
                  if ui else None,
        ui_poll_s=0.05,
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
# （注意别用海象绑定 RuntimeError——会把内置类覆盖成实例，后续所有
# raise RuntimeError 都变成 TypeError）
rt, ad, st = make_env(tmp, [Exception("router timeout")])
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
# T1 的 runner 挂在 cancel_event 上保持 running：默认 runner 会抢先几毫秒把
# T1 跑成 completed，取消回执就变"刚好已经完成了"（竞态 flake，2026-11-07）
rt, ad, st = make_env(tmp, [MIXED_PLAN], runner=lambda s, c, p: (
    c.wait(15), {"type": "canceled"})[1])
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

# ==== 场景 14：get_task 重播——交付被打断的结果再说一遍（2026-11-06
# 待办：outbox 重入语义）====================================================
GET_TASK = plan_json([{"op": "get_task", "target_task_id": "T1"}],
                     reply_mode="receipts")
rt, ad, st = make_env(tmp, [GET_TASK, GET_TASK, GET_TASK])
rt_ref[0] = rt
rt.store.ensure_session("s1")
rt.store.commit_plan("s1", "R0", C.parse_plan(CREATE_PLAN), plan_version=1)
ok = wait(lambda: rt.store.task_row("s1", "T1")["execution_status"]
          == C.EXEC_COMPLETED
          and any("北京晴" == x for x in st["speak_calls"]))
check("14 T1 completed and first delivery played", ok,
      str(st["speak_calls"]))
# 模拟"播出后被用户打断"：交付终态改 interrupted_partial（outbox 已 done）
dlv_row = rt.store._conn.execute(
    "SELECT delivery_id FROM deliveries WHERE session_id='s1' AND "
    "kind='result' AND task_id='T1'").fetchone()
rt.store._conn.execute(
    "UPDATE deliveries SET state='interrupted_partial' WHERE delivery_id=?",
    (dlv_row["delivery_id"],))
rt.store._conn.commit()
ad.handle_utterance("你还没说完呢")
ok = wait(lambda: any("我再说一遍结果" in str(x) for x in st["speak_calls"]))
check("14b interrupted delivery → get_task says redelivering receipt",
      ok, str(st["speak_calls"]))
ok = wait(lambda: st["speak_calls"].count("北京晴") >= 2)
check("14c result re-spoken via re-enqueued outbox", ok,
      str(st["speak_calls"]))
ok = wait(lambda: (rt.store.delivery_row(dlv_row["delivery_id"])
                   or {"state": ""})["state"] == C.DLV_PLAYED_CONFIRMED)
check("14d reopened delivery reaches played_confirmed again", ok)

# 14e：已完整听到（played_confirmed）→ 只报状态，不重播
ad.handle_utterance("那个天气的结果是什么来着")
ok = wait(lambda: sum("已经完成了" in str(x) for x in st["speak_calls"]) >= 1)
check("14e heard-before task → plain done receipt", ok,
      str(st["speak_calls"]))
time.sleep(0.4)
check("14f no re-delivery after plain receipt",
      st["speak_calls"].count("北京晴") == 2, str(st["speak_calls"]))

# 14g：交付在途（outbox pending）→ 不重复重排
rt.store._conn.execute(
    "UPDATE deliveries SET state='interrupted_partial' WHERE delivery_id=?",
    (dlv_row["delivery_id"],))
rt.store._conn.commit()
rt.store.enqueue_outbox("s1", "deliver", {
    "kind": "result", "task_id": "T1", "revision": 1, "run_id": "run-x",
    "outcome": C.EXEC_COMPLETED, "summary": "北京晴",
    "payload": {"summary": "北京晴"}})
n_before = rt.store._conn.execute(
    "SELECT COUNT(*) c FROM outbox WHERE kind='deliver'").fetchone()["c"]
ad.handle_utterance("好了吗")
ok = wait(lambda: any("已经完成了" in str(x) for x in st["speak_calls"]))
n_after = rt.store._conn.execute(
    "SELECT COUNT(*) c FROM outbox WHERE kind='deliver'").fetchone()["c"]
check("14g in-flight delivery not double-enqueued",
      ok and n_before == n_after, f"{n_before} -> {n_after}")

# ==== 场景 15：V2 任务面板 UI 泵（2026-11-06 enforce 实测反馈：
# enforce 下前端看不到慢脑任务）================================================
def _slow_runner(snap, cancel_event, progress_cb):
    time.sleep(0.3)     # 让 queued/running 在 0.05s 轮询下可见
    return {"type": "completed",
            "payload": {"summary": "北京晴", "facts": [], "sources": [],
                        "outcome": "complete"}}


rt, ad, st = make_env(tmp, [CREATE_PLAN], ui=True, runner=_slow_runner)
rt_ref[0] = rt
ad.handle_utterance("帮我查北京天气")
ok = wait(lambda: any(e == "task_update" and d["status"] in ("queued",
                                                             "running")
                      for e, d in st["ui_events"]))
check("15 ui pump emits task_update lifecycle", ok,
      str(st["ui_events"][:6]))
ok = wait(lambda: any(e == "task_update" and d["status"] == "delivered"
                      for e, d in st["ui_events"]))
check("15b delivered status after playback", ok,
      str(st["ui_events"][-3:]))
check("15c goal carried to panel",
      any(d.get("goal") == "查北京天气" for e, d in st["ui_events"]
          if e == "task_update"), str(st["ui_events"][:3]))
check("15d legacy message contract (task_id/goal/status/progress/result)",
      all(set(d) == {"task_id", "goal", "status", "progress", "result"}
          for e, d in st["ui_events"] if e == "task_update"),
      str(st["ui_events"][:1]))

for r in [r for r in [rt] if r]:
    pass

# ==== 场景 16：路由竞态作废——abandoned 真正生效 + 旧路由结果不得覆盖新
# 挂起计划（2026-11-06 湖人双任务事故：碎片1路由在途时碎片2已提交，
# abandoned 原是死代码——置在旧 dict 上、worker 读的是重绑后的新 dict，
# 于是碎片1的旧计划照样入库成了重复任务）====================================
OLD_FRAG_PLAN = plan_json([{"op": "create_task", "local_ref": "L1",
                            "label": "旧碎片", "goal": "旧的"}])
NEW_MERGED_PLAN = plan_json([{"op": "create_task", "local_ref": "L1",
                              "label": "新合并", "goal": "新的"}])
route1_entered = threading.Event()
route1_release = threading.Event()


def _slow_route():
    route1_entered.set()
    route1_release.wait(5.0)      # 碎片1路由卡在途（事故里 6s）
    return OLD_FRAG_PLAN


rt, ad, st = make_env(tmp, [_slow_route, NEW_MERGED_PLAN])
rt_ref[0] = rt
ad.handle_utterance("你再查一下那个湖人队")
ok = wait(lambda: route1_entered.is_set())
check("16 fragment-1 route in flight", ok)
ad.handle_utterance("你昨天的比赛结果怎么样吧？")
ok = wait(lambda: rt.pending_exists("s1"))
check("16b fragment-2 plan pending while old route still running", ok)
check("16c fragments merged for re-route",
      st["routed_texts"][1] == "你再查一下那个湖人队\n你昨天的比赛结果怎么样吧？",
      str(st["routed_texts"]))
route1_release.set()
ok = wait(lambda: any(t["label"] == "新合并" for t in rt.store.task_view("s1")))
check("16d merged plan committed", ok,
      str([t["label"] for t in rt.store.task_view("s1")]))
check("16e stale fragment-1 plan never committed",
      not any(t["label"] == "旧碎片" for t in rt.store.task_view("s1")),
      str([t["label"] for t in rt.store.task_view("s1")]))

# ==== 场景 17：续说合并链——多段碎片累积并入（turn["text"] 持合并文本）====
rt, ad, st = make_env(tmp, [CREATE_PLAN, CREATE_PLAN, CREATE_PLAN])
rt_ref[0] = rt
st["grace_s"] = 5.0                     # 门一直关着：全部停留未提交
st["last_voice_ts"] = time.time()
ad.handle_utterance("你再查一下那个湖人队")
ok = wait(lambda: rt.pending_exists("s1"))
ad.handle_utterance("你昨天的比赛结果")
ok = wait(lambda: st["router_calls"] == 2)
ad.handle_utterance("结果怎么样吧")
ok = wait(lambda: st["router_calls"] == 3)
check("17 fragment chain accumulates into one merged re-route",
      st["routed_texts"][1] == "你再查一下那个湖人队\n你昨天的比赛结果"
      and st["routed_texts"][2]
      == "你再查一下那个湖人队\n你昨天的比赛结果\n结果怎么样吧",
      str(st["routed_texts"]))

# ==== 场景 18：已提交轮不再作废——回执在播时新话轮照常独立路由，不合并、
# 不留空 discard 噪声行 ======================================================
speak_gate = threading.Event()
_gated = {"state": None}


def _gated_speak(item, stop_event):
    s = _gated["state"]
    if isinstance(item, str):
        s["speak_calls"].append(item)
        if "安排好" in item:
            speak_gate.wait(5.0)   # 回执卡住：turn 已 committed、线程存活
    return True


rt, ad, st = make_env(tmp, [CREATE_PLAN, CREATE_PLAN], speak_impl=_gated_speak)
_gated["state"] = st
rt_ref[0] = rt
ad.handle_utterance("帮我查北京天气")
ok = wait(lambda: any("安排好" in str(x) for x in st["speak_calls"]))
check("18 receipt speaking (turn committed, thread alive)", ok,
      str(st["speak_calls"]))
ad.handle_utterance("再查上海天气")
ok = wait(lambda: st["router_calls"] == 2)
check("18b committed turn: new utterance routed alone (no merge)",
      st["routed_texts"][1] == "再查上海天气", str(st["routed_texts"]))
disc = rt.store._conn.execute(
    "SELECT COUNT(*) c FROM route_log WHERE snapshot_json "
    "LIKE '%discarded%'").fetchone()["c"]
check("18c no spurious discard row for committed turn", disc == 0, str(disc))
speak_gate.set()

# ==== 场景 19：改口抑制穿越 outbox——cancel 落在"已完成但结果未播"上，
# 结果不得再播；get_task 显式追问后重播（2026-11-06 湖人纠错事故：
# 抑制只扫 deliveries 表，交付行要到 can_deliver 通过后才创建，结果
# 躺在 outbox 里照播不误）===============================================
CANCEL_PLAN = plan_json([{"op": "cancel_tasks", "targets": [
    {"task_id": "T1", "expected_revision": 1}]}], reply_mode="receipts")
GET_PLAN = plan_json([{"op": "get_task", "target_task_id": "T1"}],
                     reply_mode="receipts")

rt, ad, st = make_env(tmp, [CREATE_PLAN, CANCEL_PLAN, GET_PLAN])
rt_ref[0] = rt
st["can_deliver"] = False            # 结果压在 outbox：交付行尚未创建
ad.handle_utterance("查一下骑士队的比赛")
ok = wait(lambda: rt.store.task_row("s1", "T1") is not None
          and rt.store.task_row("s1", "T1")["execution_status"]
          == C.EXEC_COMPLETED)
check("19 task completed, result held in outbox", ok)
ad.handle_utterance("不对，我要查的是湖人队")
ok = wait(lambda: rt.store.task_row("s1", "T1")["cancel_intent"] == 1)
check("19b cancel on completed marks retraction", ok)
ok = wait(lambda: any("刚好已经完成了" in str(x) for x in st["speak_calls"]))
check("19c cancel receipt speaks already-completed template", ok,
      str(st["speak_calls"]))
st["can_deliver"] = True             # 话权放开：被改口的结果必须保持沉默
time.sleep(0.4)
for _ in range(6):
    rt.delivery.pump()
    time.sleep(0.05)
check("19d retracted result never delivered",
      not any("北京晴" in str(x) for x in st["speak_calls"]),
      str(st["speak_calls"]))
ad.handle_utterance("骑士队结果怎么样")
ok = wait(lambda: any("北京晴" in str(x) for x in st["speak_calls"]))
check("19e explicit get_task re-delivers after retraction", ok,
      str(st["speak_calls"]))
check("19f retraction flag cleared by re-delivery",
      rt.store.task_row("s1", "T1")["cancel_intent"] == 0)

# ==== 场景 20-25：双路并行（2026-11-06 闲聊路设计，方案 A/B/C）===========
# 闲聊路（chitchat_fn）与路由路并行：闲聊路负责常规口播，路由路只在
# 后台做任务机器；仲裁 = 受理/状态类回执静音（QUIET_IF_VOICED_OUTCOMES）
# + get_task 跳结果重播 + 失败/纠错类照播 + 闲聊路故障全套回退。

def _std_chitchat(stop_event):
    return iter(["好，我去查一下。"])


def _committed_count(rt):
    return rt.store._conn.execute(
        "SELECT COUNT(*) AS n FROM requests WHERE state='committed'"
    ).fetchone()["n"]


# 20 任务轮双路：闲聊路应承，受理回执静音，任务照常创建
rt, ad, st = make_env(tmp, [CREATE_PLAN], chitchat_impl=_std_chitchat)
rt_ref[0] = rt
ad.handle_utterance("帮我查北京天气")
ok = wait(lambda: any("好，我去查一下" in str(x) for x in st["speak_calls"]))
check("20 chitchat lane voices acceptance", ok, str(st["speak_calls"]))
ok = wait(lambda: rt.store.task_row("s1", "T1") is not None)
check("20b router lane still creates the task", ok)
time.sleep(0.3)
check("20c acceptance receipt quiet (voiced by chitchat)",
      not any("已经安排好了" in str(x) for x in st["speak_calls"]),
      str(st["speak_calls"]))
check("20d no silent-turn ack piled on top",
      not any(x == "好的。" for x in st["speak_calls"]),
      str(st["speak_calls"]))

# 21 闲聊路失败 → 全套回退（方案 C）：回执照旧 + direct 走表达轮。
# 两轮路由都人为放慢 0.2s：保证失败先于 commit 落账（确定性排序）——
# turn 2 的 direct 路由若是瞬时的，_chitchat_alive 可能在闲聊路线程
# 抛异常前就读到"活着"，表达轮被错误跳过（2026-11-07 暴露的竞态）。
def _boom_chitchat(stop_event):
    raise RuntimeError("chitchat down")


def _slow_create():
    time.sleep(0.2)
    return CREATE_PLAN


def _slow_direct():
    time.sleep(0.2)
    return DIRECT_PLAN


DIRECT_PLAN = plan_json([], reply_mode="direct")
rt, ad, st = make_env(tmp, [_slow_create, _slow_direct],
                      chitchat_impl=_boom_chitchat)
rt_ref[0] = rt
ad.handle_utterance("帮我查北京天气")
ok = wait(lambda: any("已经安排好了" in str(x) for x in st["speak_calls"]))
check("21 chitchat failure falls back to receipt", ok, str(st["speak_calls"]))
ad.handle_utterance("你好呀")
ok = wait(lambda: any("表达轮" in str(x) for x in st["speak_calls"]))
check("21b chitchat failure falls back to expression round", ok,
      str(st["speak_calls"]))

# 22 纠错类回执不静音：闲聊路活着，unknown_task 的拒绝回执照播
UNKNOWN_CANCEL = plan_json([{"op": "cancel_tasks", "targets": [
    {"task_id": "T9", "expected_revision": 1}]}], reply_mode="receipts")
rt, ad, st = make_env(tmp, [UNKNOWN_CANCEL], chitchat_impl=_std_chitchat)
rt_ref[0] = rt
ad.handle_utterance("把那个取消")
ok = wait(lambda: any("我没有找到你说的那个任务" in str(x)
                      for x in st["speak_calls"]))
check("22 rejected receipt still speaks with chitchat alive", ok,
      str(st["speak_calls"]))

# 23 get_task + 闲聊路活着 → 已完成结果不再重排重播（方案 A：只跳本轮，
# 交付行不动）；闲聊路失败后的下一次追问恢复重播
chat_ctl = {"boom": False}


def _ctl_chitchat(stop_event):
    if chat_ctl["boom"]:
        raise RuntimeError("chitchat down")
    return iter(["好，我去查一下。"])


rt, ad, st = make_env(tmp, [CREATE_PLAN, CANCEL_PLAN, GET_PLAN, GET_PLAN],
                      chitchat_impl=_ctl_chitchat)
rt_ref[0] = rt
st["can_deliver"] = False            # 结果压 outbox：交付行尚未创建
ad.handle_utterance("查一下骑士队的比赛")
ok = wait(lambda: rt.store.task_row("s1", "T1") is not None
          and rt.store.task_row("s1", "T1")["execution_status"]
          == C.EXEC_COMPLETED)
check("23 task completed, result held", ok)
ad.handle_utterance("不对，我要查的是湖人队")
ok = wait(lambda: rt.store.task_row("s1", "T1")["cancel_intent"] == 1)
check("23b retraction still lands (router lane unaffected)", ok)
st["can_deliver"] = True
time.sleep(0.4)
for _ in range(6):
    rt.delivery.pump()
    time.sleep(0.05)
check("23c retracted result stays silent",
      not any("北京晴" in str(x) for x in st["speak_calls"]),
      str(st["speak_calls"]))
ad.handle_utterance("骑士队结果怎么样")      # get_task + 闲聊路活着
ok = wait(lambda: _committed_count(rt) >= 3)
check("23d get_task committed with chitchat alive", ok)
time.sleep(0.4)
for _ in range(6):
    rt.delivery.pump()
    time.sleep(0.05)
# 单一出口（2026-11-07 重构）：闲聊路只说桥接短语，不再代播结果——
# get_task 显式追问必须由交付泵重播，闲聊路活着也不跳过
check("23e explicit ask replays in full even with chitchat alive",
      any("北京晴" in str(x) for x in st["speak_calls"]),
      str(st["speak_calls"]))
check("23f replay clears retraction flag",
      rt.store.task_row("s1", "T1")["cancel_intent"] == 0)
row = rt.store._conn.execute(
    "SELECT state FROM deliveries WHERE session_id='s1' AND kind='result' "
    "AND task_id='T1' ORDER BY rowid DESC LIMIT 1").fetchone()
check("23g delivery row played_confirmed after pump replay",
      row is not None and row["state"] == C.DLV_PLAYED_CONFIRMED,
      str(dict(row) if row else None))
chat_ctl["boom"] = True
ad.handle_utterance("再讲一遍骑士队结果")    # 已交付(played_confirmed)后的追问
ok = wait(lambda: sum("北京晴" in str(x) for x in st["speak_calls"]) >= 2)
check("23h re-ask on played_confirmed replays again", ok,
      str(st["speak_calls"]))

# 24 stop-only 计划掐掉闲聊路生成器（纪律是"停一下→不出声"）
STOP_PLAN = plan_json([{"op": "stop_speech",
                        "evidence": {"quote": "停一下"}}],
                      reply_mode="receipts")


def _slow_stop():
    time.sleep(0.25)
    return STOP_PLAN


def _make_gated_chitchat(gate):
    """门控闲聊路：生成器在 gate 放行前阻塞（制造"commit 先于开口"
    的确定性时序）。"""
    def _f(stop_event):
        def _gen():
            gate.wait(timeout=8)
            if stop_event.is_set():
                return
            yield "诶怎么了？"
        return _gen()
    return _f


chat_gate = threading.Event()
rt, ad, st = make_env(tmp, [_slow_stop],
                      chitchat_impl=_make_gated_chitchat(chat_gate))
rt_ref[0] = rt
ad.handle_utterance("停一下")
ok = wait(lambda: rt.store._conn.execute(
    "SELECT COUNT(*) AS n FROM outbox WHERE kind='stop_speech'"
).fetchone()["n"] >= 1)
check("24 stop-only plan committed", ok)
chat_gate.set()                      # commit 之后才放行生成器
time.sleep(0.3)
check("24b stop-only cuts chitchat generator",
      not any("诶怎么了" in str(x) for x in st["speak_calls"]),
      str(st["speak_calls"]))

# 25 续说作废同时掐闲聊路：旧轮生成器死，新轮正常开口
def _slow_old():
    time.sleep(0.3)
    return CREATE_PLAN


gate25 = threading.Event()
rt, ad, st = make_env(tmp, [_slow_old, CREATE_PLAN],
                      chitchat_impl=_make_gated_chitchat(gate25))
rt_ref[0] = rt
ad.handle_utterance("帮我查那个")     # 旧轮：闲聊路门控阻塞
time.sleep(0.05)
ad.handle_utterance("查北京天气")     # 新轮作废旧轮（含闲聊路 stop）
ok = wait(lambda: st["router_calls"] == 2)
check("25 continuation re-routes", ok)
gate25.set()                         # 放行后：旧轮 stop 已置→沉默，新轮开口
time.sleep(0.3)
check("25b abandoned turn's chitchat never speaks (only new turn's)",
      sum("诶怎么了" in str(x) for x in st["speak_calls"]) == 1,
      str(st["speak_calls"]))
check("25c abandoned turn's plan never commits",
      _committed_count(rt) == 1, str(_committed_count(rt)))

# 26 纯应答吸收：路由在飞时一句"嗯，好的好的"不作废挂起计划、不触发
# 重路由、不另起闲聊路（2026-11-06 实测：天气计划被应答作废，连环重
# 路由四次，任务 25s 才创建）；空闲时的应答照常走完整路径
chitchat_starts = {"n": 0}


def _counting_chitchat(stop_event):
    chitchat_starts["n"] += 1
    return iter(["好，我去查一下。"])


def _slow_create26():
    time.sleep(0.3)
    return CREATE_PLAN


rt, ad, st = make_env(tmp, [_slow_create26], chitchat_impl=_counting_chitchat)
rt_ref[0] = rt
ad.handle_utterance("帮我查北京天气")
time.sleep(0.05)
ad.handle_utterance("嗯，好的好的。")     # 路由在飞期间的纯应答 → 吸收
ok = wait(lambda: rt.store.task_row("s1", "T1") is not None)
check("26 ack does not discard pending plan (task still created)", ok)
check("26b ack triggers no re-route",
      st["router_calls"] == 1, str(st["router_calls"]))
check("26c ack starts no second chitchat lane",
      chitchat_starts["n"] == 1, str(chitchat_starts["n"]))
time.sleep(0.4)                          # 上一轮结束后（面板已有 T1）
ad.handle_utterance("嗯，好的好的。")     # 面板有任务 → 吸收（场景 28 验证）
time.sleep(0.3)
check("26d ack with panel tasks also absorbed",
      st["router_calls"] == 1 and chitchat_starts["n"] == 1,
      f"routes={st['router_calls']} chats={chitchat_starts['n']}")
rt2, ad2, st2 = make_env(tmp, [DIRECT_PLAN], chitchat_impl=_std_chitchat)
rt_ref[0] = rt2
ad2.handle_utterance("好的。")             # 空面板+无活跃轮 → 正常路径
ok = wait(lambda: st2["router_calls"] == 1)
check("26e idle ack with empty panel routes normally", ok,
      str(st2["router_calls"]))

# 27 手动停止（duplex_stop）：掐闲聊路、作废挂起计划、停交付泵——
# 停止后一个字都不再出（2026-11-06 实测：停止后闲聊路仍播完整段）
def _slow_create27():
    time.sleep(0.3)
    return CREATE_PLAN


gate27 = threading.Event()
rt, ad, st = make_env(tmp, [_slow_create27],
                      chitchat_impl=_make_gated_chitchat(gate27))
rt_ref[0] = rt
ad.handle_utterance("帮我查北京天气")
time.sleep(0.05)
ad.user_stop()
gate27.set()                             # 停止后放行：生成器应已被掐
time.sleep(0.3)
check("27 user stop cuts chitchat lane",
      not any("诶怎么了" in str(x) for x in st["speak_calls"]),
      str(st["speak_calls"]))
check("27b plan never commits after user stop",
      _committed_count(rt) == 0, str(_committed_count(rt)))
check("27c scheduler pump stopped",
      rt.scheduler._thread is None or not rt.scheduler._thread.is_alive())
check("27d delivery pump stopped",
      rt.delivery._thread is None or not rt.delivery._thread.is_alive())

# 28 面板有任务时的应答吸收（独立于 26 的 prev-alive 场景）：任务完成、
# 轮已结束后，用户应答不再触发闲聊路全盘复述（2026-11-07 实测：每句
# "啊好吧/嗯好的"都复述一遍天气+湖人，循环播报停不下来）
rt, ad, st = make_env(tmp, [CREATE_PLAN], chitchat_impl=_counting_chitchat)
rt_ref[0] = rt
ad.handle_utterance("帮我查北京天气")
ok = wait(lambda: rt.store.task_row("s1", "T1") is not None)
check("28 task created (panel non-empty)", ok)
time.sleep(0.4)                          # 轮结束、无挂起、线程已退
routes_before = st["router_calls"]
chats_before = chitchat_starts["n"]
ad.handle_utterance("啊，好吧。")
ad.handle_utterance("嗯，好的好的。")
time.sleep(0.4)
check("28b acks with panel tasks absorbed (no route, no chitchat)",
      st["router_calls"] == routes_before
      and chitchat_starts["n"] == chats_before,
      f"routes={st['router_calls']}/{routes_before} "
      f"chats={chitchat_starts['n']}/{chats_before}")

# 29 单一出口（2026-11-07 复播事故根修）：闲聊路面板注入不含结果原文
# （render_chitchat_context），结果首播/重播全走交付泵；get_task 对已
# 交付（played_confirmed）的结果也重开重播——"再讲一遍"由泵兑现
ctx = agent_context.render_chitchat_context([
    {"task_id": "T1", "goal": "查北京天气", "execution_status": "completed",
     "progress": "", "result": "晴转多云十五度", "delivered": False},
    {"task_id": "T2", "goal": "查湖人比赛", "execution_status": "completed",
     "progress": "", "result": "对阵开拓者", "delivered": True},
])
check("29 chitchat context hides undelivered result text",
      "晴转多云" not in ctx and "结果待播报" in ctx, ctx[:200])
check("29b chitchat context marks delivered, hides its result text",
      "开拓者" not in ctx and "结果已播报过" in ctx, ctx[:200])

rt, ad, st = make_env(
    tmp, [CREATE_PLAN, GET_PLAN], chitchat_impl=_std_chitchat,
    runner=lambda s, c, p: {"type": "completed",
                            "payload": {"summary": "晴转多云十五度",
                                        "facts": [], "sources": [],
                                        "outcome": "complete"}})
rt_ref[0] = rt
ad.handle_utterance("查一下北京天气")
ok = wait(lambda: any("晴转多云" in str(x) for x in st["speak_calls"]))
check("29c pump speaks result (first outlet)", ok, str(st["speak_calls"]))
ok = wait(lambda: rt.store._conn.execute(
    "SELECT state FROM deliveries WHERE session_id='s1' AND kind='result' "
    "AND task_id='T1' ORDER BY rowid DESC LIMIT 1").fetchone()["state"]
    == C.DLV_PLAYED_CONFIRMED)
check("29d delivery row played_confirmed", ok)
ad.handle_utterance("再讲一遍结果")     # 已交付后的显式追问
ok = wait(lambda: sum("晴转多云" in str(x) for x in st["speak_calls"]) >= 2)
check("29e re-ask on delivered result replays via pump", ok,
      str(st["speak_calls"]))

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
