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
from agent import frontend_adapter  # noqa: E402
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

    def router_fn(sid, text, snap, stop_event=None):
        with lock:
            state["routed_texts"].append(text)
        if stop_event is not None:
            state["router_stop_events"] = state.get(
                "router_stop_events", []) + [stop_event]
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
# 待办：outbox 重入语义；2026-11-10 用户新原则：被打断不主动重播）=====
GET_TASK = plan_json([{"op": "get_task", "target_task_id": "T1"}],
                     reply_mode="receipts")
rt, ad, st = make_env(tmp, [GET_TASK, GET_TASK, GET_TASK, GET_TASK,
                            GET_TASK, GET_TASK])
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
# 2026-11-10 用户新原则：被打断的交付不主动重播——播报打断已做得比
# 闲聊难，误打断留不住真想听的用户；泛泛追问只如实告知打断
# （delivery_interrupted 回执），系统不主动重排全文
ad.handle_utterance("你还没说完呢")
ok = wait(lambda: any("播到一半被打断" in str(x) for x in st["speak_calls"]))
check("14b interrupted delivery → status-only receipt (no auto replay)",
      ok, str(st["speak_calls"]))
time.sleep(0.4)
check("14c no re-delivery after interrupted status receipt",
      st["speak_calls"].count("北京晴") == 1, str(st["speak_calls"]))

# 显式重播意图（"再讲一遍"类）才重排——结果播报唯一出口是交付泵
ad.handle_utterance("那你再讲一遍")
ok = wait(lambda: any("我再说一遍结果" in str(x) for x in st["speak_calls"]))
check("14d explicit replay reopens interrupted delivery", ok,
      str(st["speak_calls"]))
ok = wait(lambda: st["speak_calls"].count("北京晴") >= 2)
check("14e result re-spoken via re-enqueued outbox", ok,
      str(st["speak_calls"]))
ok = wait(lambda: (rt.store.delivery_row(dlv_row["delivery_id"])
                   or {"state": ""})["state"] == C.DLV_PLAYED_CONFIRMED)
check("14f reopened delivery reaches played_confirmed again", ok)

# 14g：已完整听到（played_confirmed）→ 状态问句只报状态，不重播
# （2026-11-09 收紧：重播只认"再讲一遍"类显式重播词——上午事故
# "哎，你去查了吗？"把北京天气全文重播了一遍）
ad.handle_utterance("那个天气的结果是什么来着")
ok = wait(lambda: any("刚才也说过啦" in str(x) for x in st["speak_calls"]))
check("14g heard-before task → status-only receipt (no replay)", ok,
      str(st["speak_calls"]))
time.sleep(0.4)
check("14h no re-delivery after status-only receipt",
      st["speak_calls"].count("北京晴") == 2, str(st["speak_calls"]))

# 14i：交付在途（outbox pending）→ 不重复重排
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
check("14i in-flight delivery not double-enqueued",
      ok and n_before == n_after, f"{n_before} -> {n_after}")

# 14j：显式重播词（"再讲一遍"）对已完整听过的结果照旧重排重播——
# 收紧只挡状态问句，不挡显式请求（晚七单一出口语义保留）
ad.handle_utterance("再讲一遍结果")
ok = wait(lambda: any("我再说一遍结果" in str(x) for x in st["speak_calls"]))
check("14j explicit replay word still re-delivers", ok,
      str(st["speak_calls"]))
ok = wait(lambda: st["speak_calls"].count("北京晴") >= 3)
check("14j result re-spoken via re-enqueued outbox", ok,
      str(st["speak_calls"]))

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
# 19b2（2026-11-07 晚死循环事故）：被毒化的任务在面板上必须暴露
# suppressed 标记，闲聊路注入绝不再说"结果待播报"（泵永不播，闲聊路
# 会无限"我给你讲个笑话听听？"吊着用户）
import agent.context as _ctx
row = [r for r in rt.store.task_panel_rows("s1")
       if r["task_id"] == "T1"][0]
check("19b2 panel row exposes suppression",
      row["suppressed"] is True, str(row))
ctx_txt = _ctx.render_chitchat_context(rt.store.task_panel_rows("s1"))
check("19b3 chitchat context stops teasing suppressed delivery",
      "，结果待播报" not in ctx_txt and "，结果已取消播报" in ctx_txt,
      ctx_txt[:200])
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

# ==== 场景 19g：取消落在"结果正在播"的窗口 → 在途交付语音被掐断 ======
# （2026-11-09 e2e 取消竞态）状态面由 19b 覆盖（cancel_intent+suppressed），
# 这里补声音面：泵线程被 speak_fn 占住，队列里的 stop_speech 自己永远
# 轮不到——取消的提交路径必须跨线程把 stop_event 置位。同时覆盖同态
# 幂等：_suppress_deliveries 与泵的开口后复核并发写 suppressed，旧
# 状态机会抛 StoreError 炸在 complete_outbox 之前（deliver 项卡在
# leased 等租约超时自愈）。
rt, ad, st = make_env(tmp, [CREATE_PLAN, CANCEL_PLAN])
rt_ref[0] = rt
cut_log = []

def _blocking_deliver(env, text, stop_event):
    cut_log.append(("start", env.task_id))
    cut = stop_event.wait(5.0)        # 在途播报：只有被掐才提前返回
    cut_log.append(("cut" if cut else "full", env.task_id))
    return not cut                   # True=完整播出

rt.delivery.speak_fn = _blocking_deliver
ad.handle_utterance("查一下骑士队的比赛")
stop_pump = threading.Event()

def _pump_loop():
    while not stop_pump.is_set():
        rt.delivery.pump()
        time.sleep(0.02)

pump_t = threading.Thread(target=_pump_loop, daemon=True)
pump_t.start()
deadline = time.monotonic() + 5
while (not any(e[0] == "start" for e in cut_log)
       and time.monotonic() < deadline):
    rt.scheduler.pump()
    time.sleep(0.02)
check("19g delivery starts speaking",
      any(e[0] == "start" for e in cut_log), str(cut_log))
ad.handle_utterance("算了，不用查了")
ok = wait(lambda: ("cut", "T1") in cut_log)
stop_pump.set()
pump_t.join(timeout=6)
check("19g2 cancel cuts in-flight delivery speech", ok, str(cut_log))
dstate = rt.store.result_delivery_states("s1").get("T1")
check("19g3 cut delivery lands suppressed (same-state idempotent)",
      dstate == C.DLV_SUPPRESSED, str(dstate))
_rows = rt.store._conn.execute(
    "SELECT state FROM outbox WHERE kind='deliver'").fetchall()
check("19g4 deliver outbox item finalized, not stuck leased",
      _rows and all(r["state"] in ("done", "pending") for r in _rows),
      str([dict(r) for r in _rows]))

# ==== 场景 19h：过期租约重领的重复 deliver 项不得重播/不得崩溃 ======
# （2026-11-09 e2e 事故）长结果播报 >30s 租约 → 项被重领归还 pending、
# 泵的 complete_outbox token 静默失效 → 泵重领同一项，record_delivery
# 按身份去重回已 played_confirmed 的交付行再写 RENDERING → StoreError
# 崩溃循环（attempts=58）。修 = deliver 领取租约 300s + 泵见终态处置
# 的重复项直接收尾。这里用"同 payload 再排一条 deliver"精确模拟重领项。
rt, ad, st = make_env(tmp, [CREATE_PLAN])
rt_ref[0] = rt
ad.handle_utterance("查一下骑士队的比赛")
ok = wait(lambda: any("北京晴" in str(x) for x in st["speak_calls"]))
check("19h first delivery spoken once", ok, str(st["speak_calls"]))
_tid = [r for r in rt.store.task_panel_rows("s1")
        if r["task_id"] == "T1"][0]["task_id"]
_dlv = rt.store._conn.execute(
    "SELECT state FROM deliveries WHERE task_id=? ORDER BY rowid "
    "DESC LIMIT 1", (_tid,)).fetchone()
check("19h2 delivery row played_confirmed",
      _dlv and _dlv["state"] == C.DLV_PLAYED_CONFIRMED, str(dict(_dlv or {})))
_src = rt.store._conn.execute(
    "SELECT payload_json FROM outbox WHERE kind='deliver' AND state='done' "
    "ORDER BY outbox_id DESC LIMIT 1").fetchone()["payload_json"]
import json as _json
rt.store._enqueue("s1", "deliver", _json.loads(_src))
for _ in range(4):
    rt.delivery.pump()
    time.sleep(0.05)
check("19h3 duplicate deliver item does not re-speak",
      sum("北京晴" in str(x) for x in st["speak_calls"]) == 1,
      str(st["speak_calls"]))
_dup = rt.store._conn.execute(
    "SELECT state FROM outbox WHERE kind='deliver' ORDER BY outbox_id "
    "DESC LIMIT 1").fetchone()
check("19h4 duplicate deliver item finalized", _dup["state"] == "done",
      str(dict(_dup)))
# deliver 领取租约必须远大于默认 30s（覆盖长结果播报）
rt.store._enqueue("s1", "deliver", _json.loads(_src))
_items = rt.delivery._claim_kind("deliver")
check("19h5 deliver claim uses long lease",
      _items and _items[0]["lease_expires_at"] - time.time() > 200,
      str(_items[:1]))
for _i in _items:
    rt.store.complete_outbox(_i["outbox_id"], _i["token"])

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
                      chitchat_impl=_ctl_chitchat,
                      runner=lambda s, c, p: (
                          p("tool_call", {"tool": "web_search"}) or {
                              "type": "completed",
                              "payload": {"summary": "北京晴", "facts": [],
                                          "sources": [], "outcome": "complete"}}))
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
ad.handle_utterance("啊，好吧。", gen_sn=501)
ad.handle_utterance("嗯，好的好的。", gen_sn=502)
time.sleep(0.4)
check("28b acks with panel tasks absorbed (no route, no chitchat)",
      st["router_calls"] == routes_before
      and chitchat_starts["n"] == chats_before,
      f"routes={st['router_calls']}/{routes_before} "
      f"chats={chitchat_starts['n']}/{chats_before}")
# 28c（2026-11-07 gen 泄漏事故）：吸收轮也必须释放 commit 时置位的 gen
# 标志——泄漏的 gen_in_progress 让交付泵 can_deliver 永假（T2 完成后
# 14.5s 无声），并让 blip 过滤在静止期吞掉"拜拜"×4。
check("28c absorbed ack releases commit-time gen flag",
      501 in st["gen_releases"] and 502 in st["gen_releases"],
      str(st["gen_releases"]))

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

# 29b2-29b3：interrupted 渲染（2026-11-09 复读事故：被打断的播报被当
# "结果待播报"，模型无限承诺"马上到"却永不兑现——晚九修了 suppressed
# 的同型问题，interrupted_partial 漏了）
ctx = agent_context.render_chitchat_context([
    {"task_id": "T3", "goal": "查热火国王比赛",
     "execution_status": "completed", "progress": "",
     "result": "国王赢了", "interrupted": True},
])
check("29b2 interrupted delivery not rendered as pending",
      "结果播报被打断" in ctx, ctx[:200])
check("29b3 interrupted result text hidden (single outlet)",
      "国王赢了" not in ctx, ctx[:200])

# 29c 系列（2026-11-09 渐进式披露重构）：打断只记录状态，绝不附带
# 行为指令（"想听要用户再说一声"是永不关闭的开放环，事故里成了闲聊
# 路每轮主动吊用户的钩子——同一句 NBA 邀请连播两遍逐字相同）；用户
# 实际听到部分照实注入，其余结果文本仍隐藏；执行进度不再注入
ctx = agent_context.render_chitchat_context([
    {"task_id": "T4", "goal": "查NBA赛程",
     "execution_status": "completed", "progress": "",
     "result": "今天有8场比赛，湖人vs老鹰", "interrupted": True,
     "heard": "今天有8场比赛，湖人"},
    {"task_id": "T5", "goal": "查上海天气", "execution_status": "running",
     "progress": "搜索：上海 天气", "result": ""},
])
check("29c interrupted renders heard portion (progressive disclosure)",
      "用户听到了：今天有8场比赛，湖人" in ctx, ctx[:300])
check("29c2 interrupted un-heard remainder still hidden",
      "老鹰" not in ctx, ctx[:300])
check("29c3 no re-ask bait on interrupted rows",
      "再说一声" not in ctx and "想听" not in ctx, ctx[:300])
check("29c4 execution progress not injected",
      "搜索：" not in ctx, ctx[:300])

rt, ad, st = make_env(
    tmp, [CREATE_PLAN, GET_PLAN], chitchat_impl=_std_chitchat,
    runner=lambda s, c, p: (
        p("tool_call", {"tool": "web_search"}) or {
            "type": "completed",
            "payload": {"summary": "晴转多云十五度", "facts": [],
                        "sources": [], "outcome": "complete"}}))
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

# ==== 场景 30-33：长间隔落地 + 路由取消（2026-11-07 复读事故根修）=========
# 事故：路由 10-20s > 用户开口间隔 → 每句被下一句作废，请求永远
# registered、任务建不出来 → 闲聊路对空面板连应三遍"我去查一下"。
# 修 = ①窗口外（独立话轮）新话轮把挂起计划 land 落地（跳 grace +
# create 去重）不合并文本；②窗口内被取代话轮的在途路由断流（释放
# GEN_LOCK）。2026-11-09 折中根修：窗口外不再断流在途路由（断流作废
# 的是还没返回的路由，land 又只救已挂起的计划——两头落空），改由新
# worker 有界等待它返回并落地（场景 34）。

def _slow_create():
    time.sleep(0.3)
    return CREATE_PLAN

# 30 长间隔落地：turn1 路由完成但提交门关着（worker 退场、计划挂起），
# 用户隔很久说新句 → 计划被 land 落地，任务建出来，文本不合并
rt, ad, st = make_env(tmp, [_slow_create, DIRECT_PLAN],
                      chitchat_impl=_std_chitchat)
rt_ref[0] = rt
st["grace_open"] = False               # 提交门关：路由完成也提交不了
ad.handle_utterance("帮我查北京天气")
ok = wait(lambda: rt.pending_age("s1") is not None)
check("30 stale plan parked (grace shut)", ok)
time.sleep(0.15)                       # worker 退场（commit 被门退回）
st["grace_open"] = True
ad._active_turn["started_at"] -= 100   # 用户隔了 100s → 独立话轮
ad.handle_utterance("另外聊点别的")
ok = wait(lambda: rt.store.task_row("s1", "T1") is not None)
check("30 fresh turn lands parked plan (task created)", ok)
check("30b fresh turn does not merge old text",
      "北京天气" not in st["routed_texts"][-1], str(st["routed_texts"]))
ok = wait(lambda: _committed_count(rt) == 2)
check("30c both requests committed (no eternal registered)", ok)

# 31 碎片续说回归：窗口内新句仍走 discard+合并重路由（半句话不能单独路由）
rt, ad, st = make_env(tmp, [_slow_create, DIRECT_PLAN],
                      chitchat_impl=_std_chitchat)
rt_ref[0] = rt
ad.handle_utterance("帮我查那个")
time.sleep(0.05)
ad.handle_utterance("北京天气怎么样")
ok = wait(lambda: st["router_calls"] == 2)
check("31 fragment within window still merges",
      "帮我查那个" in st["routed_texts"][-1], str(st["routed_texts"]))

# 32 land 去重：挂起计划的 create 与现有任务同主题（label/goal 二元
# 字组相交）→ 丢弃该 op，不建双任务连环播报；不同主题照常落地
rt, ad, st = make_env(tmp, [])
rt_ref[0] = rt
rt.router = lambda sid, text, snap: CREATE_PLAN
rt.submit_request("s1", "查北京天气")
rt.commit_pending("s1")
ok = wait(lambda: rt.store.task_row("s1", "T1") is not None)
check("32 setup T1 exists", ok)
rt.submit_request("s1", "再查一遍北京天气")   # 同主题 create 挂起
landed = rt.land_pending("s1")
n_tasks = rt.store._conn.execute(
    "SELECT COUNT(*) AS n FROM tasks WHERE session_id='s1'"
).fetchone()["n"]
check("32 land dedupes same-topic create",
      landed is None and n_tasks == 1, f"landed={landed} n={n_tasks}")
SHANGHAI_PLAN = plan_json([
    {"op": "create_task", "local_ref": "L1", "label": "查天气",
     "goal": "查上海天气"}], reply_mode="receipts")
rt.router = lambda sid, text, snap: SHANGHAI_PLAN
rt.submit_request("s1", "查上海天气")
landed = rt.land_pending("s1")
ok = wait(lambda: rt.store.task_row("s1", "T2") is not None)
check("32b different-topic create lands", landed is not None and ok,
      str(landed is not None))

# 33 路由取消：被取代话轮的在途路由收到 stop_event；取消不是失败，
# 绝不说 ROUTER_FAIL_TEXT
_release33 = threading.Event()

def _hanging_route():
    _release33.wait(5.0)
    return CREATE_PLAN

rt, ad, st = make_env(tmp, [_hanging_route, DIRECT_PLAN],
                      chitchat_impl=_std_chitchat)
rt_ref[0] = rt
ad.handle_utterance("帮我查北京天气")   # 路由挂住在途
time.sleep(0.05)
ad.handle_utterance("算了聊别的")       # 取代：route_stop 应置位
ok = wait(lambda: st.get("router_stop_events")
          and st["router_stop_events"][0].is_set())
check("33 superseded turn's route_stop set", ok)
_release33.set()                       # 放行旧路由（真实链路里已被断流）
time.sleep(0.3)
check("33b cancelled turn never speaks ROUTER_FAIL_TEXT",
      all("没理清" not in str(x) for x in st["speak_calls"]),
      str(st["speak_calls"]))

# 34 独立话轮不杀在途路由（2026-11-09 上午事故：C罗查询路由 10.5s 在途，
# 用户 9.3s 后的补充话轮把它 route_stop 掐死——land 只救已挂起计划，
# 两头落空，请求永远 registered）。新 worker 有界等待在途路由返回并
# 落地：任务建出来、文本不合并、两请求都 committed
_release34 = threading.Event()


def _hanging_route34():
    _release34.wait(5.0)
    return CREATE_PLAN


rt, ad, st = make_env(tmp, [_hanging_route34, DIRECT_PLAN],
                      chitchat_impl=_std_chitchat)
rt_ref[0] = rt
ad.handle_utterance("帮我查北京天气")   # 路由挂住在途
time.sleep(0.05)
ad._active_turn["started_at"] -= 100   # 用户隔了 100s → 独立话轮
ad.handle_utterance("对了再帮我查点别的")
time.sleep(0.3)
check("34 outside-window turn does NOT kill inflight route",
      not (st.get("router_stop_events")
           and st["router_stop_events"][0].is_set()),
      str([e.is_set() for e in st.get("router_stop_events", [])]))
_release34.set()                       # 旧路由返回 → 挂起 → 新 worker 落地
ok = wait(lambda: rt.store.task_row("s1", "T1") is not None)
check("34b inflight route landed (task created)", ok)
ok = wait(lambda: st["router_calls"] == 2)
check("34c new turn's route submitted after landing", ok)
check("34c2 texts not merged",
      "北京天气" not in st["routed_texts"][-1], str(st["routed_texts"]))
ok = wait(lambda: _committed_count(rt) == 2)
check("34d both requests committed (no eternal registered)", ok)

# 34e 有界等待超时放行：路由挂死超过预算（测试里压到 0.3s）→ 新话轮
# 照常提交自己的路由，不无限等；旧路由返回后仍可自行走 grace 提交
_orig_land_wait = frontend_adapter._LAND_WAIT_INFLIGHT_S
frontend_adapter._LAND_WAIT_INFLIGHT_S = 0.3
_release34e = threading.Event()


def _hanging_route34e():
    _release34e.wait(5.0)
    return CREATE_PLAN


rt, ad, st = make_env(tmp, [_hanging_route34e, DIRECT_PLAN],
                      chitchat_impl=_std_chitchat)
rt_ref[0] = rt
ad.handle_utterance("帮我查上海天气")
time.sleep(0.05)
ad._active_turn["started_at"] -= 100
t0 = time.monotonic()
ad.handle_utterance("聊点别的")
ok = wait(lambda: st["router_calls"] == 2, timeout=6.0)
check("34e wait budget exhausted → new route still submitted", ok)
check("34e wait bounded (~0.3s, not 5s)",
      time.monotonic() - t0 < 3.0)
frontend_adapter._LAND_WAIT_INFLIGHT_S = _orig_land_wait
_release34e.set()
ok = wait(lambda: rt.store.task_row("s1", "T1") is not None)
check("34f slow inflight route still lands via its own grace", ok)


# ==== 场景 35：窄版 voiced-by-lane 端到端（2026-11-09 讲笑话事故）==========
# 创建轮闲聊路开口应承 + run 零工具 + 首 run → 结果标 voiced_by_lane：
# 泵不插播（事故：T4 笑话在面板挂"待播报"1 分 54 秒，闲聊路只能调和
# 矛盾 → "刚才那个笑话是不是重复啦"）；话题点名但无重播词 → "already"
# 只报状态；显式重播词 → 从 completed 事件合成 deliver、泵兑现
JOKE_PLAN = plan_json([{"op": "create_task", "local_ref": "L1",
                        "label": "讲笑话", "goal": "讲个笑话",
                        "evidence": {"quote": "讲个笑话"}}],
                      reply_mode="receipts")
rt, ad, st = make_env(
    tmp, [JOKE_PLAN, GET_PLAN, GET_PLAN], chitchat_impl=_std_chitchat,
    runner=lambda s, c, p: {"type": "completed",
                            "payload": {"summary": "从前有座山，山里有座庙。",
                                        "facts": [], "sources": [],
                                        "outcome": "complete"}})
rt_ref[0] = rt
ad.handle_utterance("给我讲个笑话")
ok = wait(lambda: rt.store.task_row("s1", "T1") is not None
          and rt.store.task_row("s1", "T1")["execution_status"]
          == C.EXEC_COMPLETED)
check("35 joke task completes without pump (voiced by lane)", ok)
time.sleep(0.3)
for _ in range(6):
    rt.delivery.pump()
    time.sleep(0.05)
check("35b pump never inserts the joke",
      not any("从前有座山" in str(x) for x in st["speak_calls"]),
      str(st["speak_calls"]))
row = rt.store._conn.execute(
    "SELECT state FROM deliveries WHERE session_id='s1' AND kind='result' "
    "AND task_id='T1' ORDER BY rowid DESC LIMIT 1").fetchone()
check("35c delivery state voiced_by_lane",
      row is not None and row["state"] == C.DLV_VOICED_BY_LANE,
      str(dict(row) if row else None))
rows = rt.store.task_panel_rows("s1")
check("35d panel no longer baits with 待播报",
      rows[0]["delivered"] is True and rows[0]["interrupted"] is False,
      str(rows[0]))
# 话题点名 + 无重播词：结果已由闲聊路说过 → "already"，不重排不重播
ad.handle_utterance("那个笑话讲得怎么样了")
ok = wait(lambda: _committed_count(rt) >= 2)
check("35e topic-named status ask commits", ok)
time.sleep(0.3)
for _ in range(6):
    rt.delivery.pump()
    time.sleep(0.05)
check("35f status ask does not replay voiced-by-lane result",
      not any("从前有座山" in str(x) for x in st["speak_calls"]),
      str(st["speak_calls"]))
# 显式重播词：合成 deliver、重开交付行、泵兑现（单一出口纪律）
ad.handle_utterance("再讲一遍那个笑话")
ok = wait(lambda: _committed_count(rt) >= 3)
check("35g explicit replay commits", ok)
ok = wait(lambda: any("从前有座山" in str(x) for x in st["speak_calls"]))
check("35h explicit replay speaks via pump (synthesized payload)", ok,
      str(st["speak_calls"]))
row = rt.store._conn.execute(
    "SELECT state FROM deliveries WHERE session_id='s1' AND kind='result' "
    "AND task_id='T1' ORDER BY rowid DESC LIMIT 1").fetchone()
check("35i delivery reopened then played_confirmed",
      row is not None and row["state"] == C.DLV_PLAYED_CONFIRMED,
      str(dict(row) if row else None))

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
