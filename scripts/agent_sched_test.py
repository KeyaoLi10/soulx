# -*- coding: utf-8 -*-
"""Offline integration tests for scheduler/delivery/runtime V2
(P1d, 手册 §8/§9/§10/§13)。

假慢脑（可注入延迟/失败/取消）+ 假语音出口（三态 True/False/None），
无 GPU/无服务/无外网。覆盖：端到端提交→执行→交付、grace 提交门、
依赖 DAG 与上游失败阻塞、运行中取消、静默掐死重试预算、播出后打断
终态、stop_speech、改口后旧结果抑制、每会话运行上限。

Run: /home/research/krisl/soulx/venv/bin/python -I scripts/agent_sched_test.py
"""
import json
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "repo" / "dialogue_system"))

from agent import contracts as C  # noqa: E402
from agent.runtime import AgentRuntimeV2  # noqa: E402

FAILS = []


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


def pump(rt, cond, timeout=8.0):
    """手动驱动调度+交付循环直到条件成立（测试不启后台线程，确定性）。"""
    deadline = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > deadline:
            return False
        rt.scheduler.pump()
        rt.delivery.pump()
        time.sleep(0.02)
    return True


def make_runtime(tmpdir, runner, speak_results, grace_open=True,
                 can_deliver=True):
    """runner/speak 均为可变状态闭包。speak_results: 弹出式三态序列。"""
    state = {
        "speak_calls": [], "runner_calls": [], "progress": [],
        "grace_open": grace_open, "can_deliver": can_deliver,
        "runner_gate": threading.Event(),
    }
    state["runner_gate"].set()

    def speak_fn(env, text, stop_event):
        state["speak_calls"].append({"text": text, "env": env,
                                     "stop": stop_event})
        r = speak_results.pop(0) if speak_results else True
        if r == "block":
            # 模拟长播放：直到被打断
            stop_event.wait(5.0)
            return False
        return r

    rt = AgentRuntimeV2(
        db_path=str(Path(tmpdir) / f"rt{time.time()}.sqlite3"),
        router_fn=None, runner=runner, speak_fn=speak_fn,
        can_deliver_fn=lambda sid: state["can_deliver"],
        grace_gate=lambda sid: state["grace_open"])
    rt.delivery.delivery_wait_s = 0.3
    return rt, state


def completed_runner(snapshot, cancel_event, progress_cb):
    return {"type": "completed",
            "payload": {"summary": f"{snapshot['goal']}：答案是42",
                        "facts": [], "sources": [], "outcome": "complete"}}



def _all_delivery_ids(rt, sid):
    with rt.store._lock:
        rows = rt.store._conn.execute(
            "SELECT delivery_id FROM deliveries WHERE session_id=?",
            (sid,)).fetchall()
    return [r["delivery_id"] for r in rows]


def _dlv_state(rt, sid):
    with rt.store._lock:
        row = rt.store._conn.execute(
            "SELECT state FROM deliveries WHERE session_id=? "
            "ORDER BY updated_at DESC LIMIT 1", (sid,)).fetchone()
    return row["state"] if row else None


# ==== 场景 1：端到端 提交→grace→执行→交付 ================================
tmp = tempfile.mkdtemp(prefix="agent_sched_test_")

rt, st = make_runtime(tmp, completed_runner, [True])
rt.router = lambda sid, text, snap: plan_json([
    {"op": "create_task", "local_ref": "L1", "label": "查天气",
     "goal": "查北京天气"}])
res = rt.submit_request("s1", "帮我查北京天气", source_commit_id="c1")
check("submit ok", res["ok"])
check("plan held until grace commit", rt.pending_exists("s1"))
out = rt.commit_pending("s1")
check("commit returns receipt texts",
      out is not None and out["texts"] == ["查天气已经安排好了，稍后告诉你结果。"],
      str(out and out["texts"]))
check("task created", rt.store.task_row("s1", "T1") is not None)
ok = pump(rt, lambda: len(st["speak_calls"]) > 0)
check("result delivered end-to-end", ok)
check("spoken text is result summary",
      st["speak_calls"][0]["text"] == "查北京天气：答案是42",
      st["speak_calls"][0]["text"])
check("task completed",
      rt.store.task_row("s1", "T1")["execution_status"] == C.EXEC_COMPLETED)
dlv = [d for d in [rt.store.delivery_row(x) for x in
                   _all_delivery_ids(rt, "s1")] if d]
check("delivery played_confirmed (estimated)",
      any(d["state"] == C.DLV_PLAYED_CONFIRMED for d in dlv))

# 幂等：同一 (request, plan, version) 的网络重试等价重放（store 层）
replay = rt.store.commit_plan("s1", res["request_id"], res["plan"],
                              plan_version=1)
check("replay no new task", replay.replayed
      and len(rt.store.task_view("s1")) == 1)




# ==== 场景 2：grace 门关闭 → 计划挂起；续说 → 作废 ========================
rt2, st2 = make_runtime(tmp, completed_runner, [], grace_open=False)
rt2.router = lambda sid, text, snap: plan_json([
    {"op": "create_task", "local_ref": "L1", "goal": "g"}])
rt2.submit_request("s2", "帮我查", source_commit_id="c1")
out = rt2.commit_pending("s2")
check("grace closed holds plan", out is None and rt2.pending_exists("s2"))
rt2.discard_pending("s2", reason="continuation")
check("discard clears plan", not rt2.pending_exists("s2"))
check("no task created from discarded plan",
      rt2.store.task_view("s2") == [])
# 门开后正常提交
st2["grace_open"] = True
rt2.submit_request("s2", "帮我查上海", source_commit_id="c2")
out = rt2.commit_pending("s2")
check("grace open commits", out is not None)

# ==== 场景 3：依赖 DAG ======================================================
dep_runner_calls = []


def dep_runner(snapshot, cancel_event, progress_cb):
    dep_runner_calls.append(snapshot)
    if snapshot["task_id"] == "T1":
        return {"type": "completed",
                "payload": {"summary": "北京明天晴，12度",
                            "facts": [], "sources": [], "outcome": "complete"}}
    if snapshot["task_id"] == "T2":
        return {"type": "completed",
                "payload": {"summary": "上海明天雨，15度",
                            "facts": [], "sources": [], "outcome": "complete"}}
    return {"type": "completed",
            "payload": {"summary": "两地温差3度，北京更冷",
                        "facts": [], "sources": [], "outcome": "complete"}}


rt3, st3 = make_runtime(tmp, dep_runner, [True, True, True])
rt3.router = lambda sid, text, snap: plan_json([
    {"op": "create_task", "local_ref": "L1", "label": "北京",
     "goal": "查北京天气"},
    {"op": "create_task", "local_ref": "L2", "label": "上海",
     "goal": "查上海天气"},
    {"op": "create_task", "local_ref": "L3", "label": "比较",
     "goal": "比较两地天气", "depends_on_local_refs": ["L1", "L2"]}])
rt3.submit_request("s3", "分别查北京上海天气然后比较", source_commit_id="c1")
rt3.commit_pending("s3")
ok = pump(rt3, lambda: len(dep_runner_calls) >= 3)
check("deps run after parents", ok and
      [s["task_id"] for s in dep_runner_calls][:2] in (
          ["T1", "T2"], ["T2", "T1"]))
snap3 = [s for s in dep_runner_calls if s["task_id"] == "T3"][0]
check("dependent snapshot carries dep results",
      len(snap3["depends"]) == 2 and
      any(d["result"] and "北京" in d["result"].get("summary", "")
          for d in snap3["depends"]),
      json.dumps(snap3["depends"], ensure_ascii=False)[:200])

# ==== 场景 4：上游失败 → 下游 blocked ======================================
def fail_runner(snapshot, cancel_event, progress_cb):
    if snapshot["task_id"] == "T1":
        return {"type": "failed",
                "payload": {"summary": "搜索服务不可用",
                            "limitations": ["搜索超时"], "outcome": "failed"}}
    return {"type": "completed",
            "payload": {"summary": "x", "facts": [], "sources": [],
                        "outcome": "complete"}}


rt4, st4 = make_runtime(tmp, fail_runner, [True])  # 只播失败话术
rt4.router = lambda sid, text, snap: plan_json([
    {"op": "create_task", "local_ref": "L1", "goal": "查A"},
    {"op": "create_task", "local_ref": "L2", "goal": "比较",
     "depends_on_local_refs": ["L1"]}])
rt4.submit_request("s4", "查A然后比较", source_commit_id="c1")
rt4.commit_pending("s4")
ok = pump(rt4, lambda: rt4.store.task_row("s4", "T2")[
    "execution_status"] == C.EXEC_BLOCKED_DEPENDENCY)
check("downstream blocked on upstream failure", ok)
check("failed task speaks controlled text",
      st4["speak_calls"] and "没有完成" in st4["speak_calls"][0]["text"],
      str(st4["speak_calls"][:1]))

# ==== 场景 5：运行中取消（cancel_event 贯穿 worker）========================
cancel_state = {"runner_started": threading.Event(),
                "saw_cancel": False}


def slow_runner(snapshot, cancel_event, progress_cb):
    if snapshot["task_id"] == "T1":
        cancel_state["runner_started"].set()
        # 长任务：直到被取消
        canceled = cancel_event.wait(8.0)
        cancel_state["saw_cancel"] = canceled
        return {"type": "canceled", "payload": {}}
    return {"type": "completed", "payload": {"summary": "x",
                                             "outcome": "complete"}}


rt5, st5 = make_runtime(tmp, slow_runner, [])
rt5.router = lambda sid, text, snap: plan_json([
    {"op": "create_task", "local_ref": "L1", "label": "长任务",
     "goal": "长任务"}])
rt5.submit_request("s5", "查个长的", source_commit_id="c1")
rt5.commit_pending("s5")
ok = pump(rt5, lambda: cancel_state["runner_started"].is_set()
          and rt5.store.task_row("s5", "T1")["active_run_id"])
check("task running", ok)
# 用户改口取消
rt5.router = lambda sid, text, snap: plan_json([
    {"op": "cancel_tasks", "targets": [
        {"task_id": "T1", "expected_revision": 1}]}])
rt5.submit_request("s5", "算了别查了", source_commit_id="c2")
out = rt5.commit_pending("s5")
check("cancel receipt text", out["texts"] == ["正在停止长任务。"],
      str(out["texts"]))
ok = pump(rt5, lambda: rt5.store.task_row("s5", "T1")[
    "execution_status"] == C.EXEC_CANCELED)
check("worker observes cancel and task canceled", ok and
      cancel_state["saw_cancel"])

# ==== 场景 6：静默掐死重试 → retry_exhausted（不写 delivered）==============
rt6, st6 = make_runtime(tmp, completed_runner, [None, None, None])
rt6.router = lambda sid, text, snap: plan_json([
    {"op": "create_task", "local_ref": "L1", "goal": "g"}])
rt6.submit_request("s6", "查", source_commit_id="c1")
rt6.commit_pending("s6")
ok = pump(rt6, lambda: _dlv_state(rt6, "s6") == C.DLV_RETRY_EXHAUSTED)
check("silent kill retried then exhausted", ok)
check("silent kill attempts bounded by budget",
      len(st6["speak_calls"]) == 3, len(st6["speak_calls"]))
check("retry_exhausted never delivered",
      _dlv_state(rt6, "s6") != "delivered")




# ==== 场景 7：播出后打断 = 终态不重播 =======================================
rt7, st7 = make_runtime(tmp, completed_runner, [False, True])
rt7.router = lambda sid, text, snap: plan_json([
    {"op": "create_task", "local_ref": "L1", "goal": "g"}])
rt7.submit_request("s7", "查", source_commit_id="c1")
rt7.commit_pending("s7")
ok = pump(rt7, lambda: _dlv_state(rt7, "s7") == C.DLV_INTERRUPTED_PARTIAL)
check("interrupted-after-audio terminal", ok)
time.sleep(0.3)
rt7.delivery.pump()
rt7.scheduler.pump()
check("no auto-replay after interruption", len(st7["speak_calls"]) == 1)

# ==== 场景 8：stop_speech 只停当前发言 ======================================
rt8, st8 = make_runtime(tmp, completed_runner, ["block"])
rt8.router = lambda sid, text, snap: plan_json([
    {"op": "create_task", "local_ref": "L1", "goal": "g"}])
rt8.submit_request("s8", "查", source_commit_id="c1")
rt8.commit_pending("s8")
# 后台交付线程：speak 阻塞期间处理 stop_speech
rt8.delivery.poll_interval = 0.05
rt8.scheduler.start()
rt8.delivery.start()
ok = pump(rt8, lambda: len(st8["speak_calls"]) > 0 and
          st8["speak_calls"][0]["stop"] is not None)
check("speech started (blocking)", ok)
rt8.store.enqueue_outbox("s8", "stop_speech", {"operation_id": "op-x"})
ok = pump(rt8, lambda: _dlv_state(rt8, "s8") == C.DLV_INTERRUPTED_PARTIAL,
          timeout=6)
check("stop_speech interrupts current speech", ok)
rt8.scheduler.stop()
rt8.delivery.stop()

# ==== 场景 9：改口后旧结果抑制 =============================================
rt9, st9 = make_runtime(tmp, completed_runner, [True])
rt9.router = lambda sid, text, snap: plan_json([
    {"op": "create_task", "local_ref": "L1", "goal": "g"}])
rt9.submit_request("s9", "查", source_commit_id="c1")
rt9.commit_pending("s9")
# 任务完成但话权不可用 → 结果在议程里等
ok = pump(rt9, lambda: rt9.store.task_row("s9", "T1")[
    "execution_status"] == C.EXEC_COMPLETED)
st9["can_deliver"] = False
check("completed result waiting in agenda", len(st9["speak_calls"]) == 0)
# 用户改口 amend → 旧 revision 的结果交付应被抑制
rt9.router = lambda sid, text, snap: plan_json([
    {"op": "amend_task", "target_task_id": "T1", "expected_revision": 1,
     "patch": {"location": "上海"}}])
rt9.submit_request("s9", "改查上海的", source_commit_id="c2")
rt9.commit_pending("s9")
st9["can_deliver"] = True
ok = pump(rt9, lambda: len(st9["speak_calls"]) > 0)
check("amended task reruns and delivers new revision",
      ok and st9["speak_calls"][0]["env"].revision == 2,
      str(st9["speak_calls"][:1]))
# 旧 revision 的结果从未被播出：议程条目在改口后过不了有效性核验
check("old-revision result never spoken",
      all(c["env"].revision == 2 for c in st9["speak_calls"]),
      str([(c["env"].revision) for c in st9["speak_calls"]]))

# ==== 场景 10：每会话运行上限 ==============================================
seen_running = {"n": 0, "max": 0}
lock = threading.Lock()


def counting_runner(snapshot, cancel_event, progress_cb):
    with lock:
        seen_running["n"] += 1
        seen_running["max"] = max(seen_running["max"], seen_running["n"])
    time.sleep(0.15)
    with lock:
        seen_running["n"] -= 1
    return {"type": "completed", "payload": {"summary": "x",
                                             "outcome": "complete"}}


rt10, st10 = make_runtime(tmp, counting_runner, [True] * 3)
rt10.scheduler.poll_interval = 0.02
rt10.router = lambda sid, text, snap: plan_json([
    {"op": "create_task", "local_ref": "L1", "goal": "g1"},
    {"op": "create_task", "local_ref": "L2", "goal": "g2"},
    {"op": "create_task", "local_ref": "L3", "goal": "g3"}])
rt10.submit_request("s10", "三件事", source_commit_id="c1")
rt10.commit_pending("s10")
ok = pump(rt10, lambda: all(
    rt10.store.task_row("s10", f"T{i}")["execution_status"]
    == C.EXEC_COMPLETED for i in (1, 2, 3)))
check("all three tasks complete", ok)
check("per-session running cap respected",
      seen_running["max"] <= rt10.scheduler.max_running,
      f"max concurrent = {seen_running['max']}")

# ==== 场景 11：窄版 voiced-by-lane（2026-11-09）============================
# commit 带 chitchat_spoke=True（创建轮闲聊路已开口）+ 完成 run 零工具 +
# 首 run → 结果直接标 voiced_by_lane：不入 deliver outbox、泵不出声、
# 面板 delivered。带工具的 run / 创建轮没开口 / 失败照常入队走泵。


def joke_runner(snapshot, cancel_event, progress_cb):
    return {"type": "completed",
            "payload": {"summary": "从前有座山，山里有座庙……",
                        "facts": [], "sources": [], "outcome": "complete"}}


def search_runner(snapshot, cancel_event, progress_cb):
    progress_cb("tool_call", {"tool": "web_search"})
    return {"type": "completed",
            "payload": {"summary": "北京晴，12 度",
                        "facts": [], "sources": [], "outcome": "complete"}}


def _deliver_count(rt, sid):
    with rt.store._lock:
        return rt.store._conn.execute(
            "SELECT COUNT(*) AS c FROM outbox WHERE session_id=? AND "
            "kind='deliver'", (sid,)).fetchone()["c"]


# 11a 事故形态：讲笑话任务，闲聊路已开口，run 零工具 → 泵被跳过
rt11, st11 = make_runtime(tmp, joke_runner, [])
rt11.router = lambda sid, text, snap: plan_json([
    {"op": "create_task", "local_ref": "L1", "label": "讲笑话",
     "goal": "讲个笑话"}])
rt11.submit_request("s11", "给我讲个笑话", source_commit_id="c1")
rt11.commit_pending("s11", chitchat_spoke=True)
ok = pump(rt11, lambda: rt11.store.task_row("s11", "T1")[
    "execution_status"] == C.EXEC_COMPLETED)
check("11a joke task completes", ok)
ok = pump(rt11, lambda: _dlv_state(rt11, "s11") == C.DLV_VOICED_BY_LANE)
check("11b zero-tool result marked voiced_by_lane", ok,
      str(_dlv_state(rt11, "s11")))
check("11c no deliver outbox enqueued",
      _deliver_count(rt11, "s11") == 0)
check("11d pump never speaks", not st11["speak_calls"],
      str(st11["speak_calls"]))
check("11e panel row delivered",
      rt11.store.task_panel_rows("s11")[0]["delivered"] is True)

# 11f 对照：同类任务但 run 带工具（查询类）→ 照常入队走泵
rt11f, st11f = make_runtime(tmp, search_runner, [True])
rt11f.router = lambda sid, text, snap: plan_json([
    {"op": "create_task", "local_ref": "L1", "label": "查天气",
     "goal": "查北京天气"}])
rt11f.submit_request("s11f", "帮我查北京天气", source_commit_id="c1")
rt11f.commit_pending("s11f", chitchat_spoke=True)
ok = pump(rt11f, lambda: len(st11f["speak_calls"]) > 0)
check("11f tool-using result still delivered by pump", ok,
      str(st11f["speak_calls"]))

# 11g 对照：创建轮闲聊路没开口（chitchat_spoke=False）→ 零工具也走泵
rt11g, st11g = make_runtime(tmp, joke_runner, [True])
rt11g.router = lambda sid, text, snap: plan_json([
    {"op": "create_task", "local_ref": "L1", "label": "讲笑话",
     "goal": "讲个笑话"}])
rt11g.submit_request("s11g", "给我讲个笑话", source_commit_id="c1")
rt11g.commit_pending("s11g")          # 默认 chitchat_spoke=False
ok = pump(rt11g, lambda: len(st11g["speak_calls"]) > 0)
check("11g unvoiced creation turn still pumps", ok,
      str(st11g["speak_calls"]))

for r in (rt, rt2, rt3, rt4, rt5, rt6, rt7, rt9, rt10, rt11, rt11f, rt11g):
    r.scheduler.stop()
    r.delivery.stop()
    r.store.close()

print()
if FAILS:
    print(f"agent_sched_test: {len(FAILS)} FAILED: {FAILS}")
    sys.exit(1)
print("agent_sched_test: all passed")
