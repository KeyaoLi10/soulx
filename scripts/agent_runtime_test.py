# -*- coding: utf-8 -*-
"""Offline tests for agent/contracts.py (P1a, 手册 §5/§7/§13).

纯 stdlib 契约层，不 import app/clients；覆盖严格 plan 解析的白名单、
引用完整性、长度上限、cancel 缺 target 拒绝、amend patch 键白名单、
回执模板与 SpeechEnvelope 按 kind 的有效性判定。

Run: /home/research/krisl/soulx/venv/bin/python -I scripts/agent_runtime_test.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "repo" / "dialogue_system"))

from agent import contracts as C  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok  {name}")
    else:
        FAILS.append(name)
        print(f"FAIL  {name}  {detail}")


def plan_ok(ops, **kw):
    d = {"schema_version": 1, "operations": ops}
    d.update(kw)
    return json.dumps(d, ensure_ascii=False)


def expect_plan_error(name, plan_json):
    try:
        C.parse_plan(plan_json)
    except C.PlanError as e:
        check(name, True)
        return str(e)
    except Exception as e:  # noqa: BLE001
        check(name, False, f"wrong exception type: {type(e).__name__}: {e}")
        return ""
    check(name, False, "no PlanError raised")
    return ""


# ---- 基本解析 -----------------------------------------------------------
p = C.parse_plan(plan_ok([
    {"op": "create_task", "local_ref": "L1", "label": "天气",
     "goal": "查北京明天天气", "constraints": {"city": "北京"},
     "depends_on_local_refs": []},
    {"op": "get_task", "list_scope": "active"},
]))
check("create+get parses", p.op_types() == ["create_task", "get_task"])
check("default reply_mode direct", p.reply_mode == "direct")
check("constraints dict kept", p.operations[0].constraints == {"city": "北京"})

p2 = C.parse_plan(plan_ok(
    [{"op": "get_task", "list_scope": "all_recent"}], reply_mode="clarify",
    clarification="你说的是查天气还是查火车票？"))
check("clarify parses", p2.reply_mode == "clarify" and
      p2.clarification.startswith("你说的是"))

# 全 op 类型可解析
p3 = C.parse_plan(plan_ok([
    {"op": "create_task", "local_ref": "L1", "goal": "g1"},
    {"op": "create_task", "local_ref": "L2", "goal": "g2",
     "depends_on_local_refs": ["L1"]},
    {"op": "amend_task", "target_task_id": "T3", "expected_revision": 2,
     "patch": {"time_range": "明天"}},
    {"op": "cancel_tasks", "targets": [
        {"task_id": "T3", "expected_revision": 3}]},
    {"op": "rerun_task", "target_task_id": "T4", "expected_revision": 1,
     "refresh_reason": "数据过期"},
    {"op": "get_task", "target_task_id": "T4"},
    {"op": "answer_task_question", "target_task_id": "T4",
     "expected_revision": 2, "question_id": "q1", "answer": "北京"},
    {"op": "followup_task", "parent_task_id": "T4", "parent_revision": 2,
     "goal": "再查后天", "local_ref": "L3"},
]))
check("all 8 op kinds parse", set(p3.op_types()) == {
    "create_task", "amend_task", "cancel_tasks", "rerun_task", "get_task",
    "answer_task_question", "followup_task"})

# ---- 拒绝路径 -----------------------------------------------------------
expect_plan_error("empty plan", "")
expect_plan_error("not json", "{oops")
expect_plan_error("array not object", "[1,2]")
expect_plan_error("wrong schema_version",
                  json.dumps({"schema_version": 2, "operations": []}))
expect_plan_error("missing operations",
                  json.dumps({"schema_version": 1}))
expect_plan_error("unknown op kind", plan_ok([{"op": "delete_everything"}]))
expect_plan_error("9 ops over cap",
                  plan_ok([{"op": "get_task", "list_scope": "active"}] * 9))
expect_plan_error("goal too long",
                  plan_ok([{"op": "create_task", "local_ref": "L1",
                            "goal": "x" * 601}]))
expect_plan_error("create missing goal",
                  plan_ok([{"op": "create_task", "local_ref": "L1"}]))
expect_plan_error("duplicate local_ref",
                  plan_ok([{"op": "create_task", "local_ref": "L1", "goal": "a"},
                           {"op": "create_task", "local_ref": "L1", "goal": "b"}]))
expect_plan_error("depends on unknown ref",
                  plan_ok([{"op": "create_task", "local_ref": "L1", "goal": "a",
                            "depends_on_local_refs": ["L9"]}]))
# G2 回归：cancel 缺 target 必须整批拒绝，绝不解释成"全部"
expect_plan_error("cancel without targets",
                  plan_ok([{"op": "cancel_tasks"}]))
expect_plan_error("cancel empty targets",
                  plan_ok([{"op": "cancel_tasks", "targets": []}]))
expect_plan_error("cancel target not snapshot id",
                  plan_ok([{"op": "cancel_tasks",
                            "targets": [{"task_id": "全部", "expected_revision": 1}]}]))
# G1：amend patch 白名单
expect_plan_error("amend writes status",
                  plan_ok([{"op": "amend_task", "target_task_id": "T1",
                            "expected_revision": 1, "patch": {"status": "done"}}]))
expect_plan_error("amend no patch",
                  plan_ok([{"op": "amend_task", "target_task_id": "T1",
                            "expected_revision": 1}]))
expect_plan_error("expected_revision zero",
                  plan_ok([{"op": "amend_task", "target_task_id": "T1",
                            "expected_revision": 0, "patch": {"query": "x"}}]))
expect_plan_error("runtime_query not whitelisted",
                  plan_ok([{"op": "runtime_query", "query": "delete_all_tasks"}]))
expect_plan_error("clarify without text",
                  plan_ok([{"op": "get_task", "list_scope": "active"}],
                          reply_mode="clarify"))
expect_plan_error("bad reply_mode",
                  plan_ok([{"op": "get_task", "list_scope": "active"}],
                          reply_mode="monologue"))
expect_plan_error("get_task without target or scope",
                  plan_ok([{"op": "get_task"}]))
# 截断 JSON（G6 同源场景：不闭合的计划绝不部分执行）
expect_plan_error("truncated json",
                  '{"schema_version": 1, "operations": [{"op": "create_tas')

# cancel scope=all_active 显式允许（用户明说"全都别做了"）
p4 = C.parse_plan(plan_ok([{"op": "cancel_tasks", "scope": "all_active"}]))
check("cancel all_active explicit", p4.operations[0].scope == "all_active")

# 证据引语：quote 或 evidence_quote 都收
p5 = C.parse_plan(plan_ok([
    {"op": "create_task", "local_ref": "L1", "goal": "g",
     "quote": "帮我查天气"}]))
check("evidence via quote", p5.operations[0].evidence == "帮我查天气")

# ---- 回执模板（§13.1）---------------------------------------------------
r = C.Receipt(operation_id="req1:p1:0", op="create_task", task_id="T1",
              outcome=C.RC_ACCEPTED)
t = C.receipt_text(r, label="查天气")
check("receipt accepted text", t == "查天气已经安排好了，稍后告诉你结果。", t)
r2 = C.Receipt(operation_id="req1:p1:1", op="cancel_tasks", task_id="T1",
               outcome=C.RC_REJECTED, reason_code="unknown_task")
check("receipt rejected unknown_task",
      C.receipt_text(r2) == "我没有找到你说的那个任务。")
r3 = C.Receipt(operation_id="x", op="cancel_tasks",
               outcome=C.RC_REJECTED, reason_code="queue_full")
check("receipt rejected queue_full",
      C.receipt_text(r3) == "现在排队的任务有点多，这个先没受理。")
r4 = C.Receipt(operation_id="x", op="amend_task", task_id="T1",
               outcome=C.RC_UPDATED)
check("receipt updated text",
      C.receipt_text(r4, label="火车票") == "好的，火车票已经按你说的改了。")
check("receipt serializable",
      C.Receipt(operation_id="x", op="get_task").to_dict()["op"] == "get_task")

# ---- SpeechEnvelope 按 kind 的有效性（§13.2）---------------------------
env = C.SpeechEnvelope(speech_id="s1", session_id="sess", connection_epoch=1,
                       kind="result", task_id="T1", revision=2)
check("result valid on completed",
      env.valid_for({"T1": {"revision": 2,
                            "execution_status": C.EXEC_COMPLETED}}))
check("result invalid on running",
      not env.valid_for({"T1": {"revision": 2,
                                "execution_status": C.EXEC_RUNNING}}))
check("result invalid on stale revision",
      not env.valid_for({"T1": {"revision": 3,
                                "execution_status": C.EXEC_COMPLETED}}))
check("result invalid on unknown task", not env.valid_for({}))
check("result invalid on retracted (cancel_intent)",
      not env.valid_for({"T1": {"revision": 2,
                                "execution_status": C.EXEC_COMPLETED,
                                "cancel_intent": 1}}))
q_env = C.SpeechEnvelope(speech_id="s2", session_id="sess", connection_epoch=1,
                         kind="question", task_id="T1", revision=2)
check("question valid only waiting_input",
      q_env.valid_for({"T1": {"revision": 2,
                              "execution_status": C.EXEC_WAITING_INPUT}})
      and not q_env.valid_for({"T1": {"revision": 2,
                                      "execution_status": C.EXEC_RUNNING}}))
c_env = C.SpeechEnvelope(speech_id="s3", session_id="sess", connection_epoch=1,
                         kind="receipt", task_id="T1", revision=2)
check("receipt valid on cancel_requested",
      c_env.valid_for({"T1": {"revision": 2,
                              "execution_status": C.EXEC_CANCEL_REQUESTED}}))
check("taskless envelope always valid",
      C.SpeechEnvelope(speech_id="s4", session_id="sess",
                       connection_epoch=1, kind="result").valid_for({}))

# ---- operation_id 幂等拼接（I3）----------------------------------------
check("operation_id deterministic",
      C.operation_id_for("r1", 1, 0) == C.operation_id_for("r1", 1, 0) ==
      "r1:p1:0")
check("operation_id varies by version",
      C.operation_id_for("r1", 2, 0) != C.operation_id_for("r1", 1, 0))

# ---- 事件信封（§10）----------------------------------------------------
ev = C.event_envelope("e1", "sess", "T1", 2, "run9", "tool_result",
                      {"tool": "web_search", "ok": True})
check("event envelope shape",
      ev["type"] == "tool_result" and ev["run_id"] == "run9"
      and ev["payload"]["tool"] == "web_search")

# ---- TaskExecutionSnapshot ---------------------------------------------
snap = C.TaskExecutionSnapshot(
    session_id="sess", task_id="T1", revision=2, run_id="run9",
    goal="查天气", constraints={"city": "北京"}, context="",
    user_text="帮我看看北京明天天气")
check("snapshot keeps user_text", snap.to_dict()["user_text"].startswith("帮我"))

print()
if FAILS:
    print(f"agent_runtime_test: {len(FAILS)} FAILED: {FAILS}")
    sys.exit(1)
print("agent_runtime_test: all passed")
