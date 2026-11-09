# -*- coding: utf-8 -*-
"""Offline tests for agent/store.py (P1b, 手册 §5/§8/§9)。

无 GPU/无服务：内存级别的 SQLite（临时文件）+ 直接调用 commit_plan。
覆盖：幂等重放、批内原子（崩溃窗口回滚）、expected_revision 复验、
cancel 三层、amend 原子性、队列上限、依赖成环、outbox lease、run 围栏、
delivery 状态机与抑制。

Run: /home/research/krisl/soulx/venv/bin/python -I scripts/agent_store_test.py
"""
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "repo" / "dialogue_system"))

from agent import contracts as C  # noqa: E402
from agent import store as S  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok  {name}")
    else:
        FAILS.append(name)
        print(f"FAIL  {name}  {detail}")


def plan(ops, **kw):
    d = {"schema_version": 1, "operations": ops}
    d.update(kw)
    return C.parse_plan(json.dumps(d, ensure_ascii=False))


def create(label, ref="L1", goal="查天气", deps=()):
    return {"op": "create_task", "local_ref": ref, "label": label,
            "goal": goal, "depends_on_local_refs": list(deps)}


tmp = tempfile.mkdtemp(prefix="agent_store_test_")
db = S.AgentStore(os.path.join(tmp, "t.sqlite3")) if (os := __import__("os")) \
    else None

SID = "sess-test"

# ---- 会话与请求登记 -----------------------------------------------------
db.ensure_session(SID)
r1 = db.register_request(SID, "帮我查北京天气", source_commit_id="c1",
                         engine_epoch="web-1", input_revision=1)
r1b = db.register_request(SID, "帮我查北京天气", source_commit_id="c1",
                          engine_epoch="web-1", input_revision=1)
check("request registration idempotent", r1 == r1b)
r2 = db.register_request(SID, "再查上海", source_commit_id="c2",
                         engine_epoch="web-1", input_revision=1)
check("command_seq increments",
      db.request_row(r2)["command_seq"] == 2)
r3 = db.register_request(SID, "两遍ASR修订", source_commit_id="c1",
                         engine_epoch="web-1", input_revision=2)
check("input_revision distinguishes requests", r3 not in (r1, r2))

# ---- create + 幂等重放 ---------------------------------------------------
p = plan([create("天气", goal="查北京明天天气")])
res = db.commit_plan(SID, r1, p)
check("create accepted", res.receipts[0].outcome == C.RC_ACCEPTED
      and res.receipts[0].task_id == "T1")
check("binding local_ref", res.bindings.get("L1") == "T1")
view = db.task_view(SID)
check("task queued in view",
      len(view) == 1 and view[0]["execution_status"] == C.EXEC_QUEUED
      and view[0]["revision"] == 1)
n_events = len(db.events_for_task(SID, "T1", limit=100))
# 同一 request + plan_version 重放
res2 = db.commit_plan(SID, r1, p)
check("replay returns same receipt", res2.replayed
      and res2.receipts[0].to_dict() == res.receipts[0].to_dict())
check("replay creates nothing new",
      len(db.task_view(SID)) == 1
      and len(db.events_for_task(SID, "T1", limit=100)) == n_events)
# 不同 plan_version（重新路由）= 新 operation_id，允许引用旧回执或补充操作
res3 = db.commit_plan(SID, r1, plan([{"op": "get_task", "list_scope": "all_recent"}]),
                      plan_version=2)
check("replan new version commits", not res3.replayed)

# ---- 批内原子：崩溃窗口 ---------------------------------------------------
before_tasks = len(db.task_view(SID))
try:
    db.commit_plan(SID, r2, plan([
        create("任务A", ref="L1"), create("任务B", ref="L2")]),
        _fail_after=0)
    crashed = False
except S.StoreError:
    crashed = True
check("injected crash raises", crashed)
check("crash leaves no half batch",
      len(db.task_view(SID)) == before_tasks)
r4 = db.register_request(SID, "atomic test", source_commit_id="c4",
                         engine_epoch="web-1")
res = db.commit_plan(SID, r4, plan([
    create("任务A", ref="L1"), create("任务B", ref="L2")]))
check("both creates commit atomically",
      [x.outcome for x in res.receipts] == [C.RC_ACCEPTED, C.RC_ACCEPTED])

# ---- expected_revision 复验 ---------------------------------------------
r5 = db.register_request(SID, "amend", source_commit_id="c5",
                         engine_epoch="web-1")
res = db.commit_plan(SID, r5, plan([
    {"op": "amend_task", "target_task_id": "T1", "expected_revision": 9,
     "patch": {"time_range": "后天"}}]))
check("amend stale revision rejected",
      res.receipts[0].outcome == C.RC_REJECTED
      and res.receipts[0].reason_code == "stale_revision")
check("stale amend leaves task untouched",
      db.task_row(SID, "T1")["current_revision"] == 1)
res = db.commit_plan(SID, r5, plan([
    {"op": "amend_task", "target_task_id": "T1", "expected_revision": 1,
     "patch": {"time_range": "后天"}}]), plan_version=2)
check("amend correct revision updates",
      res.receipts[0].outcome == C.RC_UPDATED
      and db.task_row(SID, "T1")["current_revision"] == 2)
rev2 = db.revision_row(SID, "T1", 2)
check("amend patch merged into constraints",
      json.loads(rev2["constraints_json"]).get("time_range") == "后天"
      and rev2["goal"] == "查北京明天天气")

# ---- amend 抑制旧 delivery + 停旧 run ------------------------------------
d1 = db.record_delivery(SID, "result", task_id="T1", revision=2,
                        run_id=None)
run_id = db.create_run(SID, "T1", 2, snapshot={"goal": "查北京明天天气"})
tok = db.lease_run(run_id)
check("run leased", bool(tok))
r6 = db.register_request(SID, "amend2", source_commit_id="c6",
                         engine_epoch="web-1")
db.commit_plan(SID, r6, plan([
    {"op": "amend_task", "target_task_id": "T1", "expected_revision": 2,
     "patch": {"location": "上海"}}]))
check("amend suppresses old-revision delivery",
      db.delivery_row(d1)["state"] == C.DLV_SUPPRESSED)
run_row = db._conn.execute("SELECT state FROM runs WHERE run_id=?",
                           (run_id,)).fetchone()
check("amend marks old run stop_requested",
      run_row["state"] == "stop_requested")
check("task back to queued on new revision",
      db.task_row(SID, "T1")["execution_status"] == C.EXEC_QUEUED
      and db.task_row(SID, "T1")["current_revision"] == 3
      and db.task_row(SID, "T1")["active_run_id"] is None)

# ---- run 围栏：过期 run 结果只进审计 --------------------------------------
stale_run = db.create_run(SID, "T1", 3)
stale_tok = db.lease_run(stale_run)
r7 = db.register_request(SID, "amend3", source_commit_id="c7",
                         engine_epoch="web-1")
db.commit_plan(SID, r7, plan([
    {"op": "amend_task", "target_task_id": "T1", "expected_revision": 3,
     "patch": {"detail_level": "high"}}]))
out = db.finish_run(SID, "T1", 3, stale_run, stale_tok, C.EXEC_COMPLETED,
                    result_summary="旧结果")
check("stale run finish fenced", out == "fenced")
check("fenced result does not touch task",
      db.task_row(SID, "T1")["execution_status"] == C.EXEC_QUEUED
      and db.task_row(SID, "T1")["current_revision"] == 4)
evs = [e["type"] for e in db.events_for_task(SID, "T1", limit=100)]
check("fenced finish audited as stale_run_finished",
      "stale_run_finished" in evs and "completed" not in evs)

# ---- run lease 排他 -------------------------------------------------------
cur_run = db.create_run(SID, "T1", 4)
t_a = db.lease_run(cur_run)
t_b = db.lease_run(cur_run)
check("second lease rejected while held", t_b is None)
check("heartbeat extends", db.heartbeat_run(cur_run, t_a))
check("heartbeat with wrong token fails", not db.heartbeat_run(cur_run, "xx"))
try:
    db.finish_run(SID, "T1", 4, cur_run, "wrong-token", C.EXEC_COMPLETED)
    wrong_tok = False
except S.StoreError:
    wrong_tok = True
check("finish with wrong token rejected", wrong_tok)
# 正常完成路径
out = db.finish_run(SID, "T1", 4, cur_run, t_a, C.EXEC_COMPLETED,
                    result_summary="北京晴")
check("finish updates task completed",
      out == C.EXEC_COMPLETED
      and db.task_row(SID, "T1")["execution_status"] == C.EXEC_COMPLETED)

# ---- cancel 三层 ----------------------------------------------------------
# 1) completed 任务取消 → already_completed + delivery suppressed
d2 = db.record_delivery(SID, "result", task_id="T1", revision=4)
r8 = db.register_request(SID, "cancel done", source_commit_id="c8",
                         engine_epoch="web-1")
res = db.commit_plan(SID, r8, plan([
    {"op": "cancel_tasks", "targets": [
        {"task_id": "T1", "expected_revision": 4}]}]))
check("cancel completed → already_completed",
      res.receipts[0].outcome == C.RC_ALREADY_COMPLETED)
check("completed delivery suppressed",
      db.delivery_row(d2)["state"] == C.DLV_SUPPRESSED)
check("completed task not rewritten to canceled",
      db.task_row(SID, "T1")["execution_status"] == C.EXEC_COMPLETED)

# 2) queued 无 run → 直接 canceled
r9 = db.register_request(SID, "cancel queued", source_commit_id="c9",
                         engine_epoch="web-1")
db.commit_plan(SID, r9, plan([create("火车票", ref="L1")]))
t2 = db.task_view(SID)[-1]["task_id"]
r10 = db.register_request(SID, "cancel it", source_commit_id="c10",
                          engine_epoch="web-1")
res = db.commit_plan(SID, r10, plan([
    {"op": "cancel_tasks", "targets": [
        {"task_id": t2, "expected_revision": 1}]}]))
check("cancel queued no-run → canceled",
      res.receipts[0].outcome == C.RC_CANCELED
      and db.task_row(SID, t2)["execution_status"] == C.EXEC_CANCELED)

# 3) running → cancel_requested，等停止确认
r11 = db.register_request(SID, "cancel running", source_commit_id="c11",
                          engine_epoch="web-1")
db.commit_plan(SID, r11, plan([create("长任务", ref="L1")]))
t3 = db.task_view(SID)[-1]["task_id"]
run3 = db.create_run(SID, t3, 1)
tok3 = db.lease_run(run3)
r12 = db.register_request(SID, "cancel it now", source_commit_id="c12",
                          engine_epoch="web-1")
res = db.commit_plan(SID, r12, plan([
    {"op": "cancel_tasks", "targets": [
        {"task_id": t3, "expected_revision": 1}]}]))
check("cancel running → cancel_requested",
      res.receipts[0].outcome == C.RC_CANCEL_REQUESTED
      and db.task_row(SID, t3)["execution_status"]
      == C.EXEC_CANCEL_REQUESTED)
out = db.finish_run(SID, t3, 1, run3, tok3, C.EXEC_CANCELED)
check("worker confirms cancellation",
      out == C.EXEC_CANCELED
      and db.task_row(SID, t3)["execution_status"] == C.EXEC_CANCELED)

# ---- 队列上限 -------------------------------------------------------------
SID2 = "sess-queue"
db.ensure_session(SID2)
rq = db.register_request(SID2, "flood", source_commit_id="q0",
                         engine_epoch="web-1")
outcomes = []
for i in range(S.MAX_QUEUED_PER_SESSION + 2):
    p = plan([create(f"任务{i}", ref="L1", goal=f"g{i}")],
             ) if i == 0 else plan([create(f"任务{i}", ref="L1",
                                           goal=f"g{i}")])
    res = db.commit_plan(SID2, f"{rq}-v{i}", p)
    outcomes.append(res.receipts[0].outcome)
check("queue full rejects at cap",
      outcomes[:S.MAX_QUEUED_PER_SESSION] == [C.RC_ACCEPTED] *
      S.MAX_QUEUED_PER_SESSION
      and outcomes[-1] == C.RC_REJECTED)
full_res = db.commit_plan(SID2, f"{rq}-v-final",
                          plan([create("溢出", ref="L1")]))
check("queue-full reason recorded",
      full_res.receipts[0].reason_code == "queue_full")

# ---- 依赖与成环 -----------------------------------------------------------
SID3 = "sess-dep"
db.ensure_session(SID3)
rd = db.register_request(SID3, "deps", source_commit_id="d0",
                         engine_epoch="web-1")
res = db.commit_plan(SID3, rd, plan([
    create("查A", ref="L1"), create("查B", ref="L2"),
    create("比较AB", ref="L3", goal="比较A和B", deps=["L1", "L2"])]))
check("dependent create accepted",
      [x.outcome for x in res.receipts] == [C.RC_ACCEPTED] * 3)
dep_rev = db.revision_row(SID3, "T3", 1)
deps = json.loads(dep_rev["depends_json"])
check("depends resolved to task ids",
      sorted(d["task_id"] for d in deps) == ["T1", "T2"])
# 环：批内 L1 依赖 L3，L3 依赖 L1
try:
    db.commit_plan(SID3, rd, plan([
        create("x", ref="L1", goal="x", deps=["L3"]),
        create("y", ref="L3", goal="y", deps=["L1"])]),
        plan_version=2)
    cyc = False
except S.StoreError:
    cyc = True
check("batch dependency cycle rejected", cyc)
# followup 绑定父 revision
res = db.commit_plan(SID3, rd, plan([
    {"op": "followup_task", "parent_task_id": "T1", "parent_revision": 1,
     "goal": "再查细一点", "local_ref": "L9"}]), plan_version=3)
check("followup accepted with parent dep",
      res.receipts[0].outcome == C.RC_ACCEPTED)
fu = json.loads(db.revision_row(SID3, res.bindings["L9"], 1)["depends_json"])
check("followup depends on parent revision",
      fu == [{"task_id": "T1", "revision": 1}])

# ---- answer_task_question --------------------------------------------------
SID4 = "sess-q"
db.ensure_session(SID4)
rq4 = db.register_request(SID4, "q", source_commit_id="q0",
                          engine_epoch="web-1")
db.commit_plan(SID4, rq4, plan([create("订机票", ref="L1")]))
run4 = db.create_run(SID4, "T1", 1)
tok4 = db.lease_run(run4)
out = db.finish_run(SID4, "T1", 1, run4, tok4, C.EXEC_WAITING_INPUT,
                    payload={"question_id": "q1", "question": "出发城市是哪？"})
check("run can finish waiting_input",
      out == C.EXEC_WAITING_INPUT
      and db.task_row(SID4, "T1")["execution_status"]
      == C.EXEC_WAITING_INPUT)
# 非 waiting 状态回答 → 拒绝
rq5 = db.register_request(SID4, "q2", source_commit_id="q2",
                          engine_epoch="web-1")
res = db.commit_plan(SID4, rq5, plan([
    {"op": "answer_task_question", "target_task_id": "T1",
     "expected_revision": 1, "question_id": "q9", "answer": "北京"}]))
check("answer with wrong question_id rejected",
      res.receipts[0].outcome == C.RC_REJECTED
      and res.receipts[0].reason_code == "unknown_question")
# 正确回答 → 新 revision → queued
res = db.commit_plan(SID4, rq5, plan([
    {"op": "answer_task_question", "target_task_id": "T1",
     "expected_revision": 1, "question_id": "q1", "answer": "北京"}]),
    plan_version=2)
check("answer binds question to new revision queued",
      res.receipts[0].outcome == C.RC_ANSWERED
      and db.task_row(SID4, "T1")["current_revision"] == 2
      and db.task_row(SID4, "T1")["execution_status"] == C.EXEC_QUEUED)

# ---- outbox lease ----------------------------------------------------------
SID5 = "sess-ob"
db.ensure_session(SID5)
ro = db.register_request(SID5, "ob", source_commit_id="o0",
                         engine_epoch="web-1")
db.commit_plan(SID5, ro, plan([create("A", ref="L1"), create("B", ref="L2")]))
items = db.claim_outbox(limit=10, session_id=SID5)
check("outbox has schedule entries", len(items) == 2
      and all(i["kind"] == "schedule" for i in items))
check("outbox claimed entries not re-claimable",
      db.claim_outbox(limit=10, session_id=SID5) == [])
check("complete with wrong token fails",
      not db.complete_outbox(items[0]["outbox_id"], "bad"))
check("complete with token works",
      db.complete_outbox(items[0]["outbox_id"], items[0]["token"]))
# 过期 lease 可被重新领取
db._conn.execute(
    "UPDATE outbox SET lease_expires_at=? WHERE outbox_id=?",
    (time.time() - 1, items[1]["outbox_id"]))
again = db.claim_outbox(limit=10, session_id=SID5)
check("expired outbox lease reclaimable",
      len(again) == 1 and again[0]["outbox_id"] == items[1]["outbox_id"]
      and again[0]["attempts"] == 2)

# ---- delivery 状态机 --------------------------------------------------------
dd = db.record_delivery(SID5, "receipt", operation_id="op-x")
check("delivery identity dedupes",
      db.record_delivery(SID5, "receipt", operation_id="op-x") == dd)
db.set_delivery_state(dd, C.DLV_RENDERING)
db.set_delivery_state(dd, C.DLV_SENDING)
db.set_delivery_state(dd, C.DLV_SENT_UNCONFIRMED)
db.set_delivery_state(dd, C.DLV_INTERRUPTED_PARTIAL)
try:
    db.set_delivery_state(dd, C.DLV_SENDING)
    back = False
except S.StoreError:
    back = True
check("terminal delivery state immutable", back)
try:
    db.set_delivery_state(dd, "delivered")
    bad = False
except S.StoreError:
    bad = True
check("unknown delivery state rejected", bad)
# 同态幂等（2026-11-09 e2e 取消竞态）：cancel 的 _suppress_deliveries
# 与交付泵"开口后再核"并发写 suppressed——多写者收敛不是非法迁移
db.set_delivery_state(dd, C.DLV_INTERRUPTED_PARTIAL)
dd3 = db.record_delivery(SID5, "result", task_id="T9")
db.set_delivery_state(dd3, C.DLV_SUPPRESSED)
db.set_delivery_state(dd3, C.DLV_SUPPRESSED)
check("same-state delivery write is a no-op",
      db.delivery_row(dd3)["state"] == C.DLV_SUPPRESSED)

# ---- delivery_heard（渐进式披露，2026-11-09）-------------------------------
# 打断交付的"用户实际听到部分"由语音前端回写（delivery_heard 事件），
# task_panel_rows 暴露给闲聊路面板——用户问"刚才说到哪"时凭事实回答
db.record_delivery_heard(SID, "T1", "今天有8场比赛，湖人")
rows = {r["task_id"]: r for r in db.task_panel_rows(SID)}
check("heard exposed on panel row",
      rows["T1"]["heard"].startswith("今天有8场比赛"), str(rows.get("T1")))
# 最新一条覆盖旧值（按 seq 排序取每任务最新）
db.record_delivery_heard(SID, "T1", "更新后的听到部分")
rows = {r["task_id"]: r for r in db.task_panel_rows(SID)}
check("latest delivery_heard wins",
      rows["T1"]["heard"] == "更新后的听到部分", str(rows["T1"]))
# 空文本不写事件
n_heard = db._conn.execute(
    "SELECT COUNT(*) c FROM events WHERE type='delivery_heard'"
).fetchone()["c"]
db.record_delivery_heard(SID, "T1", "   ")
check("empty heard text ignored",
      db._conn.execute(
          "SELECT COUNT(*) c FROM events WHERE type='delivery_heard'"
      ).fetchone()["c"] == n_heard)

# ---- 窄版 voiced-by-lane（2026-11-09）--------------------------------------
# 标记面（creation_voiced 事件，幂等）+ 判别器（零工具 + 首 run）+
# 状态机（pending→voiced_by_lane 合法、终态、重开白名单、面板 delivered）
SID6 = "sess-vbl"
db.ensure_session(SID6)
rv = db.register_request(SID6, "讲个笑话", source_commit_id="v0",
                         engine_epoch="web-1")
res = db.commit_plan(SID6, rv, plan([create("讲笑话", ref="L1",
                                            goal="讲个笑话")]))
tj = res.bindings["L1"]
check("creation_voiced not yet marked",
      not db.task_creation_voiced(SID6, tj))
db.record_creation_voiced(SID6, [tj])
db.record_creation_voiced(SID6, [tj])
check("record_creation_voiced idempotent",
      db.task_creation_voiced(SID6, tj)
      and len([e for e in db.events_for_task(SID6, tj, limit=50)
               if e["type"] == "creation_voiced"]) == 1)
runj = db.create_run(SID6, tj, 1)
tokj = db.lease_run(runj)
check("zero-tool first run qualifies as voiced by lane",
      db.result_voiced_by_lane(SID6, tj, runj))
db.append_event(SID6, "tool_call", task_id=tj, revision=1, run_id=runj,
                payload={"tool": "web_search"})
check("tool-using run does not qualify",
      not db.result_voiced_by_lane(SID6, tj, runj)
      and db.run_tool_count(runj) == 1)
check("empty run_id is conservatively False",
      not db.result_voiced_by_lane(SID6, tj, None))
db.finish_run(SID6, tj, 1, runj, tokj, C.EXEC_COMPLETED,
              result_summary="从前有座山")
# rerun/amend 后 run 数 >1：不再判 voiced_by_lane（防误杀重执行）
rva = db.register_request(SID6, "再讲个别的", source_commit_id="v1",
                          engine_epoch="web-1")
db.commit_plan(SID6, rva, plan([
    {"op": "amend_task", "target_task_id": tj, "expected_revision": 1,
     "patch": {"detail_level": "high"}}]))
runj2 = db.create_run(SID6, tj, 2)
check("second run breaks first-run condition",
      db.task_run_count(SID6, tj) == 2
      and not db.result_voiced_by_lane(SID6, tj, runj2))
# 状态机：pending -> voiced_by_lane 合法；终态不可迁出；重开白名单含它
dv = db.record_delivery(SID6, "result", task_id=tj, revision=1, run_id=runj)
db.set_delivery_state(dv, C.DLV_VOICED_BY_LANE)
check("pending -> voiced_by_lane legal",
      db.delivery_row(dv)["state"] == C.DLV_VOICED_BY_LANE)
try:
    db.set_delivery_state(dv, C.DLV_SENDING)
    back = False
except S.StoreError:
    back = True
check("voiced_by_lane is terminal", back)
db._reopen_delivery(dv)
check("voiced_by_lane reopenable for explicit replay",
      db.delivery_row(dv)["state"] == C.DLV_PENDING)
db.set_delivery_state(dv, C.DLV_VOICED_BY_LANE)
rows = {r["task_id"]: r for r in db.task_panel_rows(SID6)}
check("panel delivered counts voiced_by_lane",
      rows[tj]["delivered"] is True, str(rows.get(tj)))

# ---- route log --------------------------------------------------------------
db.log_route(SID5, ro, "shadow", snapshot={"tasks": 2}, plan_json=None,
             parse_ok=False, error="timeout", latency_ms=42)
check("route log writes",
      db._conn.execute("SELECT COUNT(*) c FROM route_log"
                       ).fetchone()["c"] == 1)

db.close()
print()
if FAILS:
    print(f"agent_store_test: {len(FAILS)} FAILED: {FAILS}")
    sys.exit(1)
print("agent_store_test: all passed")
