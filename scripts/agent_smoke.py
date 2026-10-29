#!/usr/bin/env python
"""Agent 冒烟测试：工具层（离线部分）+ 慢脑端到端（需 :6008 慢脑实例在线）。

用法（服务已拉起时）：
  /home/research/krisl/soulx/venv/bin/python -I scripts/agent_smoke.py
加 --offline 只跑工具层离线部分（不碰 LLM）。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "repo", "dialogue_system"))

import argparse

PASS, FAIL = 0, 0


def check(name, ok, detail=""):
    global PASS, FAIL
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)
    PASS, FAIL = PASS + ok, FAIL + (not ok)


def smoke_tools():
    print("== 工具层（离线/网络部分）==", flush=True)
    from agent import tools as T

    out = T.web_search("SoulX-Duplug full-duplex dialogue")
    check("web_search 返回编号结果", out.splitlines()[0].startswith("1.") and "http" in out,
          out.splitlines()[0][:80])

    url = [ln.strip() for ln in out.splitlines()
           if ln.strip().startswith("http")][:1]
    if url:
        page = T.fetch_page(url[0])
        check("fetch_page 抓到正文", len(page) > 200, f"{url[0][:50]} -> {len(page)} chars")
    else:
        check("fetch_page 抓到正文", False, "no url from search")

    out = T.python_repl("print(123456789**2)")
    check("python_repl 计算正确", "15241578750190521" in out, out.strip()[:60])

    out = T.python_repl("import time\ntime.sleep(30)")
    check("python_repl 超时保护", "timeout" in out, out.strip()[:40])

    out = T.get_datetime()
    check("get_datetime", "202" in out and "(" in out, out)

    out = T.execute_tool("no_such_tool", {})
    check("execute_tool 未知工具不抛异常", "unknown tool" in out, out)


def smoke_slow_brain():
    print("== 慢脑端到端（:6008，thinking=on + 工具循环）==", flush=True)
    from agent.slow_brain import SlowBrain

    sb = SlowBrain()
    check("慢脑实例健康", sb.available())

    import time

    events = []

    t0 = time.time()
    result = sb.run_task(
        "smoke-calc", "计算 123456789 的平方，把结果准确地说出来。",
        event_cb=lambda e: events.append(e),
    )
    dt = time.time() - t0
    kinds = [e["type"] for e in events]
    check("纯计算任务 completed",
          bool(kinds and kinds[-1] == "completed" and result),
          f"{dt:.1f}s, {len(kinds)} events")
    check("结果数值正确（用工具算，不是心算）", "15241578750190521" in (result or ""),
          (result or "")[:80])
    check("有 progress 事件（python_repl 被调用）",
          any("Python" in e.get("desc", "") for e in events if e["type"] == "progress"))

    events = []
    t0 = time.time()
    result = sb.run_task(
        "smoke-search",
        "搜索一下 SoulX-Duplug 是一个什么项目，用两句话总结。",
        event_cb=lambda e: events.append(e),
    )
    dt = time.time() - t0
    kinds = [e["type"] for e in events]
    check("搜索任务 completed",
          bool(kinds and kinds[-1] == "completed" and result),
          f"{dt:.1f}s, {len(kinds)} events")
    check("搜索任务有 search progress",
          any(e["type"] == "progress" and "搜索" in e.get("desc", "") for e in events))
    print(f"  搜索任务结果：{(result or '')[:150]}", flush=True)

    # 取消：起一个慢任务，2 秒后取消
    import threading

    cancel = threading.Event()
    events = []
    t = threading.Timer(2.0, cancel.set)
    t.start()
    t0 = time.time()
    result = sb.run_task(
        "smoke-cancel",
        "搜索 2025 年人工智能领域最重要的十项进展，详细总结每一项。",
        event_cb=lambda e: events.append(e), cancel_event=cancel,
    )
    dt = time.time() - t0
    kinds = [e["type"] for e in events]
    check("取消任务返回 None 且发 canceled 事件",
          result is None and kinds and kinds[-1] == "canceled", f"{dt:.1f}s")


def smoke_runtime_delivery():
    """确定性验证 Agent Runtime 的交付仲裁（不碰 LLM/网络）。

    用假 SlowBrain 覆盖三条路径：completed → 议程 → delivered；
    failed → 同样进议程并交付（用户等不到结果也必须听到解释）；
    can_deliver 不满足时等待。回归 2026-10-29 E2E 的 T2 失败静默问题。
    """
    print("== Agent Runtime 交付仲裁（离线 stub）==", flush=True)
    import threading
    import time
    from agent import runtime as RT

    class FakeSlow:
        def __init__(self, outcome, result=None, error=None):
            self.outcome, self.result, self.error = outcome, result, error

        def available(self):
            return True

        def run_task(self, task_id, goal, context="", event_cb=None,
                     cancel_event=None):
            if self.outcome == "hang":  # stay running until canceled
                if event_cb:
                    event_cb({"type": "started"})
                (cancel_event or threading.Event()).wait()
                return None
            for evt in ({"type": "started"},
                        {"type": self.outcome,
                         **({"result": self.result} if self.outcome == "completed"
                            else {"error": self.error})}):
                if event_cb:
                    event_cb(evt)

    delivered = []
    emitted = []              # (event, view) pairs the frontend would receive
    busy = threading.Event()  # can_deliver gate

    def make_rt(outcome, **kw):
        rt = RT.AgentRuntime(
            "smoke-client",
            emit=lambda ev, data: emitted.append((ev, data)),
            can_deliver=lambda: not busy.is_set(),
            deliver=lambda entry: (delivered.append(entry) or True),
        )
        rt.slow = FakeSlow(outcome, **kw)
        return rt

    # 1) completed -> delivered
    rt = make_rt("completed", result="答案是 42。")
    rt.start()
    rt.delegate("计算人生意义", "")
    ok = _wait(lambda: any(t.status == "delivered" for t in rt.tasks.values()), 5)
    check("completed 任务经议程交付为 delivered", ok)
    check("交付内容带 result", delivered and delivered[-1]["result"] == "答案是 42。")
    check("delivered 状态有 task_update 推送（前端可见）",
          any(d.get("status") == "delivered" for _, d in emitted))
    rt.shutdown()

    # 2) failed -> 同样交付（回归点）
    delivered.clear()
    rt = make_rt("failed", error="task timeout")
    rt.start()
    rt.delegate("查一个查不到的东西", "")
    ok = _wait(lambda: any(t.status == "delivered" for t in rt.tasks.values()), 5)
    check("failed 任务也进议程并交付（不静默）", ok,
          "delivery entry error=" + (delivered[0]["error"] if delivered else "无"))
    rt.shutdown()

    # 3) 用户忙时不交付，空闲后交付
    delivered.clear()
    rt = make_rt("completed", result="晚到的结果。")
    busy.set()
    rt.start()
    rt.delegate("慢任务", "")
    time.sleep(1.2)
    check("用户忙时（can_deliver=False）不交付", not delivered)
    busy.clear()
    ok = _wait(lambda: len(delivered) >= 1, 5)
    check("空闲后补交付", ok)
    rt.shutdown()

    # 4) 确定性取消意图检测（auto_cancel）
    rt = make_rt("hang")
    rt.start()
    rt.delegate("查珠峰高度", "")
    ok = _wait(lambda: any(t.status == "running" for t in rt.tasks.values()), 5)
    check("前置：任务已存在", ok)
    note = rt.auto_cancel("算了算了，不用查了，取消掉吧。")
    check("改口意图 → 自动取消并返回提示", note is not None
          and all(t.status == "canceled" for t in rt.tasks.values()), note or "")
    check("闲聊不触发取消", rt.auto_cancel("你觉得听音乐对放松有帮助吗？") is None)
    check("无任务时改口返回 None", rt.auto_cancel("算了不用查了") is None)
    rt.shutdown()

    rt = make_rt("hang")
    rt.start()
    rt.delegate("查天气", "")
    _wait(lambda: any(t.status == "running" for t in rt.tasks.values()), 5)
    check("否定式（别取消，继续查）不触发", rt.auto_cancel("别取消，继续查") is None)
    rt.shutdown()


def _wait(pred, timeout):
    import time
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.1)
    return False


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="只跑工具层，不碰 LLM")
    args = ap.parse_args()
    smoke_runtime_delivery()
    smoke_tools()
    if not args.offline:
        smoke_slow_brain()
    print(f"\n合计: {PASS} pass, {FAIL} fail")
    sys.exit(1 if FAIL else 0)
