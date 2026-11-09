# -*- coding: utf-8 -*-
"""Offline regression for the _speak delivery-defer gate + first-segment
prefetch (2026-11-10, session c9123a49 事故 A/B 修).

事故（23:12-23:14，会话 c9123a49）：3-chunk 的"嗯，说说。"被 commit 路
（_delivery_hold_turn）正确 hold 之后，同话轮闲聊路 _speak 仍凭
last_voice_ts 无条件 interrupt，把慢脑播报掐到 Ratio 0.37——三条执行
路径里只有 _speak 上口路不受证据阈值约束，hold 形同虚设；随后 2.1s
静默（interrupt 到 LLM 首段之间的空窗）+ 闲聊 LLM 编造结果内容。

根修（A）：_speak 的等待循环在 delivery_speech_active 期间与信号路/
commit 路同源判定——证据未达 delivery_bargein 阈值不掐播报；播报已死
（stop_event 已置）/已播完（_reply_in_flight 转 False）/证据攒够才放行。
本话轮被取代（stop_event 参数已置）直接安静退出，不再落入 interrupt 段。
根修（B）：先从生成器预取首段文本再接管话权（interrupt 后移），预取期
的打断让本回复作废；空流不接管话权。

方法：AST 抽取 app.py 的嵌套 _speak（_make_agent_v2 内闭包）与配套
模块级函数，对着假 session/假管线重放。函数签名变化会大声失败。

Run: /home/research/krisl/soulx/venv/bin/python -I scripts/speak_defer_replay.py
"""
import ast
import sys
import threading
import time
import types
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "repo" / "dialogue_system" / "app.py"
src = APP.read_text(encoding="utf-8")
tree = ast.parse(src)

# 模块级函数 + 常量（与 delivery_bargein_replay 同一抽取模式）
WANT_FNS = ("_begin_generation", "_reply_in_flight",
            "_delivery_bargein_thresholds")
WANT_CONSTS = ("_RESULT_DELIVERY_BARGEIN_NONIDLE",
               "_RESULT_DELIVERY_BARGEIN_SPAN_S")

segments = {}
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in WANT_FNS:
        segments[node.name] = ast.get_source_segment(src, node)
    elif isinstance(node, ast.Assign) and len(node.targets) == 1 \
            and isinstance(node.targets[0], ast.Name) \
            and node.targets[0].id in WANT_CONSTS:
        segments[node.targets[0].id] = ast.get_source_segment(src, node)

# 嵌套 _speak（_make_agent_v2 闭包内）
def _find_nested(root, outer, inner):
    for node in root.body:
        if isinstance(node, ast.FunctionDef) and node.name == outer:
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == inner:
                    return ast.get_source_segment(src, sub)
    return None

speak_src = _find_nested(tree, "_make_agent_v2", "_speak")
assert speak_src is not None, "app.py: _speak not found inside _make_agent_v2"
# 去掉闭包缩进（模块命名空间里 exec 需要零缩进的 def）
speak_src = "\n".join(
    line[4:] if line.startswith("        ") else line
    for line in speak_src.splitlines())

missing = [n for n in WANT_FNS + WANT_CONSTS if n not in segments]
assert not missing, f"app.py missing: {missing}"


class _Log:
    def info(self, *a, **k):
        pass
    def warning(self, *a, **k):
        pass
    def error(self, *a, **k):
        pass


class FakeSession:
    """_speak 及其依赖读到的全部 session 状态。"""
    def __init__(self, **kw):
        self.lock = threading.Lock()
        self.settings = dict(kw.pop(
            "settings", {"delivery_bargein_chunks": 5,
                         "delivery_bargein_span_ms": 1000}))
        self.playback_end = kw.pop("playback_end", 0.0)
        self.last_voice_ts = kw.pop("last_voice_ts", 0.0)
        self.assistant_audible_ts = kw.pop("assistant_audible_ts", 0.0)
        self.delivery_speech_active = kw.pop("delivery_speech_active", False)
        self.bargein_times = kw.pop("bargein_times", [])
        self.gen_in_progress = kw.pop("gen_in_progress", False)
        self.gen_serial = kw.pop("gen_serial", 0)
        self.stop_event = kw.pop("stop_event", threading.Event())
        self.calls = []           # ("cut",) ("commit",) ("interrupt",) ("reset",)
        self.pending_cut = 0

    def cut_pending_audio(self):
        self.calls.append("cut")

    def interrupt(self):
        self.calls.append("interrupt")
        self.stop_event.set()
        self.playback_end = 0.0

    def reset_interrupt(self):
        self.calls.append("reset")
        self.stop_event = threading.Event()


class Harness:
    """exec 抽取的 _speak，依赖全部打桩。records 记录时序。"""
    def __init__(self, session):
        self.session = session
        self.records = []          # ("first_yield", t) ("pipeline", chunks)
        self.tts = ("fake-tts", {})
        ns = {
            "time": time, "itertools": __import__("itertools"),
            "logger": _Log(),
            # app.py 从 pvad_gate 导入（值见 pvad_gate.py:107）
            "_PVAD_PLAYBACK_GRACE_S": 1.5,
            "session": session, "client_id": "s-test",
            "_tts_pair": lambda: self.tts,
            "_commit_pending_message":
                lambda s, c: s.calls.append("commit"),
            "_run_speech_pipeline": self._pipeline,
        }
        for name in WANT_CONSTS:
            exec(compile(segments[name], str(APP), "exec"), ns)
        for name in WANT_FNS:
            exec(compile(segments[name], str(APP), "exec"), ns)
        exec(compile(speak_src, str(APP) + ":_speak", "exec"), ns)
        self.speak = ns["_speak"]

    def _pipeline(self, session, client_id, gen, stop_event, tts, opts):
        chunks = list(gen)
        self.records.append(("pipeline", chunks))
        return "SPOKE:" + "|".join(str(c) for c in chunks)


def gen_of(items, on_first=None, on_close=None):
    """测试生成器：首段产出与关闭都可挂钩。"""
    def _g():
        try:
            for i, it in enumerate(items):
                if i == 0 and on_first is not None:
                    on_first()
                yield it
        finally:
            if on_close is not None:
                on_close()
    return _g()


PASS, FAIL = 0, 0

def check(name, ok, detail=""):
    global PASS, FAIL
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    PASS, FAIL = PASS + (1 if ok else 0), FAIL + (0 if ok else 1)


def wait_for(fn, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if fn():
            return True
        time.sleep(0.05)
    return fn()


# ---- A1：播报期短证据不掐——3 chunk 窗口下 _speak 挂起等待 ---------------
s = FakeSession(delivery_speech_active=True, gen_in_progress=True,
                playback_end=time.time() + 8.0,
                bargein_times=[time.time()] * 3)
h = Harness(s)
result = {}
t = threading.Thread(target=lambda: result.setdefault(
    "rc", h.speak(gen_of(["第一段", "第二段"]), stop_event=threading.Event())))
t.start()
ok_no_kill = not wait_for(
    lambda: "interrupt" in [c for c in s.calls], timeout=0.8)
check("A1 short evidence (3 chunks) does not kill delivery",
      ok_no_kill, str(s.calls))
# 证据攒够（5 chunk 且首块 ≥1s 前）→ 放行接管话权
s.bargein_times[:] = [time.time() - 2.0] * 5
t.join(timeout=6)
check("A1b sustained evidence (5 chunks/2s) proceeds",
      t.is_alive() is False and result.get("rc") == "SPOKE:第一段|第二段",
      str(result))
check("A1c took the floor exactly once",
      s.calls.count("interrupt") == 1 and s.calls.count("reset") == 1,
      str(s.calls))

# ---- A2：播报已死（session.stop_event 已置——真打断）立即接替 -------------
s = FakeSession(delivery_speech_active=True, gen_in_progress=True,
                playback_end=time.time() + 8.0, bargein_times=[])
s.stop_event.set()
h = Harness(s)
rc = h.speak(gen_of(["接替"]), stop_event=threading.Event())
check("A2 dead delivery (stop_event set) proceeds immediately",
      rc == "SPOKE:接替" and s.calls.count("interrupt") == 1, str(s.calls))

# ---- A3：播报已播完（_reply_in_flight 转 False）放行 ----------------------
s = FakeSession(delivery_speech_active=True, gen_in_progress=False,
                playback_end=time.time() - 1.0)
h = Harness(s)
rc = h.speak(gen_of(["接替"]), stop_event=threading.Event())
check("A3 finished delivery (not in flight) proceeds",
      rc == "SPOKE:接替", str(s.calls))

# ---- A4：本话轮被取代（stop_event 参数已置）安静退出，不 interrupt -------
s = FakeSession(delivery_speech_active=True, gen_in_progress=True,
                playback_end=time.time() + 8.0, bargein_times=[])
h = Harness(s)
dead = threading.Event()
dead.set()
rc = h.speak(gen_of(["过时回复"]), stop_event=dead)
check("A4 superseded turn (stop_event arg) exits silently",
      rc is None and s.calls == [], str(s.calls))

# ---- B1：非播报期原语义——安静收听等播完 ------------------------------------
s = FakeSession(playback_end=time.time() + 0.5,
                last_voice_ts=1.0, assistant_audible_ts=2.0)
h = Harness(s)
t0 = time.time()
rc = h.speak(gen_of(["让路回执"]), stop_event=threading.Event())
waited = time.time() - t0
check("B1 quiet listener waits out non-delivery playback",
      rc == "SPOKE:让路回执" and 0.3 < waited < 3.0, f"waited={waited:.2f}s")

# ---- B2：非播报期原语义——用户插话过立即接替 -------------------------------
s = FakeSession(playback_end=time.time() + 8.0,
                last_voice_ts=time.time(), assistant_audible_ts=1.0)
h = Harness(s)
t0 = time.time()
rc = h.speak(gen_of(["接替"]), stop_event=threading.Event())
check("B2 user spoke since playback → immediate takeover",
      rc == "SPOKE:接替" and time.time() - t0 < 1.0, "")

# ---- C1：预取期话权易主（interrupt+reset 换新事件）→ 本回复作废 ---------
s = FakeSession()
h = Harness(s)


def _take_floor():
    # 别的路在预取期间完成了一轮接管（如交付泵/回执上口）
    s.interrupt()
    s.reset_interrupt()


rc = h.speak(gen_of(["被抢走的话"], on_first=_take_floor),
             stop_event=threading.Event())
# calls 里的 interrupt/reset 是 _take_floor 自己做的接管动作；_speak 自己
# 的接管必带 cut→commit 前奏——没有它们即证明 _speak 未再抢话权
check("C1 floor takeover during prefetch invalidates the reply",
      rc is None and s.calls == ["interrupt", "reset"], str(s.calls))

# ---- C1b：预取期原地 set（barge-in confirm 杀掉播报）≠ 话权易主 --------
# 等在死播报后面的接替者不能被旧 set 误杀——回复照常上口（A2 同源）
s = FakeSession()
h = Harness(s)


def _kill_delivery():
    # 信号路 confirm：只 set，未 reset——没有新持有者诞生
    s.stop_event.set()


rc = h.speak(gen_of(["接替播报的话"], on_first=_kill_delivery),
             stop_event=threading.Event())
check("C1b in-place set (delivery killed, no reset) does NOT drop reply",
      rc == "SPOKE:接替播报的话", str(s.calls))

# ---- C2：预取顺序——首段到手才接管话权 --------------------------------------
s = FakeSession()
h = Harness(s)
order = []


def _mark2():
    order.append("first")


orig_interrupt = s.interrupt
s.interrupt = lambda: (order.append("interrupt"), orig_interrupt())
rc = h.speak(gen_of(["首段", "次段"], on_first=_mark2),
             stop_event=threading.Event())
check("C2 first segment precedes floor takeover (no silent gap)",
      rc == "SPOKE:首段|次段"
      and order == ["first", "interrupt"]
      and s.calls == ["cut", "commit", "interrupt", "reset"],
      f"order={order} calls={s.calls}")

# ---- C3：空流不接管话权（三态语义保留）-------------------------------------
s = FakeSession()
h = Harness(s)
closed = []
rc = h.speak(gen_of([], on_close=lambda: closed.append(1)),
             stop_event=threading.Event())
check("C3 empty stream exits before taking the floor",
      rc is None and s.calls == [] and closed == [1], str(s.calls))

# ---- C4：预取期本话轮被取代 → 作废并关生成器 ------------------------------
s = FakeSession()
h = Harness(s)
dead = threading.Event()
closed = []


def _mark4():
    dead.set()      # 新话轮置 chitchat_stop


rc = h.speak(gen_of(["过时"], on_first=_mark4, on_close=lambda: closed.append(1)),
             stop_event=dead)
check("C4 superseded during prefetch → drop + close generator",
      rc is None and s.calls == [] and closed == [1], str(s.calls))

print()
if FAIL:
    print(f"{FAIL} FAILED, {PASS} passed")
    sys.exit(1)
print(f"ALL {PASS} PASS")
