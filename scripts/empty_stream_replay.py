# -*- coding: utf-8 -*-
"""Offline regression for the empty-stream false-success bug (2026-11-05,
session 8bb3abba 问题二): a barge-in stop landing inside the LLM generator
BEFORE its first yield exits _run_speech_pipeline's for-loop with zero
iterations — every interrupted=True assignment lives inside the loop body,
so the pipeline reported True ("fully spoken") for a delivery that never
said a word, and the arbiter marked the task delivered instead of retrying
(13s of silence in the incident).

Method: AST-extract the real _run_speech_pipeline source from app.py and
exec it against mocks (no heavy imports — app.py pulls the whole service
stack at module level). If the function's signature changes, this script
fails loudly.

Run: /home/research/krisl/soulx/venv/bin/python -I scripts/empty_stream_replay.py
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
fn = next(
    (n for n in tree.body
     if isinstance(n, ast.FunctionDef) and n.name == "_run_speech_pipeline"),
    None,
)
assert fn is not None, "_run_speech_pipeline not found in app.py"
fn_deliver = next(
    (n for n in tree.body
     if isinstance(n, ast.FunctionDef) and n.name == "_agent_deliver_entry"),
    None,
)
assert fn_deliver is not None, "_agent_deliver_entry not found in app.py"
fn_begin = next(
    (n for n in tree.body
     if isinstance(n, ast.FunctionDef) and n.name == "_begin_generation"),
    None,
)
assert fn_begin is not None, "_begin_generation not found in app.py"
code = ast.get_source_segment(src, fn)


class _Logger:
    def info(self, *a, **k):
        pass
    warning = error = debug = info


messages = []          # (role, text) committed to LLM history
events = []            # websocket emits


ns = {
    "time": time,
    # _run_speech_pipeline 打断分支带线程标识暂存"用户实际听到部分"
    # （渐进式披露，2026-11-09）——抽取执行需要 threading 在命名空间里
    "threading": __import__("threading"),
    "logger": _Logger(),
    "emit_to_room": lambda cid, ev, data=None: events.append((ev, data)),
    "llm": types.SimpleNamespace(
        add_message=lambda cid, role, msg: messages.append((role, msg))
    ),
    # deliver-path stubs (overwritten per test below)
    "_commit_pending_message": lambda session, cid: None,
    "TTS_CLIENTS": {},
    "DEFAULT_SETTINGS": {"tts_engine": "mock"},
}
exec(compile(code, str(APP), "exec"), ns)
exec(compile(ast.get_source_segment(src, fn_begin), str(APP), "exec"), ns)
exec(compile(ast.get_source_segment(src, fn_deliver), str(APP), "exec"), ns)
pipeline = ns["_run_speech_pipeline"]
deliver = ns["_agent_deliver_entry"]


class _AudioTTS:
    def __init__(self):
        self.written = 0
        self.breaks = 0
        self.cuts = []

    def run_break(self, since=None):
        self.breaks += 1

    def write(self, b):
        self.written += len(b)

    def mark_cut(self, t, since=None):
        self.cuts.append(t)


def mk_session():
    s = types.SimpleNamespace()
    s.lock = threading.Lock()
    s.pending_message = None
    s.pending_audio_duration = 0.0
    s.pending_start_time = None
    s.interruption_time = None
    s.last_voice_ts = 0.0
    s.assistant_chunks = []
    s.assistant_audible_ts = 0.0
    s.playback_end = 0.0
    s.playback_run_serial = 0
    s.playback_run_first_ts = None
    s.playback_run_total = 0.0
    # 慢脑播报打断保护（2026-11-07 起交付路径会触碰这两个字段）
    import collections as _c
    s.bargein_times = _c.deque()
    s.delivery_speech_active = False
    s.audio_tts = _AudioTTS()
    # app.py 的 ChatSession.cut_pending_audio：forensic cut 的会话侧入口
    # （pending 状态还在时把未播尾巴切在打断点）。桩里直接转发给
    # audio_tts.mark_cut，行为与真实现一致（pending 为空时 no-op）。
    def _cut_pending_audio(t=None):
        with s.lock:
            since = s.pending_start_time
            if s.pending_message is None or not since:
                return
            if t is None:
                t = s.interruption_time or time.time()
        s.audio_tts.mark_cut(t, since)

    s.cut_pending_audio = _cut_pending_audio
    return s


class MockTTS:
    """Two 0.5s wav chunks per segment (24000 B = 0.5s @ 24k int16)."""

    def synthesize(self, chunk, streaming=False, options=None):
        for _ in range(2):
            yield b"\x00" * 24000


class StopMidSynthTTS:
    """Sets the stop during the FIRST segment's synthesis, after one wav
    chunk was already emitted — audible interruption (classic mid-audio
    barge-in), handled by _speak_chunk's wav-loop check."""

    def __init__(self, stop):
        self.stop = stop

    def synthesize(self, chunk, streaming=False, options=None):
        for i in range(2):
            if i == 1:
                self.stop.set()
            yield b"\x00" * 24000


class StopBeforeFirstWavTTS:
    """Sets the stop before yielding ANY wav chunk — interrupted during
    the first synthesis, nothing audible yet (silent kill)."""

    def __init__(self, stop):
        self.stop = stop

    def synthesize(self, chunk, streaming=False, options=None):
        self.stop.set()
        yield b"\x00" * 24000


# --- generator shapes -------------------------------------------------
# The incident: first next() finds the stop already set (it landed while
# the generator was blocked inside the streaming HTTP) and returns without
# ever yielding.
def gen_zero_yield_on_stop(stop):
    stop.set()
    return
    yield  # noqa: unreachable — makes this a generator function


def gen_two_segments(stop):
    yield "北京今天天气晴朗，"
    yield "气温在五到十七度之间。"


def gen_full_then_stop(stop):
    # all content delivered, THEN the stop arrives (user starts talking
    # right as the report ends) — must still count as fully spoken
    yield "北京今天天气晴朗，"
    yield "气温在五到十七度之间。"
    stop.set()
    return
    yield  # noqa: unreachable


def gen_stop_between_segments(stop):
    # stop lands while the generator is between two yields: it exits
    # without yielding the rest. KNOWN BOUNDARY (documented, unchanged
    # behavior): the first segment was fully spoken, the result note is
    # already in history — currently reported as fully spoken.
    yield "北京今天天气晴朗，"
    stop.set()
    return
    yield  # noqa: unreachable


results = []


def check(name, expect, got, *conds):
    ok = (expect == got) and all(conds)
    results.append(ok)
    print(("PASS" if ok else "FAIL"),
          f"{name}: expected={expect}, got={got}" + ("" if ok else " (violated)"))


def run(gen_factory, tts=None, grace_until=None):
    s = mk_session()
    stop = threading.Event()
    messages.clear()
    events.clear()
    ret = pipeline(
        s, 1, gen_factory(stop), stop, tts or MockTTS(), {},
        grace_until=grace_until,
    )
    return ret, s


# 1. THE BUG: stop inside the generator, zero segments -> silent kill (None)
ret, s = run(gen_zero_yield_on_stop)
check("零产出+stop -> 静默掐死(None)", None, ret,
      not messages, s.pending_message is None, not s.assistant_chunks)

# 2. normal two-segment reply, no stop -> True, full text pending
ret, s = run(gen_two_segments)
check("正常两段播完 -> True", True, ret,
      s.pending_message == "北京今天天气晴朗，气温在五到十七度之间。",
      len(s.assistant_chunks) == 2, not messages)

# 3. full reply spoken, stop only AFTER the last segment -> still True
#    (guards against the naive `not (interrupted or stop.is_set())` fix,
#    which would re-deliver an already-complete report)
ret, s = run(gen_full_then_stop)
check("播完之后才 stop -> 仍 True(不重投)", True, ret,
      s.pending_message == "北京今天天气晴朗，气温在五到十七度之间。")

# 4. KNOWN BOUNDARY: stop between yields truncates at the last completed
#    (audible) segment and is (still) reported as fully spoken — the
#    between-yields window is far smaller than the first-segment window
ret, s = run(gen_stop_between_segments)
check("[边界]段间 stop 截断 -> 仍 True(既定边界)", True, ret,
      s.pending_message == "北京今天天气晴朗，")

# 5. stop during wav synthesis, one chunk already audible -> False
#    (audible rejection: consumes a delivery attempt)
s = mk_session()
stop = threading.Event()
messages.clear()
ret = pipeline(s, 1, gen_two_segments(stop), stop,
               StopMidSynthTTS(stop), {})
check("音频已出后合成中 stop -> False(算拒绝)", False, ret,
      not messages, s.pending_message is None, len(s.audio_tts.cuts) == 1)

# 6. stop before the FIRST wav chunk of the first segment -> None
#    (nothing audible: silent kill, must not consume the budget)
s = mk_session()
stop = threading.Event()
messages.clear()
ret = pipeline(s, 1, gen_two_segments(stop), stop,
               StopBeforeFirstWavTTS(stop), {})
check("首包合成前 stop -> None(静默掐死)", None, ret, not messages)

# 7. grace path: zero-yield stop during the hold -> None
ret, s = run(gen_zero_yield_on_stop, grace_until=time.time() + 30)
check("宽限窗内零产出+stop -> None", None, ret, not messages)

# 8. grace path: user resumes during the hold (premature drop) -> None
s = mk_session()
stop = threading.Event()
messages.clear()


def gen_resume_during_hold(stop, sess):
    sess.last_voice_ts = 5.0  # sustained voice after hold start (ref 0.0)
    yield "这句不该被说出来。"


ret = pipeline(s, 1, gen_resume_during_hold(stop, s), stop,
               MockTTS(), {}, grace_until=time.time() + 30)
check("宽限窗内复语放弃(premature) -> None", None, ret,
      not messages, not s.assistant_chunks)

# 9. grace path: short reply finishes inside the window, no resumed
#    speech -> flushed and fully spoken -> True
ret, s = run(gen_two_segments, grace_until=time.time() + 0.25)
check("宽限窗内完成并冲刷 -> True", True, ret,
      s.pending_message == "北京今天天气晴朗，气温在五到十七度之间。",
      len(s.assistant_chunks) == 2)

# ---- part 2: pop_message (note retraction) ---------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "repo" / "dialogue_system"))
import clients.llm_client as llm_mod  # noqa: E402

NOTE = "[系统提示：后台任务 T1 已完成，以下是真实结果……]"


def fresh_llm():
    m = llm_mod.QwenLLM_stream()
    m.sessions = {}
    return m


# 10. note alone at the tail -> popped cleanly
m = fresh_llm()
m.add_message(1, "assistant", "我去查一下。")
m.add_message(1, "user", NOTE)
ok = m.pop_message(1, "user", NOTE) and m.get_session(1) == [
    {"role": "assistant", "content": "我去查一下。"}]
results.append(ok)
print(("PASS" if ok else "FAIL"), "note 独占尾部 -> 整条弹出")

# 11. barge-in user text merged AFTER the note (prefix strip)
m = fresh_llm()
m.add_message(1, "assistant", "我去查一下。")
m.add_message(1, "user", NOTE)
m.add_message(1, "user", "嗯，好的好的。")   # add_message merges same-role
ok = m.pop_message(1, "user", NOTE) and m.get_session(1) == [
    {"role": "assistant", "content": "我去查一下。"},
    {"role": "user", "content": "嗯，好的好的。"}]
results.append(ok)
print(("PASS" if ok else "FAIL"), "note 与后续 user 合并 -> 前缀剥离")

# 12. note merged AFTER a previous user message (suffix strip)
m = fresh_llm()
m.add_message(1, "user", "查一下北京天气。")
m.add_message(1, "user", NOTE)
ok = m.pop_message(1, "user", NOTE) and m.get_session(1) == [
    {"role": "user", "content": "查一下北京天气。"}]
results.append(ok)
print(("PASS" if ok else "FAIL"), "note 与前序 user 合并 -> 后缀剥离")

# 13. tail is an unrelated message -> no-op, returns False
m = fresh_llm()
m.add_message(1, "assistant", " unrelated ")
ok = (not m.pop_message(1, "user", NOTE)) and len(m.get_session(1)) == 1
results.append(ok)
print(("PASS" if ok else "FAIL"), "尾部无关消息 -> 不动返回 False")

# ---- part 3: _agent_deliver_entry end-to-end -------------------------
class DeliverSession(types.SimpleNamespace):
    def interrupt(self):
        pass

    def reset_interrupt(self):
        self.stop_event.clear()


def mk_deliver_session():
    s = DeliverSession(**vars(mk_session()))
    s.stop_event = threading.Event()
    s.settings = {"tts_engine": "mock", "tts_cfg": 2.0, "tts_timesteps": 10}
    s.gen_serial = 0
    s.gen_in_progress = False
    return s


def run_deliver(gen_factory):
    """Drive the real _agent_deliver_entry with a real QwenLLM_stream
    (only generate_with_history is patched). Returns (ret, llm)."""
    s = mk_deliver_session()
    m = fresh_llm()
    m.generate_with_history = lambda cid, stop_event=None, **k: gen_factory(stop_event)
    ns["llm"] = m
    ns["TTS_CLIENTS"] = {"mock": MockTTS()}
    events.clear()
    entry = {"task_id": "T1", "goal": "查北京天气", "result": "晴，17度",
             "error": None, "attempts": 0}
    ret = deliver(s, 1, entry)
    return ret, m


# 14. silent-killed delivery -> None AND the note is retracted from
#     history (the retry re-injects a fresh one)
ret, m = run_deliver(gen_zero_yield_on_stop)
hist = m.get_session(1)
note_gone = not any("T1" in (msg.get("content") or "") for msg in hist)
check("投递静默掐死 -> None 且 note 收回", None, ret, note_gone)

# 15. fully-spoken delivery -> True AND the note stays in history
ret, m = run_deliver(gen_two_segments)
hist = m.get_session(1)
note_kept = any("T1" in (msg.get("content") or "") for msg in hist)
check("投递完整播出 -> True 且 note 保留", True, ret, note_kept)

print()
if all(results):
    print(f"ALL {len(results)} PASS")
else:
    print(f"{results.count(False)}/{len(results)} FAIL")
    raise SystemExit(1)
