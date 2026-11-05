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
code = ast.get_source_segment(src, fn)


class _Logger:
    def info(self, *a, **k):
        pass
    warning = error = debug = info


messages = []          # (role, text) committed to LLM history
events = []            # websocket emits


ns = {
    "time": time,
    "logger": _Logger(),
    "emit_to_room": lambda cid, ev, data=None: events.append((ev, data)),
    "llm": types.SimpleNamespace(
        add_message=lambda cid, role, msg: messages.append((role, msg))
    ),
}
exec(compile(code, str(APP), "exec"), ns)
pipeline = ns["_run_speech_pipeline"]


class _AudioTTS:
    def __init__(self):
        self.written = 0
        self.breaks = 0
        self.cuts = []

    def run_break(self):
        self.breaks += 1

    def write(self, b):
        self.written += len(b)

    def mark_cut(self, t, first):
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
    s.audio_tts = _AudioTTS()
    return s


class MockTTS:
    """Two 0.5s wav chunks per segment (24000 B = 0.5s @ 24k int16)."""

    def synthesize(self, chunk, streaming=False, options=None):
        for _ in range(2):
            yield b"\x00" * 24000


class StopMidSynthTTS:
    """Sets the stop during the FIRST segment's synthesis — the classic
    mid-audio barge-in, already handled by _speak_chunk's wav-loop check."""

    def __init__(self, stop):
        self.stop = stop

    def synthesize(self, chunk, streaming=False, options=None):
        for i in range(2):
            if i == 1:
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


# 1. THE BUG: stop inside the generator, zero segments -> must be False
ret, s = run(gen_zero_yield_on_stop)
check("零产出+stop -> interrupted(False)", False, ret,
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
#    segment and is (still) reported as fully spoken — unchanged behavior
ret, s = run(gen_stop_between_segments)
check("[边界]段间 stop 截断 -> 仍 True(既定边界)", True, ret,
      s.pending_message == "北京今天天气晴朗，")

# 5. stop during wav synthesis (classic mid-audio barge-in) -> False
s = mk_session()
stop = threading.Event()
messages.clear()
ret = pipeline(s, 1, gen_two_segments(stop), stop,
               StopMidSynthTTS(stop), {})
check("音频合成中 stop -> False(原有路径)", False, ret,
      not messages, s.pending_message is None, len(s.audio_tts.cuts) == 1)

# 6. grace path: zero-yield stop during the hold -> False
ret, s = run(gen_zero_yield_on_stop, grace_until=time.time() + 30)
check("宽限窗内零产出+stop -> False", False, ret, not messages)

# 7. grace path: short reply finishes inside the window, no resumed
#    speech -> flushed and fully spoken -> True
ret, s = run(gen_two_segments, grace_until=time.time() + 0.25)
check("宽限窗内完成并冲刷 -> True", True, ret,
      s.pending_message == "北京今天天气晴朗，气温在五到十七度之间。",
      len(s.assistant_chunks) == 2)

print()
if all(results):
    print(f"ALL {len(results)} PASS")
else:
    print(f"{results.count(False)}/{len(results)} FAIL")
    raise SystemExit(1)
