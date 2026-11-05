# -*- coding: utf-8 -*-
"""Offline replay of the 2026-11-05 swallowing incident (session 9e727c1f)
through the NEW _pvad_selftalk_text physics window, plus true-echo cases
that must still drop. Run: soulx/venv/bin/python -I scripts/selftalk_replay.py
"""
import sys, time, types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "repo" / "dialogue_system"))

# pvad_gate imports emitter/settings — stub the heavy deps it pulls? It
# imports emitter (which imports...?) and settings only. Try direct import.
import pvad_gate

def mk_session():
    s = types.SimpleNamespace()
    s.playback_end = 0.0
    s.interruption_time = None
    s.playback_run_serial = 0
    s.playback_run_first_ts = None
    s.playback_run_total = 0.0
    s.assistant_chunks = []
    return s

class Clock:
    def __init__(self, t): self.t = t
    def set(self, t): self.t = t

clk = Clock(0.0)
pvad_gate.time = types.SimpleNamespace(time=lambda: clk.t)

def start_run(s, first, chunks):
    """chunks: list of (text, cum_end_seconds) of a continuous run."""
    s.playback_run_serial += 1
    s.playback_run_first_ts = first
    s.playback_run_total = chunks[-1][1]
    s.playback_end = first + chunks[-1][1] + 0.5
    for text, cum in chunks:
        s.assistant_chunks.append(
            {"text": text, "run": s.playback_run_serial, "cum": cum,
             "ts": first + cum - 1.0})

T = pvad_gate._SELF_TALK_ECHO_TAIL_S
results = []

def check(name, expect_drop, s, t, q):
    clk.set(t)
    got = pvad_gate._pvad_selftalk_text(s, q)
    ok = got == expect_drop
    results.append((ok, name, expect_drop, got))
    print(("PASS" if ok else "FAIL"),
          f"{name}: expect={'DROP' if expect_drop else 'PASS'}, got={'DROP' if got else 'PASS'}")

# ---- incident timeline (2026-11-05, wall-clock as h*3600+m*60+s offsets)
H = 10 * 3600 + 24 * 60  # 10:24:00
S1 = "让我查一下星期天的天气，稍等一下哦～"   # spoken 10:24:16, ~4s

# run A: first audio 10:24:16, one chunk, 4s
sA = mk_session()
start_run(sA, H + 16, [(S1, 4.0)])

# 1. 10:24:57 user correction quoting A (the killer false drop)
check("也不是星期天的天气 (40s stale source)", False, sA, H + 57, "也不是星期天的天气。")
# 2. 10:25:08 nothing playing
check("你听到了吗 (nothing playing)", False, sA, H + 68, "你听到了吗？")
# 3. 10:25:16 long correction (passed even before the fix)
check("我让你查的是北京的天气 (long correction)", False, sA, H + 76, "嗯，我让你查的是北京的天气，不是星期天的天气。")

# run B (apology) starts 10:25:17.5, two chunks ~9s total; A's chunks
# carry old ts
sB = mk_session()
start_run(sB, H + 16, [(S1, 4.0)])
clk.set(H + 77.5)
start_run(sB, H + 77.5, [
    ("啊，抱歉抱歉，刚才好像没记清你具体要查的是哪天。", 4.5),
    ("我现在马上去查一下北京最新的天气情况，马上就好！", 9.0),
])
# 4/5. 10:25:25 + 10:25:27 while apology plays — old s1 now excluded
check("就查今天的 (during apology, old src stale)", False, sB, H + 85, "就查今天的。")
check("查今天的北京天气 (during apology)", False, sB, H + 87, "查今天的北京天气。")
# 6. 10:25:38 apology finished
check("就是查今天的北京的天气 (after apology)", False, sB, H + 98, "就是查今天的北京的天气。")

# ---- true echoes must STILL drop
# E1. verbatim echo of what is playing right now
check("E1 verbatim echo of playing chunk", True, sA, H + 18, "星期天的天气")
# E2. contiguous garbled prefix of the playing chunk
check("E2 garbled prefix echo", True, sA, H + 18, "让我查一下星期天")
# E3. historical echo shape: 怎么又喊我婉儿呀 (one garbled char, contiguous)
sC = mk_session()
start_run(sC, H, [("怎么又喊我“我儿”呀，是不是喊累了", 3.0)])
check("E3 怎么又喊我婉儿呀 (garbled echo)", True, sC, H + 2, "怎么又喊我婉儿呀")
# E4. echo just after a confirmed barge-in (recent_interrupt path)
sD = mk_session()
start_run(sD, H, [("今天我们聊聊周末的计划吧，你想做什么呢", 4.0)])
clk.set(H + 2.0)
sD.playback_end = 0.0        # interrupt() zeroed it
sD.interruption_time = H + 2.0
check("E4 echo right after barge-in", True, sD, H + 3.0, "聊聊周末的计划")
# E5. echo of the chunk CURRENTLY at the playback head of a long run
sE = mk_session()
start_run(sE, H, [(f"这是第{i}段内容举例说明", 4.0 * i) for i in range(1, 6)])
check("E5 echo of playing chunk in long run", True, sE, H + 18, "这是第5段内容")
# E6. KNOWN BOUNDARY: user quoting the sentence FRAME of the playing chunk
# ("也不是第1段内容" while "这是第4段内容举例说明" plays) — matched frame
# words are contiguous in the source (span 0.83), indistinguishable from a
# garbled echo by any text rule. Documented as boundary, expected to drop.
check("E6 frame-quote boundary (documented)", True, sE, H + 18, "也不是第1段内容")

fails = [r for r in results if not r[0]]
print(f"\n{len(results)-len(fails)}/{len(results)} checks passed")
sys.exit(1 if fails else 0)
