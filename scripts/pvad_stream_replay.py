# -*- coding: utf-8 -*-
"""Offline regression for the 2026-11-05 pvad integration fixes
(docs/PVAD背景视频失效调查报告-20261105.md → 修复实施):

  A. Pre-ASR stream gate (pvad_stream.PvadStreamGate) state machine:
     hysteresis (2x sub-(threshold-0.20) to suppress, >=threshold to
     release, dead band holds), fail-open (unenrolled / unscoreable /
     service-blind), shadow never suppresses, pvad off = zero scoring.
  B. Gap-tolerant barge-in accumulation (pvad_gate.pvad_streak_append):
     nonidle chunks survive idle gaps <= 3 s, evicted by age/cap, (ts,
     wav) snapshot format.
  C. High-margin re-enroll/bank gate (pvad_gate): marginal verified
     passes (threshold <= cos < threshold+0.15) no longer bank/re-enroll.
  D. Real-audio replay of the background-video session (a5f92f1d mic
     sidecar) through the ACTUAL _evaluate + _pvad_ring_tail + in-process
     ECAPA: pre-video user speech never muted, video-only stretches
     suppressed, the loud user turn released promptly.

Run (parts A-C on any python with numpy; part D needs the ECAPA venv):
  full_duplex/venv-frc/bin/python -I scripts/pvad_stream_replay.py
"""
import sys
import threading
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "repo" / "dialogue_system"))

try:
    import omegaconf  # noqa: F401  (web-app dep pulled in via settings)
except ImportError:
    # Part D runs under the ECAPA venv (full_duplex/venv-frc), which lacks
    # the web app's client stack. settings.py needs only the
    # QwenLLM_stream.DEFAULT_IDENTITY default — stub the module.
    _llm = types.ModuleType("clients.llm_client")
    _llm.QwenLLM_stream = types.SimpleNamespace(DEFAULT_IDENTITY="")
    _pkg = types.ModuleType("clients")
    _pkg.__path__ = []
    sys.modules["clients"] = _pkg
    sys.modules["clients.llm_client"] = _llm

import pvad_gate
import pvad_stream
from pvad_stream import (PvadStreamGate, _STREAM_MARGIN, _STREAM_WINDOW_S,
                         _SUPPRESS_LOW_RUN)

SR = 16000
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(("PASS" if ok else "FAIL"), name, detail)


# ---------------------------------------------------------------- helpers
class FakePvad:
    def __init__(self, enrolled=True, scores=None):
        self.enrolled_flag = enrolled
        self.scores = list(scores or [])
        self.calls = 0

    @property
    def enrolled(self):
        return self.enrolled_flag

    def score(self, wav):
        self.calls += 1
        return self.scores.pop(0) if self.scores else None


def mk_session(settings=None, pvad=None, ring_s=30.0):
    s = types.SimpleNamespace()
    s.client_id = "replay000"
    s.settings = {"pvad_mode": "enforce", "pvad_threshold": 0.5,
                  "pvad_stream_gate": True}
    s.settings.update(settings or {})
    s.pvad = pvad or FakePvad()
    s.ring_lock = threading.Lock()
    s.mic_ring = []
    s.playback_end = 0.0
    s.pvad_turn_verified = False
    s.pvad_turn_cos = None
    s.pvad_speech_bank = None
    s.pvad_stream_gate = None
    s.voxlock_isolator = None
    s.bargein_audio = []
    return s


def ring_speak(session, t0, dur, rms=0.05):
    """Append dur seconds of 'speech' (>= the 0.005 RMS floor) to the ring
    at synthetic time t0, 160 ms chunks — enough for _pvad_ring_tail to
    return a >=1 s window."""
    n = int(dur / 0.16)
    rng = np.random.default_rng(int(t0 * 1000) % (2**31))
    for i in range(n):
        chunk = (rng.standard_normal(2560) * rms).astype(np.float32)
        session.mic_ring.append((t0 + i * 0.16, chunk))
    # 30 s ring retention, like pvad_ring_append
    max_chunks = int(30.0 / 0.16) + 8
    if len(session.mic_ring) > max_chunks:
        del session.mic_ring[: len(session.mic_ring) - max_chunks]
    return t0 + n * 0.16


# ================================================================ A. gate
def part_a():
    print("\n=== A. stream-gate state machine ===")

    # A1: unenrolled -> never suppress, and never even scores
    s = mk_session(pvad=FakePvad(enrolled=False, scores=[0.0, 0.0]))
    g = PvadStreamGate(s, start_thread=False)
    ring_speak(s, 100.0, 3.0)
    g._evaluate(); g._evaluate()
    check("A1 unenrolled never suppresses (and never scores)",
          not g.suppressing() and s.pvad.calls == 0)

    # A2: 2 consecutive low scores -> suppressed on the 2nd
    s = mk_session(pvad=FakePvad(scores=[0.10, 0.12]))
    g = PvadStreamGate(s, start_thread=False)
    ring_speak(s, 100.0, 3.0)
    g._evaluate()
    ok1 = not g.suppressing()
    g._evaluate()
    check("A2 two sub-T_low scores suppress", ok1 and g.suppressing())

    # A3: one high score releases immediately
    s.pvad.scores = [0.62]
    g._evaluate()
    check("A3 one >=threshold score releases", not g.suppressing())

    # A4: dead band holds state, and breaks the low-run
    s = mk_session(pvad=FakePvad(scores=[0.10, 0.12, 0.38, 0.10, 0.10]))
    g = PvadStreamGate(s, start_thread=False)
    ring_speak(s, 100.0, 3.0)
    g._evaluate(); g._evaluate()
    sup = g.suppressing()
    g._evaluate()                       # 0.38 dead band: holds
    hold = g.suppressing() == sup
    g._evaluate()                       # 0.10: first low after a dead-band
    g._evaluate()                       # 0.10: second low -> suppress
    check("A4 dead band holds state but breaks the low-run",
          hold and g.suppressing())

    # A5: shadow mode never suppresses
    s = mk_session(settings={"pvad_mode": "shadow"},
                   pvad=FakePvad(scores=[0.05, 0.05, 0.05, 0.05]))
    g = PvadStreamGate(s, start_thread=False)
    ring_speak(s, 100.0, 3.0)
    for _ in range(4):
        g._evaluate()
    check("A5 shadow evaluates but never suppresses",
          not g.suppressing() and s.pvad.calls >= 2)

    # A6: silent ring -> unscoreable -> released
    s = mk_session(pvad=FakePvad(scores=[0.10, 0.10, 0.10]))
    g = PvadStreamGate(s, start_thread=False)
    ring_speak(s, 100.0, 3.0)
    g._evaluate(); g._evaluate()
    sup = g.suppressing()
    s.mic_ring = []                     # all-quiet: window below RMS floor
    g._evaluate()
    check("A6 unscoreable (silent) window releases", sup and not g.suppressing())

    # A7: pvad off -> released without any scoring
    s = mk_session(settings={"pvad_mode": "off"},
                   pvad=FakePvad(scores=[0.0, 0.0]))
    g = PvadStreamGate(s, start_thread=False)
    ring_speak(s, 100.0, 3.0)
    g._evaluate(); g._evaluate()
    check("A7 pvad off = zero scoring, never suppresses",
          not g.suppressing() and s.pvad.calls == 0)

    # A8: stream-gate switch off -> app never creates the gate; the
    # filter is a passthrough when no gate exists
    s = mk_session(settings={"pvad_stream_gate": False})
    wav = np.ones(2560, dtype=np.float32) * 0.1
    out = pvad_stream.stream_gate_filter(s, wav)
    check("A8 no gate object -> passthrough",
          out is wav or np.array_equal(out, wav))

    # A9: suppression actually silences the chunk
    s = mk_session(pvad=FakePvad(scores=[0.10, 0.10]))
    g = PvadStreamGate(s, start_thread=False)
    ring_speak(s, 100.0, 3.0)
    g._evaluate(); g._evaluate()
    s.pvad_stream_gate = g
    out = pvad_stream.stream_gate_filter(s, wav)
    check("A9 suppressing -> zeros to ASR",
          g.suppressing() and out.shape == wav.shape and not out.any())


# ================================================== B. bargein accumulation
def part_b():
    print("\n=== B. gap-tolerant bargein accumulation ===")
    chunk = lambda: np.ones(2560, dtype=np.float32) * 0.05

    # B1: the incident shape — user's 停停停 interleaved with quiet video
    # chunks rewritten to idle. Old code: any idle chunk cleared the
    # audio; new code must accumulate >=0.9 s across <=3 s gaps.
    s = mk_session()
    t = 100.0
    # 4 nonidle + 2 idle + 4 nonidle + 2 idle + 4 nonidle (~2.9 s span)
    for burst in range(3):
        for _ in range(4):
            pvad_gate.pvad_streak_append(s, t, chunk()); t += 0.16
        t += 0.32  # idle gap (video quiet patch)
    total = sum(c.shape[0] for _, c in s.bargein_audio) / SR
    check("B1 interleaved burst keeps 1.92 s of evidence",
          total >= 0.9 and abs(total - 1.92) < 0.01,
          f"({total:.2f}s)")

    # B2: eviction by age — evidence older than _PVAD_BARGEIN_RETAIN_S
    # must not vouch for a later barge-in
    s = mk_session()
    pvad_gate.pvad_streak_append(s, 100.0, chunk())
    pvad_gate.pvad_streak_append(s, 103.1, chunk())   # >3 s later
    ages = [ts for ts, _ in s.bargein_audio]
    check("B2 stale evidence (>3 s) evicted",
          len(s.bargein_audio) == 1 and ages == [103.1])

    # B3: 16-chunk audio cap
    s = mk_session()
    for i in range(20):
        pvad_gate.pvad_streak_append(s, 100.0 + i * 0.05, chunk())
    check("B3 capped at 16 chunks (~2.5 s)",
          len(s.bargein_audio) == 16)

    # B4: handle_bargein snapshot consumes the (ts, wav) format
    s = mk_session()
    pvad_gate.pvad_streak_append(s, 100.0, chunk())
    pvad_gate.pvad_streak_append(s, 100.2, chunk())
    pvad_gate.pvad_streak_append(s, 100.4, chunk())
    pvad_gate.pvad_streak_append(s, 100.6, chunk())
    pvad_gate.pvad_streak_append(s, 100.8, chunk())
    pvad_gate.pvad_streak_append(s, 101.0, chunk())
    snap = np.concatenate([c for _, c in s.bargein_audio])
    check("B4 snapshot is contiguous float32 audio",
          snap.shape[0] == 6 * 2560 and snap.dtype == np.float32)


# ================================================== C. high-margin enroll
def part_c():
    print("\n=== C. high-margin re-enroll/bank ===")
    threshold, margin = 0.5, pvad_gate._PVAD_REENROLL_MARGIN

    # C1: marginal verified pass (0.55 < 0.65) must NOT bank
    s = mk_session()
    s.pvad.enrolled_flag = True
    s.pvad_turn_verified = True
    s.pvad_turn_cos = 0.55
    ring_speak(s, 100.0, 3.0)
    pvad_gate.pvad_bank_turn_audio(s)
    banked = s.pvad_speech_bank is not None
    check("C1 marginal verified pass does not bank", not banked)

    # C2: high-margin pass (0.75) banks
    s.pvad_turn_cos = 0.75
    pvad_gate.pvad_bank_turn_audio(s)
    check("C2 high-margin pass banks",
          s.pvad_speech_bank is not None
          and s.pvad_speech_bank.shape[0] >= SR)

    # C3: unenrolled (bootstrap) still banks on any pass
    s2 = mk_session()
    s2.pvad.enrolled_flag = False
    s2.pvad_turn_verified = False      # fail-open pass while unenrolled
    s2.pvad_turn_cos = None
    ring_speak(s2, 100.0, 3.0)
    pvad_gate.pvad_bank_turn_audio(s2)
    check("C3 bootstrap (unenrolled) pass still banks",
          s2.pvad_speech_bank is not None)

    # C4: maybe_auto_enroll skips marginal, allows high-margin
    # (pvad_try_enroll spawns a thread -> monkeypatch to record calls)
    calls = []
    orig = pvad_gate.pvad_try_enroll
    pvad_gate.pvad_try_enroll = lambda sess, reason: calls.append(reason)
    try:
        s.pvad_turn_cos = 0.60
        pvad_gate.pvad_maybe_auto_enroll(s)
        s.pvad_turn_cos = 0.80
        pvad_gate.pvad_maybe_auto_enroll(s)
    finally:
        pvad_gate.pvad_try_enroll = orig
    check("C4 re-enroll requires margin", calls == ["auto"],
          f"(calls={calls})")


# ============================================== D. real-audio ECAPA replay
def part_d():
    print("\n=== D. real-audio replay (session a5f92f1d, background video) ===")
    logdir = Path(__file__).resolve().parents[1] / "logs" / "audio"
    sid = "a5f92f1d-d6bc-471c-bd6a-cba3b8b13d53"
    pcm = logdir / f"{sid}.mic.pcm"
    meta = logdir / f"{sid}.mic.json"
    if not pcm.exists() or not meta.exists():
        print("SKIP (no sidecar for", sid, ")")
        return
    try:
        import torch
        # 2026-11-09: in-repo processor (services/pvad_proc.py) replaces the
        # livekit plugin import — same _spk_extractor.get_embedding surface
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services"))
        from pvad_proc import PVADProcessor
    except ImportError as e:
        print("SKIP (no ECAPA venv:", e, ")")
        return

    import json
    mic = np.fromfile(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    runs = json.loads(meta.read_text())["runs"]
    t0 = float(runs[0][1])           # epoch of PCM sample 0 (19:54:58.7 local)

    # wall-clock helper: (h, m, s) of the session's local day -> epoch
    import datetime
    _day = datetime.datetime.fromtimestamp(t0).replace(
        hour=0, minute=0, second=0, microsecond=0).timestamp()

    def epoch(h, m, s):
        return _day + h * 3600 + m * 60 + s

    def seg(h, m, s, d):
        a = epoch(h, m, s) - t0
        return mic[int(a * SR):int((a + d) * SR)]

    proc = PVADProcessor()

    def embed(wav):
        w = np.ascontiguousarray(wav[-80000:])    # pvad_server ENROLL_WINDOW
        emb = proc._spk_extractor.get_embedding(
            torch.from_numpy(w).unsqueeze(0))
        emb = np.asarray(emb, dtype=np.float64).reshape(-1)
        n = np.linalg.norm(emb)
        return emb / n if n > 0 else emb

    # clean reference: the enrolled user's pre-video speech (the E2
    # enrollment window of the report — last clean 5 s: 19:55:14-19:55:19)
    ref = embed(seg(19, 55, 13, 6.0))

    class EcapaPvad:
        enrolled_flag = False

        @property
        def enrolled(self):
            return self.enrolled_flag

        def score(self, wav):
            if wav.shape[0] < 8000:
                return None
            return float(np.dot(embed(wav), ref))

    s = mk_session(pvad=EcapaPvad())
    g = PvadStreamGate(s, start_thread=False)
    timeline = []            # (t_epoch, suppressed)
    chunk_n = mic.shape[0] // 2560
    t = t0
    eval_every = max(1, int(0.4 / 0.16))    # ~0.4 s cadence like the thread
    ENROLL_AT = epoch(19, 55, 19)           # E2 enrolled ~19:55:18.6
    since_eval = 0
    for i in range(chunk_n):
        chunk = mic[i * 2560: (i + 1) * 2560]
        # receive-loop order: ring append (original audio) THEN gate filter
        s.mic_ring.append((t, chunk))
        max_chunks = int(30.0 / 0.16) + 8
        if len(s.mic_ring) > max_chunks:
            del s.mic_ring[: len(s.mic_ring) - max_chunks]
        if not s.pvad.enrolled_flag and t >= ENROLL_AT:
            s.pvad.enrolled_flag = True
        since_eval += 1
        if since_eval >= eval_every:
            since_eval = 0
            g._evaluate()
            timeline.append((t, g.suppressing()))
        t += 0.16

    def cov(h1, m1, s1, h2, m2, s2):
        a, b = epoch(h1, m1, s1), epoch(h2, m2, s2)
        rows = [sup for tt, sup in timeline if a <= tt < b]
        return (len(rows), (sum(rows) / len(rows)) if rows else 0.0)

    # D1: actual user speech pre-video (E1+E2 utterances, rms>=0.02 span
    # 19:55:08-19:55:17) is never muted — the no-collateral clause. The
    # quiet spans around it MAY be suppressed (ambient noise, cos~0.05);
    # suppressing silence is harmless — no speech to lose — and the gate
    # reopens the moment the user speaks.
    n, c = cov(19, 55, 8, 19, 55, 17)
    check("D1 pre-video user speech never suppressed",
          n > 0 and c == 0.0, f"(cov {c:.0%} over {n} rounds)")

    # D2: sustained video-only stretch (19:55:29-19:55:43, continuous
    # video speech per the report) engages suppression and holds it.
    # Brief releases during the video's own pauses are the fail-open
    # design (unscoreable -> release); the gate re-arms in ~0.8 s.
    n, c = cov(19, 55, 29, 19, 55, 42)
    check("D2 video-only stretch suppressed >=60%", c >= 0.6,
          f"(cov {c:.0%} over {n} rounds)")

    # D3: the loud user turn (19:55:57 「喂，你从一数到十...」, cos
    # 0.51-0.55 = dead band holding release) is never muted through its
    # full span
    n, c = cov(19, 55, 56, 19, 56, 1)
    check("D3 loud user turn stays released", n > 0 and c == 0.0,
          f"(cov {c:.0%} over {n} rounds)")

    # D4: overall video-period coverage (report only)
    n, c = cov(19, 55, 25, 19, 57, 38)
    print(f"     video period 19:55:25-19:57:38: suppression coverage "
          f"{c:.0%} over {n} rounds (leaks = re-arm delay after each "
          f"video pause, by design fail-open)")


# ============================================== E. voxlock shadow hook
def part_e():
    print("\n=== E. VoxLock isolation shadow hook (2026-11-10) ===")
    import voxlock_isolate

    wav = np.ones(19200, dtype=np.float32) * 0.05

    class FakeIso:
        def __init__(self, tail=None):
            self._tail = tail

        def isolated_tail(self, seconds):
            return self._tail

    # E1: switch off (default) / no isolator -> no isolated scoring
    s = mk_session()
    check("E1 speaker_sep off -> shadow returns None",
          not voxlock_isolate.voxlock_enabled(s)
          and voxlock_isolate.sep_shadow_round(s, wav) is None)

    # E2: switch on + isolator with ring -> scores what ASR heard
    # (the isolated tail), exactly one pvad call
    s = mk_session(settings={"speaker_sep": True},
                   pvad=FakePvad(scores=[0.6]))
    s.voxlock_isolator = FakeIso(tail=wav)
    out = voxlock_isolate.sep_shadow_round(s, wav)
    check("E2 shadow scores the isolated tail",
          out == {"sep_cos": 0.6} and s.pvad.calls == 1, f"({out})")

    # E3: empty ring / pvad blind -> None, gate state machine unaffected
    s3 = mk_session(pvad=FakePvad(scores=[0.10, 0.10, 0.10, 0.10]))
    s3.voxlock_isolator = FakeIso(tail=None)
    out = voxlock_isolate.sep_shadow_round(s3, wav)
    g = PvadStreamGate(s3, start_thread=False)
    ring_speak(s3, 100.0, 3.0)
    g._evaluate(); g._evaluate()
    check("E3 no isolated audio -> shadow None, gate still suppresses",
          out is None and g.suppressing())

    # E4: gate round with sep on reports sep_cos but decides on raw cos
    # (scores per round: raw 0.10, isolated 0.90)
    s4 = mk_session(settings={"speaker_sep": True},
                    pvad=FakePvad(scores=[0.10, 0.90, 0.10, 0.90]))
    s4.voxlock_isolator = FakeIso(tail=wav)
    g4 = PvadStreamGate(s4, start_thread=False)
    ring_speak(s4, 100.0, 3.0)
    g4._evaluate(); g4._evaluate()
    check("E4 gate suppresses on raw cos despite high sep cos",
          g4.suppressing() and g4.sep_status() == {"sep_cos": 0.9})


if __name__ == "__main__":
    part_a()
    part_b()
    part_c()
    part_d()
    part_e()
    n_ok = sum(results)
    print(f"\n{'=' * 50}\n{len(results) - n_ok} failed / {len(results)} checks")
    sys.exit(0 if n_ok == len(results) else 1)
