# -*- coding: utf-8 -*-
"""Offline regression for voxlock_isolate (2026-11-10 说话人分离方案
替换：MossFormer2 sidecar → VoxLock 进程内隔离).

Covers the guards that keep the isolation off the critical path:
  A. switch off = inert filter, no threads
  B. sample conservation — identity isolation forwards every sample
     exactly once; a slow worker's fail-open sends debit their
     duplicates; nothing is ever replayed or stretched
  C. failure modes — push error degrades to permanent bypass;
     mid-session toggle-off stops feeding
  D. seeding — creation seeds from the pvad bank when enrolled;
     live() otherwise; seed() is one-way
  E. the isolated ring feeds isolated_tail for the shadow scorer

Run: /home/research/krisl/soulx/venv/bin/python -I \
     scripts/voxlock_isolate_test.py
"""
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "repo" / "dialogue_system"))

import voxlock_isolate as VI  # noqa: E402


# ---- stub the voxlock package before any isolator spawns its worker ----
class FakeStream:
    """Configurable fake: identity isolation, optional output lag."""
    mode = "identity"          # identity | lag3 | boom

    def __init__(self, sample):
        self.sample = sample
        self.locked = sample is not None
        self._pending = []     # held-back outputs (lag mode)
        self._skip = 3         # pushes to swallow before releasing

    def push(self, chunk):
        cls = type(self)
        if cls.mode == "boom":
            raise RuntimeError("synthetic push failure")
        if cls.mode == "lag3" and self._skip > 0:
            self._skip -= 1
            self._pending.append(chunk)
            return np.zeros(0, np.float32)
        if self._pending:
            self._pending.append(chunk)
            out = np.concatenate(self._pending)
            self._pending = []
            return out
        return chunk


def _fake_voxlock():
    m = types.ModuleType("voxlock")

    class FakeVoxLock:
        def __init__(self, threads=1):
            pass

        def stream(self, sample):
            return FakeStream(sample)

        def live(self):
            return FakeStream(None)

    m.VoxLock = FakeVoxLock
    return m


sys.modules["voxlock"] = _fake_voxlock()
FakeStream.mode = "identity"

PASS, FAIL = 0, 0


def check(name, ok, detail=""):
    global PASS, FAIL
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    PASS, FAIL = PASS + (1 if ok else 0), FAIL + (0 if ok else 1)


class FakePvad:
    def __init__(self, enrolled=False):
        self.enrolled = enrolled


def mk_session(settings=None, enrolled=False, bank=None):
    s = types.SimpleNamespace()
    s.client_id = "vl-test000"
    s.settings = dict(settings or {})
    s.pvad = FakePvad(enrolled)
    s.ring_lock = threading.Lock()
    s.mic_ring = []
    s.pvad_speech_bank = bank
    s.voxlock_isolator = None
    return s


def feed(s, n_chunks=20, chunk=512):
    """Push n_chunks through the receive-loop hook; return (sent, rng)."""
    rng = np.random.default_rng(7)
    sent = []
    for i in range(n_chunks):
        wav = rng.standard_normal(chunk).astype(np.float32) * 0.1
        out = VI.voxlock_asr_filter(s, wav)
        sent.append((out, wav))
    return sent


# ---- A: off = inert -----------------------------------------------------
s = mk_session()
wav = np.ones(512, np.float32) * 0.1
out = VI.voxlock_asr_filter(s, wav)
check("A1 switch off -> filter is passthrough",
      out is wav and s.voxlock_isolator is None)

# ---- B1: identity isolation forwards every sample exactly once ----------
FakeStream.mode = "identity"
s = mk_session(settings={"speaker_sep": True})
VI.ensure_isolator(s)
check("B1a unenrolled -> live() (no seed)", not s.voxlock_isolator._seeded)
sent = feed(s, 40)
n_in = sum(w.shape[0] for _, w in sent)
n_out = sum(o.shape[0] for o, _ in sent)
ok_content = all(np.array_equal(o, w) for o, w in sent[1:])  # 1-chunk lag max
check("B1b identity: every sample sent exactly once",
      n_in == n_out, f"in={n_in} out={n_out}")
check("B1c identity: content unchanged (lag <= 1 chunk)", ok_content)
s.voxlock_isolator.stop()
time.sleep(0.05)

# ---- B2: lagging worker -> fail-open + debit, conservation intact -------
FakeStream.mode = "lag3"
s = mk_session(settings={"speaker_sep": True})
VI.ensure_isolator(s)
sent = feed(s, 30)
n_in = sum(w.shape[0] for _, w in sent)
n_out = sum(o.shape[0] for o, _ in sent)
# first 3 rounds fail open (original audio), the 4th drains the 4-chunk
# burst minus the 3-chunk debit
failopens = [i for i, (o, w) in enumerate(sent)
             if o is w]
check("B2a lag: first rounds fail open to original", failopens[:3] == [0, 1, 2],
      f"fail-open rounds={failopens[:6]}")
check("B2b lag: sample conservation holds",
      n_in == n_out, f"in={n_in} out={n_out}")
check("B2c lag: no stretch/replay (out <= in + chunk)",
      n_out <= n_in + 512)
s.voxlock_isolator.stop()
time.sleep(0.05)

# ---- C1: worker error -> permanent bypass --------------------------------
FakeStream.mode = "boom"
s = mk_session(settings={"speaker_sep": True})
VI.ensure_isolator(s)
deadline = time.time() + 2
while not s.voxlock_isolator._degraded and time.time() < deadline:
    time.sleep(0.02)
sent = feed(s, 5)
check("C1 push failure -> degraded bypass to original",
      s.voxlock_isolator._degraded
      and all(o is w for o, w in sent))

# ---- C2: mid-session toggle off ------------------------------------------
FakeStream.mode = "identity"
s = mk_session(settings={"speaker_sep": True})
VI.ensure_isolator(s)
feed(s, 5)
s.settings["speaker_sep"] = False
wav = np.ones(512, np.float32)
out = VI.voxlock_asr_filter(s, wav)
check("C2 toggle off -> stops feeding, passthrough",
      out is wav and s.voxlock_isolator is None)

# ---- D1: creation seeds from the bank when enrolled ----------------------
bank = (np.random.default_rng(3)
        .standard_normal(int(4.0 * 16000)).astype(np.float32) * 0.2)
s = mk_session(settings={"speaker_sep": True}, enrolled=True, bank=bank)
VI.ensure_isolator(s)
time.sleep(0.2)
check("D1 enrolled -> seeded from bank at creation",
      s.voxlock_isolator._seeded)
s.voxlock_isolator.stop()

# ---- D2: seed() is one-way -----------------------------------------------
s = mk_session(settings={"speaker_sep": True})
VI.ensure_isolator(s)
time.sleep(0.05)
ok_seed = s.voxlock_isolator.seed(bank)
ok_reseed = s.voxlock_isolator.seed(bank)
time.sleep(0.3)
check("D2 seed applies once, re-seed refused",
      ok_seed and not ok_reseed and s.voxlock_isolator._seeded)
s.voxlock_isolator.stop()

# ---- D3: short sample refused --------------------------------------------
s = mk_session(settings={"speaker_sep": True})
VI.ensure_isolator(s)
time.sleep(0.05)
check("D3 sample < 3 s refused",
      not s.voxlock_isolator.seed(bank[:int(2.9 * 16000)]))
s.voxlock_isolator.stop()

# ---- E: isolated ring feeds isolated_tail --------------------------------
FakeStream.mode = "identity"
s = mk_session(settings={"speaker_sep": True})
VI.ensure_isolator(s)
sent = feed(s, int(2.0 / 0.032) + 10)   # > 2 s of audio
time.sleep(0.1)
tail = s.voxlock_isolator.isolated_tail(1.2)
check("E1 isolated_tail returns the last 1.2 s",
      tail is not None and tail.shape[0] == int(1.2 * 16000))
check("E2 tail matches the tail of what was sent",
      np.allclose(tail, np.concatenate([o for o, _ in sent])[-tail.shape[0]:],
                  atol=1e-6))
empty = s.voxlock_isolator.isolated_tail(60.0)
check("E3 longer-than-ring tail -> None", empty is None)
s.voxlock_isolator.stop()

# ---- shadow round contract ------------------------------------------------
class FakeScorer:
    enrolled = True

    def __init__(self):
        self.calls = 0

    def score(self, wav):
        self.calls += 1
        return 0.42


s = mk_session(settings={"speaker_sep": True})
s.pvad = FakeScorer()
VI.ensure_isolator(s)
feed(s, int(1.5 / 0.032))
time.sleep(0.1)
out = VI.sep_shadow_round(s, np.zeros(16000, np.float32))
check("F1 shadow scores the isolated tail (never the raw wav arg)",
      out == {"sep_cos": 0.42} and s.pvad.calls == 1, f"({out})")
s.voxlock_isolator.stop()

s2 = mk_session()   # off
check("F2 shadow off -> None",
      VI.sep_shadow_round(s2, np.zeros(16000, np.float32)) is None)

print()
if FAIL:
    print(f"{FAIL} FAILED, {PASS} passed")
    sys.exit(1)
print(f"ALL {PASS} PASS")
