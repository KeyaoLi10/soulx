# -*- coding: utf-8 -*-
"""Step 2 of the TSE offline validation (run with the ECAPA venv):

  /home/research/krisl/full_duplex/venv-frc/bin/python -I \
      scripts/tse_score_bands.py [probe|full]

Scores the ORIGINAL mic audio and both MossFormer2_SS output streams
(written by scripts/tse_offline_eval.py to /tmp/tse_eval/) against the
clean enrolled-user reference (19:55:14-19:55:19 of session a5f92f1d),
then reports the ECAPA cosine bands over ground-truth spans:

  USER   actual user speech (should score HIGH on the selected stream)
  VIDEO  video-only stretches    (should score LOW on the selected stream)

Stream selection is per-window: pick whichever of s1/s2 scores higher vs
the reference — exactly what a runtime "separate + pick" gate would do.
The question is whether the selected stream's USER band and VIDEO band
separate cleanly (they fully overlap on the original).
"""
import json
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/research/krisl/soulx/repo/dialogue_system")
try:
    import omegaconf  # noqa: F401
except ImportError:
    _llm = types.ModuleType("clients.llm_client")
    _llm.QwenLLM_stream = types.SimpleNamespace(SYSTEM_PROMPT="")
    _pkg = types.ModuleType("clients"); _pkg.__path__ = []
    sys.modules["clients"] = _pkg
    sys.modules["clients.llm_client"] = _llm

import torch
from livekit.plugins.fireredchat_pvad.vad import VadProcessor

SR = 16000
LOGDIR = Path("/home/research/krisl/soulx/logs/audio")
SID = "a5f92f1d-d6bc-471c-bd6a-cba3b8b13d53"
OUT = Path("/tmp/tse_eval")

# ground-truth spans (wall clock, from the investigation report + the
# 2026-11-05 rms/cos traces)
USER_SPANS = [(19 * 3600 + 55 * 60 + 8, 19 * 3600 + 55 * 60 + 17),
              (19 * 3600 + 55 * 60 + 56, 19 * 3600 + 56 * 60 + 1)]
VIDEO_SPANS = [(19 * 3600 + 55 * 60 + 29, 19 * 3600 + 55 * 60 + 42)]


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "probe"
    import soundfile as sf

    mic = np.fromfile(LOGDIR / f"{SID}.mic.pcm", dtype=np.int16
                      ).astype(np.float32) / 32768.0
    mic_t0 = float(json.loads(
        (LOGDIR / f"{SID}.mic.json").read_text())["runs"][0][1])
    m = json.loads((OUT / f"meta_{tag}.json").read_text())
    slice_t0 = float(m["t_start_abs"])   # absolute epoch of output sample 0

    orig = sf.read(OUT / f"orig_{tag}.wav", dtype="float32")[0]
    s1 = sf.read(OUT / f"ss_s1_{tag}.wav", dtype="float32")[0]
    s2 = sf.read(OUT / f"ss_s2_{tag}.wav", dtype="float32")[0]

    proc = VadProcessor()

    def embed(wav):
        w = np.ascontiguousarray(wav[-80000:])
        e = np.asarray(
            proc._spk_extractor.get_embedding(
                torch.from_numpy(w).unsqueeze(0)),
            dtype=np.float64).reshape(-1)
        n = np.linalg.norm(e)
        return e / n if n > 0 else e

    import datetime
    day = datetime.datetime.fromtimestamp(mic_t0).replace(
        hour=0, minute=0, second=0, microsecond=0).timestamp()

    def epoch(h, mi, s):
        return day + h * 3600 + mi * 60 + s

    # spans are written as (h*3600+m*60+s) of the session's local day —
    # convert to epoch once
    user_spans = [(lo + day, hi + day) for lo, hi in USER_SPANS]
    video_spans = [(lo + day, hi + day) for lo, hi in VIDEO_SPANS]

    # clean reference: always from the ORIGINAL mic pcm (the probe slice
    # does not contain the pre-video user speech)
    ref = embed(mic[int((epoch(19, 55, 13) - mic_t0) * SR):
                    int((epoch(19, 55, 19) - mic_t0) * SR)])

    n = min(len(orig), len(s1), len(s2))
    orig, s1, s2 = orig[:n], s1[:n], s2[:n]

    WIN, STEP = 1.2, 0.4
    rows = []      # (span_label, t_abs, cos_orig, cos_sel, which)
    for i in range(0, int((n - WIN * SR) / (STEP * SR))):
        a = int(i * STEP * SR)
        b = a + int(WIN * SR)
        t_abs = slice_t0 + a / SR
        label = None
        for lo, hi in user_spans:
            if lo <= t_abs < hi:
                label = "USER"
        if label is None:
            for lo, hi in video_spans:
                if lo <= t_abs < hi:
                    label = "VIDEO"
        if label is None:
            continue
        c_orig = float(np.dot(embed(orig[a:b]), ref))
        c1 = float(np.dot(embed(s1[a:b]), ref))
        c2 = float(np.dot(embed(s2[a:b]), ref))
        which, c_sel = (1, c1) if c1 >= c2 else (2, c2)
        rows.append((label, t_abs, c_orig, c_sel, which))

    def band(label, idx):
        vals = [r[idx] for r in rows if r[0] == label]
        return (min(vals), max(vals),
                sum(vals) / len(vals)) if vals else (None,) * 3

    print(f"\n=== {tag}: {len(rows)} scored windows "
          f"(USER={sum(1 for r in rows if r[0]=='USER')}, "
          f"VIDEO={sum(1 for r in rows if r[0]=='VIDEO')}) ===")
    for label in ("USER", "VIDEO"):
        lo_o, hi_o, mu_o = band(label, 2)
        lo_s, hi_s, mu_s = band(label, 3)
        if lo_o is None:
            print(f"{label:5s} (no windows in this slice)")
            continue
        print(f"{label:5s} orig cos [{lo_o:+.3f}, {hi_o:+.3f}] mean {mu_o:+.3f}")
        print(f"{label:5s} sel  cos [{lo_s:+.3f}, {hi_s:+.3f}] mean {mu_s:+.3f}")
    ul, uh, _ = band("USER", 3)
    vl, vh, _ = band("VIDEO", 3)
    if ul is not None and vh is not None:
        print(f"\nselected-stream band gap: USER_min − VIDEO_max = "
              f"{ul - vh:+.3f}  (>0 ⇒ a plain threshold works again)")
    flips = sum(1 for i in range(1, len(rows))
                if rows[i][0] == rows[i-1][0] and rows[i][4] != rows[i-1][4])
    print(f"stream selection flips within same span: {flips}")


if __name__ == "__main__":
    main()
