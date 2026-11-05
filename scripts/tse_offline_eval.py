# -*- coding: utf-8 -*-
"""Offline target-speaker-extraction validation (2026-11-06, 任务2).

Question: does MossFormer2_SS_16K two-speaker separation + ECAPA stream
selection WIDEN the speaker-verification bands on the background-video
forensics session (a5f92f1d)?  Band overlap (video-only −0.06–0.39 vs
user-under-video 0.11–0.40, report docs/PVAD背景视频失效调查报告
-20261105.md §5) is exactly why the stream gate needs hysteresis; if
separation moves video-only windows far below and user windows far above
the threshold, a plain threshold gate becomes safe again.

Two-process protocol (numpy/torch isolation discipline):
  step 1 (this script, venv-tse):  MossFormer2_SS_16K over the session,
      writes /tmp/tse_eval/ss_s{1,2}_<slice>.wav
  step 2 (venv-frc, ECAPA):  scripts/tse_score_bands.py scores the
      original vs separated streams against the clean reference.

Usage:
  MODELSCOPE_CACHE=/mnt/disk3/krisl/soulx/modelscope_cache \
  /mnt/disk3/krisl/soulx/venv-tse/bin/python -I scripts/tse_offline_eval.py [--full]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

SR = 16000
LOGDIR = Path("/home/research/krisl/soulx/logs/audio")
SID = "a5f92f1d-d6bc-471c-bd6a-cba3b8b13d53"
OUT = Path("/tmp/tse_eval")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="process the whole session (default: 40s probe slice)")
    args = ap.parse_args()

    from clearvoice import ClearVoice
    import soundfile as sf

    mic = np.fromfile(LOGDIR / f"{SID}.mic.pcm", dtype=np.int16
                      ).astype(np.float32) / 32768.0
    meta = json.loads((LOGDIR / f"{SID}.mic.json").read_text())
    t0 = float(meta["runs"][0][1])

    if args.full:
        wav = mic
    else:
        # probe slice: video-only stretch 19:55:25-19:56:05 (video voice,
        # then the loud user turn at ~19:55:57) — contains both regimes
        import datetime
        day = datetime.datetime.fromtimestamp(t0).replace(
            hour=0, minute=0, second=0, microsecond=0).timestamp()
        a = day + 19 * 3600 + 55 * 60 + 25
        i0 = int((a - t0) * SR)
        wav = mic[i0:i0 + 40 * SR]

    print(f"input: {wav.shape[0]/SR:.1f}s", flush=True)
    cv = ClearVoice(task="speech_separation", model_names=["MossFormer2_SS_16K"])
    x = np.reshape(wav, [1, -1]).astype(np.float32)
    t1 = time.time()
    out = cv(x, False)                     # [spk, batch, length]
    dt = time.time() - t1
    print(f"separation took {dt:.1f}s ({wav.shape[0]/SR/dt:.2f}x RT, CPU)",
          flush=True)

    OUT.mkdir(exist_ok=True, parents=True)
    tag = "full" if args.full else "probe"
    sf.write(OUT / f"orig_{tag}.wav", wav, SR)
    for spk in range(out.shape[0]):
        sf.write(OUT / f"ss_s{spk+1}_{tag}.wav", out[spk, 0, :], SR)
        print(f"wrote ss_s{spk+1}_{tag}.wav "
              f"rms={float(np.sqrt((out[spk,0,:]**2).mean())):.4f}")
    # wall-clock mapping for the scorer
    (OUT / f"meta_{tag}.json").write_text(json.dumps({
        "tag": tag, "t_start_abs": (t0 if args.full else a),
    }))


if __name__ == "__main__":
    sys.exit(main())
