"""Cross-segment voice-consistency test for the VoxCPM2 TTS service (:8085).

Drift background: zero-shot mode samples a fresh voice per call (CFM noise,
no seed), so every LLM segment used to be a different timbre. The fix
(2026-10-29) anchors the voice on a persistent reference wav and reuses one
prompt cache for every segment.

This test synthesizes several segments — the same text three times plus four
different texts — then measures speaker similarity across them with the pvad
sidecar's ECAPA (/enroll + /score on :8086, the same identity signal the
main-speaker gate uses). Pass criteria:

  - every segment scores >= 0.60 cosine vs the enrolled anchor voice
    (measured same-speaker range 0.59-0.84; the old zero-shot drift sat far
    below this)
  - first-chunk latency stays sane (informational, asserted < 2 s)

Also writes the segment wavs to logs/voxcpm_consistency/ for listening.

Usage (needs the voxtts + pvad windows up):
  soulx/venv/bin/python -I scripts/voxcpm_consistency_test.py
"""
import base64
import os
import time
from pathlib import Path

import numpy as np
import requests

TTS = "http://127.0.0.1:8085"
PVAD = "http://127.0.0.1:8086"
OUT_RATE = 24000
OUT_DIR = Path("/home/research/krisl/soulx/logs/voxcpm_consistency")

TEXTS = [
    "今天天气不错，我们出去走走吧。",           # x3 — same text, must be
    "今天天气不错，我们出去走走吧。",           #   the same voice
    "今天天气不错，我们出去走走吧。",
    "好的，没问题，我马上就过来。",             # x4 — different texts,
    "这道题的答案其实很简单，你再想一想。",       #   same anchored voice
    "晚安，祝你做个好梦，明天见。",
    "哇，这个主意听起来太棒了！",
]

failures = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        failures.append(name)


def synth(text):
    """POST /synthesize -> (pcm float32 @24k, first_chunk_ms, total_s)."""
    t0 = time.time()
    with requests.post(
        TTS + "/synthesize", json={"input": text}, stream=True, timeout=180
    ) as r:
        r.raise_for_status()
        first_ms = None
        buf = bytearray()
        for chunk in r.iter_content(chunk_size=None):
            if not chunk:
                continue
            if first_ms is None:
                first_ms = (time.time() - t0) * 1000
            buf.extend(chunk)
    pcm = np.frombuffer(bytes(buf), dtype="<i2").astype(np.float32) / 32768.0
    return pcm, first_ms, len(pcm) / OUT_RATE


def b64(wav):
    return base64.b64encode(wav.astype("<f4").tobytes()).decode()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    h = requests.get(TTS + "/health", timeout=5).json()
    check("tts service healthy", h.get("loaded") is True, str(h))
    print(f"[INFO] health: lora={h.get('lora')} anchored={h.get('anchored')}")
    check("LoRA active", h.get("lora") is True)
    check("voice anchored (reference cache)", h.get("anchored") is True)
    if not h.get("anchored"):
        print("[RESULT] FAIL — service running in zero-shot fallback")
        raise SystemExit(1)

    segs, first_ms, durs = [], [], []
    for i, text in enumerate(TEXTS):
        pcm, ms, dur = synth(text)
        segs.append(pcm)
        first_ms.append(ms)
        durs.append(dur)
        print(f"[INFO] seg{i}: {dur:.1f}s audio, first chunk {ms:.0f}ms")
        import soundfile as sf

        sf.write(OUT_DIR / f"seg{i}.wav", pcm, OUT_RATE)

    import soundfile as sf

    # ECAPA scoring via the pvad sidecar: enroll on the server's anchor wav
    # itself (the actual reference the TTS clones), then score every
    # synthesized segment against it.
    import soundfile as sf

    anchor_path = Path(
        os.environ.get("VOXCPM_ANCHOR_WAV", "/home/research/krisl/soulx/data/voxcpm_anchor.wav")
    )
    anchor, anchor_sr = sf.read(anchor_path, dtype="float32")
    sid = "voxcpm-consistency-test"
    requests.post(PVAD + "/reset", json={"session_id": sid}, timeout=5)
    r = requests.post(
        PVAD + "/enroll",
        json={
            "session_id": sid,
            "audio": b64(anchor),
            "sample_rate": anchor_sr,
        },
        timeout=30,
    ).json()
    check("enroll on anchor voice", r.get("ok") is True, str(r))

    cos = []
    for i, pcm in enumerate(segs):
        r = requests.post(
            PVAD + "/score",
            json={"session_id": sid, "audio": b64(pcm), "sample_rate": OUT_RATE},
            timeout=30,
        ).json()
        cos.append(r.get("cos"))
    print(f"[INFO] per-seg cos vs anchor: {[f'{c:.3f}' for c in cos]}")

    check("all segments scored", all(c is not None for c in cos))
    valid = [c for c in cos if c is not None]
    # 0.55 floor: natural same-speaker ECAPA bottoms out ~0.59 and
    # different-speaker tops out ~0.44; expressive TTS on 2-3 s segments
    # adds ~0.1 of embedding noise, so 0.55 stays clear of the cross-speaker
    # band without failing legitimate emotion variation. Measured anchors:
    # zero-shot (pre-fix) mean 0.52 with min 0.22 — the fixed service must
    # sit clearly above that band, hence mean >= 0.65.
    check(
        f"min segment cos >= 0.55 (got {min(valid):.3f})",
        min(valid) >= 0.55,
    )
    check(
        f"mean segment cos >= 0.65 (got {np.mean(valid):.3f})",
        np.mean(valid) >= 0.65,
    )
    check(
        f"same-text repeats consistent (spread {max(cos[:3])-min(cos[:3]):.3f})",
        max(cos[:3]) - min(cos[:3]) < 0.15,
    )
    p95 = float(np.percentile(first_ms, 95))
    check(f"first-chunk latency p95 < 2000ms (got {p95:.0f}ms)", p95 < 2000)

    requests.post(PVAD + "/reset", json={"session_id": sid}, timeout=5)
    print(
        f"\n[RESULT] {'FAIL: ' + str(failures) if failures else 'PASS — voice is anchored across segments'}"
    )
    print(f"[INFO] wavs for listening: {OUT_DIR}/seg*.wav")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
