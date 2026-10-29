"""Calibrate the ECAPA barge-in/turn gate windows empirically.

Question: with enrollment done the way the web app does it (speech-chunk
concatenation, like the bank), how does cosine separate
  - same speaker at 0.5 / 1 / 2 / 3 s natural windows (barge-in streaks)
  - a different speaker
  - the assistant voice itself (echo)
under natural vs concatenated enrollment? Prints a table; no assertions —
this is a measurement, run it to pick thresholds.

Usage: soulx/venv/bin/python -I scripts/ecapa_window_calib.py
"""
import base64
import wave
from pathlib import Path

import numpy as np
import requests

PVAD = "http://127.0.0.1:8086"
SR = 16000
DATA = Path("/home/research/krisl/soulx/data/eval_inputs")
SAME = DATA / "eotbench_zh/zh__2284/input.wav"          # enrollment speaker
DIFF = DATA / "eotbench_zh/zh__2289/input.wav"          # another speaker
ANCHOR = Path("/home/research/krisl/soulx/data/voxcpm_anchor.wav")  # assistant

SID = "calib"


def load(path):
    with wave.open(str(path), "rb") as w:
        x = (
            np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
            .astype(np.float32)
            / 32768.0
        )
    if x.shape[0] and abs(x.shape[0] / SR - w.getnframes() / w.getframerate()) > 1:
        pass  # anchor is 48k
    return x


def load48(path):
    import soundfile as sf

    x, sr = sf.read(path, dtype="float32")
    if sr != SR:
        import soxr

        x = soxr.resample(x, sr, SR)
    return x


def speech_chunks(wav, cap_s=None):
    """Mimic _pvad_ring_speech: keep 160ms chunks with rms>=0.005, concat."""
    n = (wav.shape[0] // 2560) * 2560
    chunks = wav[:n].reshape(-1, 2560)
    rms = np.sqrt(np.mean(chunks * chunks, axis=1))
    keep = chunks[rms >= 0.005]
    if cap_s:
        keep = keep[-int(cap_s / 0.16):]
    return np.concatenate(keep)


def b64(wav):
    return base64.b64encode(wav.astype("<f4").tobytes()).decode()


def enroll(wav):
    requests.post(PVAD + "/reset", json={"session_id": SID}, timeout=5)
    r = requests.post(
        PVAD + "/enroll",
        json={"session_id": SID, "audio": b64(wav), "sample_rate": SR},
        timeout=30,
    ).json()
    assert r.get("ok"), r


def score(wav):
    r = requests.post(
        PVAD + "/score",
        json={"session_id": SID, "audio": b64(wav), "sample_rate": SR},
        timeout=30,
    ).json()
    return r.get("cos")


def seg(wav, start_s, dur_s):
    return wav[int(start_s * SR): int((start_s + dur_s) * SR)]


def main():
    same = load(SAME)
    diff = load(DIFF)
    anchor = load48(ANCHOR)

    # find speech-rich regions in each file (for fair sampling)
    def speech_offsets(wav, count, dur):
        n = (wav.shape[0] // 2560) * 2560
        chunks = wav[:n].reshape(-1, 2560)
        rms = np.sqrt(np.mean(chunks * chunks, axis=1))
        idx = np.where(rms >= 0.03)[0]
        # pick `count` offsets spaced apart, each with `dur` of speech ahead
        offs = []
        step = max(1, len(idx) // count)
        for i in idx[::step][:count]:
            offs.append(i * 0.16)
        return offs

    enrollments = {
        "natural-6s (21.0s window)": seg(same, 21.0, 6.0),
        "concat-6s (bank-style)": speech_chunks(same, 6.0),
    }
    durs = [0.5, 1.0, 2.0, 3.0]

    for name, emb in enrollments.items():
        print(f"\n=== enrollment: {name} ({emb.shape[0]/SR:.1f}s) ===")
        enroll(emb)
        for dur in durs:
            cos_same = [score(seg(same, o, dur)) for o in speech_offsets(same, 4, dur)]
            cos_diff = [score(seg(diff, o, dur)) for o in speech_offsets(diff, 3, dur)]
            cos_echo = [score(seg(anchor, o, dur)) for o in speech_offsets(anchor, 3, dur)]
            fmt = lambda xs: "/".join(f"{c:.2f}" if c is not None else "None" for c in xs)
            print(f"  {dur:>4.1f}s  same={fmt(cos_same)}  diff={fmt(cos_diff)}  "
                  f"echo={fmt(cos_echo)}")

    requests.post(PVAD + "/reset", json={"session_id": SID}, timeout=5)


if __name__ == "__main__":
    main()
