#!/usr/bin/env python3
"""Build offline_infer eval input dirs from eot-bench-zh and xyz parquets.

For each selected row, writes <out>/<source>/<sample_id>/input.wav
(16 kHz mono float wav — the layout offline_infer.py walks for "input.wav").

Row selection: durations filtered to [min_dur, max_dur], then N evenly spaced
picks across the duration-sorted rows, so the sample set spans short and long
turns. Deterministic (no RNG).

Run from ANY directory with:
  python -I build_eval_inputs.py --out <dir> --num 12
"""
import argparse
import glob
import io
import os
import wave

import numpy as np
import pyarrow.parquet as pq

TARGET_SR = 16000

SOURCES = {
    "eotbench_zh": os.path.join(
        "/home/research/krisl/eot/eot_bench/hf/hub/"
        "datasets--livekit--eot-bench-data/snapshots",
        "*/data/zh/validation-*.parquet",
    ),
    "xyz": "/home/research/krisl/eot/eot_bench/data/xyz-eot-data/"
    "validation-00000-of-00001.parquet",
}


def decode_wav_bytes(raw: bytes):
    """Decode WAV bytes -> (float32 mono samples, sample_rate)."""
    with wave.open(io.BytesIO(raw), "rb") as w:
        nch = w.getnchannels()
        sr = w.getframerate()
        sw = w.getsampwidth()
        frames = w.readframes(w.getnframes())
    if sw == 2:
        audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 4:
        audio = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported sample width {sw}")
    if nch > 1:
        audio = audio.reshape(-1, nch).mean(axis=1)
    return audio, sr


def resample(audio, sr):
    if sr == TARGET_SR:
        return audio
    import soxr

    return soxr.resample(audio, sr, TARGET_SR, quality="VHQ")


def pick_rows(table, num, min_dur, max_dur):
    rows = []
    for i in range(table.num_rows):
        d = table.column("duration")[i].as_py()
        if d is None or not (min_dur <= d <= max_dur):
            continue
        rows.append((i, d))
    rows.sort(key=lambda x: x[1])  # by duration
    if not rows:
        raise SystemExit("no rows in duration range")
    if len(rows) <= num:
        return [i for i, _ in rows]
    step = (len(rows) - 1) / (num - 1)
    return [rows[int(round(k * step))][0] for k in range(num)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--num", type=int, default=12)
    ap.add_argument("--min-dur", type=float, default=8.0)
    ap.add_argument("--max-dur", type=float, default=90.0)
    ap.add_argument("--sources", nargs="+", default=list(SOURCES))
    args = ap.parse_args()

    manifest = []
    for name in args.sources:
        pattern = SOURCES[name]
        if "*" in pattern:
            matches = sorted(glob.glob(pattern))
            if not matches:
                raise SystemExit(f"no parquet found for {pattern}")
            path = matches[0]
        else:
            path = pattern
        table = pq.read_table(path)
        idxs = pick_rows(table, args.num, args.min_dur, args.max_dur)
        print(f"[{name}] {path}: picked {len(idxs)} of {table.num_rows} rows")
        for i in idxs:
            rid = table.column("id")[i].as_py()
            dur = table.column("duration")[i].as_py()
            raw = table.column("audio")[i].as_py()["bytes"]
            audio, sr = decode_wav_bytes(raw)
            audio = resample(audio, sr)
            out_dir = os.path.join(args.out, name, str(rid))
            os.makedirs(out_dir, exist_ok=True)
            out_wav = os.path.join(out_dir, "input.wav")
            pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
            with wave.open(out_wav, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(TARGET_SR)
                w.writeframes(pcm.tobytes())
            print(f"  {rid}: {dur:.1f}s (src sr={sr}) -> {out_wav}")
            manifest.append(
                {"source": name, "id": rid, "duration": dur, "dir": out_dir}
            )

    import json

    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"wrote {len(manifest)} samples, manifest at {args.out}/manifest.json")


if __name__ == "__main__":
    main()
