"""Offline VoxCPM2 synthesis smoke test: load, stream one Chinese sentence,
measure first-chunk latency and RTF, save 48k + 24k wavs for listening.

Usage:
  CUDA_VISIBLE_DEVICES=7 LD_LIBRARY_PATH=/mnt/disk1/miniconda3/envs/tempo/lib \
  venv-voxcpm/bin/python -I scripts/voxcpm_smoke.py
"""
import time
import wave
from pathlib import Path

import numpy as np
import soxr

MODEL_PATH = "/mnt/disk3/krisl/soulx/VoxCPM2"
VOICE_DESC = "(年轻女性，声音温柔亲切，语速适中，吐字清晰)"
TEXT = "你好呀，很高兴认识你。今天天气真不错，要不要一起出去走走，顺便聊聊天呢？"
OUT_DIR = Path("/home/research/krisl/soulx/results")


def save_wav(path, samples, sr):
    data = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(data.tobytes())


def main():
    from voxcpm import VoxCPM

    t0 = time.time()
    model = VoxCPM.from_pretrained(MODEL_PATH, load_denoiser=False, optimize=True)
    print(f"[smoke] load+compile: {time.time()-t0:.1f}s", flush=True)

    # warmup (compiled path) — short text
    t0 = time.time()
    for _ in model.generate_streaming(f"{VOICE_DESC}你好。"):
        pass
    print(f"[smoke] warmup gen: {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    t_first = None
    chunks = []
    for wav in model.generate_streaming(
        f"{VOICE_DESC}{TEXT}", cfg_value=2.0, inference_timesteps=10
    ):
        if t_first is None:
            t_first = time.time() - t0
        chunks.append(np.asarray(wav, dtype=np.float32))
    t_gen = time.time() - t0

    audio = np.concatenate(chunks)
    dur = audio.size / 48000
    print(f"[smoke] first-chunk: {t_first:.2f}s | gen: {t_gen:.2f}s | "
          f"audio: {dur:.2f}s | RTF: {t_gen/dur:.3f} | chunks: {len(chunks)}",
          flush=True)

    OUT_DIR.mkdir(exist_ok=True)
    save_wav(OUT_DIR / "voxcpm_smoke_48k.wav", audio, 48000)
    rs = soxr.ResampleStream(48000, 24000, 1, dtype="float32")
    out = [rs.resample_chunk(c) for c in chunks]
    out.append(rs.resample_chunk(np.zeros(0, dtype=np.float32), last=True))
    audio24 = np.concatenate(out)
    save_wav(OUT_DIR / "voxcpm_smoke_24k.wav", audio24, 24000)
    print(f"[smoke] saved 48k ({audio.size/48000:.2f}s) and 24k "
          f"({audio24.size/24000:.2f}s) wavs to {OUT_DIR}", flush=True)

    # second run on the same model — steady-state latency (what users feel)
    t0 = time.time()
    t_first = None
    n = 0
    for wav in model.generate_streaming(
        f"{VOICE_DESC}好的，那我们出发吧，路上我再跟你讲讲最近发生的趣事。"
    ):
        if t_first is None:
            t_first = time.time() - t0
        n += np.asarray(wav).size
    t_gen = time.time() - t0
    dur = n / 48000
    print(f"[smoke] run2 first-chunk: {t_first:.2f}s | gen: {t_gen:.2f}s | "
          f"audio: {dur:.2f}s | RTF: {t_gen/dur:.3f}", flush=True)


if __name__ == "__main__":
    main()
