"""Target-speaker separation (TSE) microservice for the SoulX web app —
task 2 of the 2026-11-05/06 pvad follow-up (user request: 背景有其他人说话
下的主说话人音频分离, modular with an independent switch).

Wraps ClearVoice's MossFormer2_SS_16K two-speaker separator as an HTTP
sidecar, run from /mnt/disk3/krisl/soulx/venv-tse (the only env with
clearvoice; created 2026-11-06 — torch matching the driver matters: use
+cu124 wheels, the driver is 12.4). GPU strongly preferred: CPU measures
0.89x realtime (44.9 s for 40 s audio) — fine for offline replays, too
slow for the per-round gate cadence.

This service is deliberately STATELESS and DECISION-FREE:
  - no sessions, no reference embeddings — it separates whatever wav it
    is handed into two streams and returns both;
  - picking "which stream is the main speaker" is the CALLER's job (the
    app does it with the pvad ECAPA /score call).
Stream identity is NOT stable across time (measured: the background
video lands in s1 during some stretches and s2 during others — 17 flips
in a 160 s session), so callers must select per-window, never assume.

Wire contract:
  GET  /health    -> {"status": "ok", "loaded": bool, "device": str}
  POST /separate  {"audio": base64 float32 LE, "sample_rate": 16000}
                 -> {"s1": base64 float32, "s2": base64 float32,
                     "duration_s": float}
                    Both streams are the same length as the input.
                    >= 2 s of audio recommended (the model needs context;
                    < 1 s returns 400).

Offline validation (2026-11-06, session a5f92f1d — see
docs/说话人分离验证报告-20261106.md): separation cleanly isolates the
user during user-only turns, but ECAPA verification bands do NOT
separate (USER-under-video [0.06, 0.70] vs VIDEO-only [0.01, 0.36],
still overlapping) and feeding the gate with separated-audio scores
changes nothing (interrupt bursts 61% vs 64% suppression). The service
therefore ships as EXPERIMENTAL infrastructure behind the app-side
`speaker_sep` switch (default off, shadow-logging only).

Usage:
  CUDA_VISIBLE_DEVICES=7 MODELSCOPE_CACHE=/mnt/disk3/krisl/soulx/modelscope_cache \
    /mnt/disk3/krisl/soulx/venv-tse/bin/python services/tse_server.py [--port 8087]
"""

import argparse
import base64
import os

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

SAMPLE_RATE = 16000
MIN_SEPARATE_S = 1.0
# ClearVoice resolves its checkpoint dir ("checkpoints/<model>", from the
# bundled YAML) against the process cwd — pin to this dedicated workdir on
# /mnt/disk3 (home disk is 98% full; the MossFormer2_SS_16K weights are
# ~640 MB and already live there).
WORKDIR = "/mnt/disk3/krisl/soulx/tse_work"

app = FastAPI()

_cv = None          # ClearVoice instance (loaded lazily at startup)
_device = "cpu"


class SeparateRequest(BaseModel):
    audio: str
    sample_rate: int = SAMPLE_RATE


def _decode(audio_b64: str, sr: int) -> np.ndarray:
    wav = np.frombuffer(base64.b64decode(audio_b64), dtype=np.float32)
    if sr != SAMPLE_RATE:
        raise HTTPException(400, f"resampling not supported (got {sr} Hz)")
    return wav


@app.on_event("startup")
def load_model():
    global _cv, _device
    import torch
    import clearvoice.networks as _cn
    from clearvoice import ClearVoice
    # ClearVoice resolves its model dir from the YAML config as the
    # RELATIVE path "checkpoints/MossFormer2_SS_16K" — i.e. against the
    # process cwd, whatever it happens to be. Home is nearly full, so the
    # weights live on /mnt/disk3; pin the process to that workdir (the
    # service is dedicated, chdir is safe).
    os.chdir(WORKDIR)
    # ClearVoice picks "the GPU with most free memory" by calling
    # nvidia-smi DIRECTLY — it sees all 8 GPUs of the box and returns a
    # global index, which breaks under CUDA_VISIBLE_DEVICES ("invalid
    # device ordinal"). Respect the operator's choice instead: with
    # CUDA_VISIBLE_DEVICES=N exactly one device is visible, so pin it to
    # cuda:0 and let the env var do the selecting (the shared-GPU
    # discipline on this box).
    _cn.SpeechModel.get_free_gpu = lambda self: 0
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _cv = ClearVoice(task="speech_separation",
                     model_names=["MossFormer2_SS_16K"])
    # warm up with 2 s of silence so the first real request doesn't pay
    # the lazy-init cost
    _cv(np.zeros((1, 2 * SAMPLE_RATE), dtype=np.float32), False)


@app.get("/health")
def health():
    return {"status": "ok", "loaded": _cv is not None, "device": _device}


@app.post("/separate")
def separate(req: SeparateRequest):
    if _cv is None:
        raise HTTPException(503, "model not loaded")
    wav = _decode(req.audio, req.sample_rate)
    if wav.shape[0] < int(MIN_SEPARATE_S * SAMPLE_RATE):
        raise HTTPException(400, "need >= 1 s of audio")
    out = _cv(np.reshape(wav, [1, -1]).astype(np.float32), False)
    # out: [spk, batch, length]
    s1, s2 = out[0, 0, :], out[1, 0, :]
    return {
        "s1": base64.b64encode(np.ascontiguousarray(s1, dtype=np.float32).tobytes()).decode(),
        "s2": base64.b64encode(np.ascontiguousarray(s2, dtype=np.float32).tobytes()).decode(),
        "duration_s": wav.shape[0] / SAMPLE_RATE,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8087)
    args = ap.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
