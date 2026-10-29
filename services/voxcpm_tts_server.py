"""VoxCPM2 TTS microservice for the SoulX web dialogue app.

Serves raw int16 LE PCM @ 24 kHz mono (the web app / frontend contract —
frontend ring buffer is 24000 samples/s, app.py computes duration as
len(chunk)/48000.0 i.e. 2 bytes/sample at 24 kHz). VoxCPM2 generates at
48 kHz, so every chunk passes through a stateful soxr.ResampleStream
(no block-boundary artifacts).

Wire contract (differs from FireRedTTS :8081 on purpose — no WAV header,
no 44-byte sentinel to skip):
  GET  /health     -> {"status": "ok", "loaded": bool, "sample_rate": 24000}
  POST /synthesize {"input": str} -> chunked stream of raw int16 PCM bytes

Voice: text-description prefix (voice design) from VOXCPM_VOICE_DESC —
change timbre by editing the env, no code change. Voice cloning via
reference_wav is possible upstream but not wired here (see docs).

Concurrency model (lessons from the FireRedTTS SYNTH_LOCK deadlock):
  - GEN_LOCK serializes GPU access, held only by the producer thread.
  - The lock is NEVER held across a network yield — the consumer runs in
    the request thread and only reads from a queue.
  - If the client disconnects (barge-in), the consumer sets a cancel flag
    and closes the generator from the producer's side on next check;
    worst case the abandoned generation runs to the end of its single
    short segment (LLM already splits text at punctuation), so zombie
    work is bounded.

Usage:
  CUDA_VISIBLE_DEVICES=7 venv-voxcpm/bin/python services/voxcpm_tts_server.py \
      [--port 8085]
Env: VOXCPM_MODEL_PATH (default /mnt/disk3/krisl/soulx/VoxCPM2),
     VOXCPM_VOICE_DESC (default warm female Chinese voice description).
Startup note: optimize=True (torch.compile) makes the first launch take
2-4 minutes; /health reports loaded=true once ready.
"""

import argparse
import os
import queue
import threading
import time

import numpy as np
import soxr
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

MODEL_PATH = os.environ.get("VOXCPM_MODEL_PATH", "/mnt/disk3/krisl/soulx/VoxCPM2")
VOICE_DESC = os.environ.get(
    "VOXCPM_VOICE_DESC", "(年轻女性，声音温柔亲切，语速适中，吐字清晰)"
)
OUT_RATE = 24000  # web app / frontend contract — do not change unilaterally

MODEL = None
MODEL_SR = None  # native sample rate of the loaded model (VoxCPM2: 48000)
GEN_LOCK = threading.Lock()  # serializes GPU generation; producer-only

_DONE = object()  # sentinel: producer finished (after error or normally)

app = FastAPI(title="voxcpm-tts")


class SynthRequest(BaseModel):
    input: str
    # Live-tunable per-request knobs (web UI settings panel). All optional;
    # missing/None falls back to the env defaults.
    cfg_value: float | None = None          # expressiveness, ~0.5-5
    inference_timesteps: int | None = None  # quality/speed tradeoff, ~4-40
    voice_desc: str | None = None           # voice-design prefix override


def _clamp(value, lo, hi, default):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v))


def load_model():
    """Load VoxCPM2 at startup (blocking — takes minutes with compile)."""
    global MODEL, MODEL_SR
    from voxcpm import VoxCPM

    t0 = time.time()
    MODEL = VoxCPM.from_pretrained(
        MODEL_PATH,
        load_denoiser=False,  # no reference-audio denoising needed here
        optimize=True,        # torch.compile: slow first start, fast steady state
    )
    MODEL_SR = MODEL.tts_model.sample_rate
    print(
        f"[voxcpm-tts] model loaded from {MODEL_PATH} "
        f"(sr={MODEL_SR}, load+compile {time.time()-t0:.1f}s)",
        flush=True,
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "loaded": MODEL is not None,
        "sample_rate": OUT_RATE,
        "model_sample_rate": MODEL_SR,
    }


@app.post("/synthesize")
def synthesize(req: SynthRequest):
    text = (req.input or "").strip()
    if not text or MODEL is None:
        return StreamingResponse(iter(()), media_type="audio/L16")

    # per-request tunables (clamped; None -> env/server defaults)
    cfg = _clamp(req.cfg_value, 0.5, 5.0, 2.0) if req.cfg_value is not None else 2.0
    steps = (
        int(_clamp(req.inference_timesteps, 4, 40, 10))
        if req.inference_timesteps is not None
        else 10
    )
    voice = (req.voice_desc or "").strip()[:200] or VOICE_DESC

    # queue carries raw pcm bytes, ("err", msg) tuples, or _DONE.
    # Unbounded on purpose: bounded by one short LLM segment (~few hundred
    # KB); a bounded queue could block the producer on a dead reader.
    q = queue.Queue()
    cancel = threading.Event()

    def producer():
        gen = None
        try:
            with GEN_LOCK:
                if cancel.is_set() or MODEL is None:
                    return
                rs = soxr.ResampleStream(MODEL_SR, OUT_RATE, 1, dtype="float32")
                gen = MODEL.generate_streaming(
                    f"{voice}{text}",
                    cfg_value=cfg,
                    inference_timesteps=steps,
                )
                for wav in gen:
                    if cancel.is_set():
                        break
                    wav = np.asarray(wav, dtype=np.float32)
                    out = rs.resample_chunk(wav)
                    if out.size:
                        q.put(
                            (np.clip(out, -1.0, 1.0) * 32767.0)
                            .astype("<i2")
                            .tobytes()
                        )
                # flush the resampler tail
                tail = rs.resample_chunk(np.zeros(0, dtype=np.float32), last=True)
                if tail.size and not cancel.is_set():
                    q.put((np.clip(tail, -1.0, 1.0) * 32767.0).astype("<i2").tobytes())
        except Exception as exc:
            q.put(("err", f"{type(exc).__name__}: {exc}"))
        finally:
            # cancel path: close the generator in its own (producer) thread —
            # never from the consumer thread, where close() on an
            # actively-executing generator raises ValueError.
            if gen is not None:
                try:
                    gen.close()
                except Exception:
                    pass
            q.put(_DONE)

    threading.Thread(target=producer, daemon=True).start()

    def stream():
        try:
            while True:
                item = q.get()
                if item is _DONE:
                    break
                if isinstance(item, tuple) and item[0] == "err":
                    print(f"[voxcpm-tts] generation failed: {item[1]}", flush=True)
                    break
                yield item
        finally:
            # client gone (barge-in breaks out of app.py's loop, FastAPI
            # closes this generator) or stream finished — either way stop
            # the producer at the next chunk boundary.
            cancel.set()

    return StreamingResponse(
        stream(),
        media_type="audio/L16",
        headers={
            "X-Sample-Rate": str(OUT_RATE),
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8085)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    load_model()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
