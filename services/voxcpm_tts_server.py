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

Voice — reference cloning with a persistent anchor (2026-10-29):
  Zero-shot mode (what this server did before) samples fresh Gaussian noise
  per call (modules/locdit/unified_cfm.py: z = randn_like(x1), no seed), so
  every LLM text segment got an independently sampled timbre — that was the
  cross-segment voice/emotion drift. Fix: VoxCPM2 reference mode. One anchor
  wav is encoded once via build_prompt_cache() and EVERY segment reuses that
  cache through _generate_with_prompt_cache() — same streaming contract,
  consistent voice.

  Anchor resolution order:
    1. VOXCPM_REF_WAV   — explicit reference audio (e.g. a voice you like)
    2. VOXCPM_ANCHOR_WAV (default $SOULX/data/voxcpm_anchor.wav) — persists
       across restarts; delete the file to re-roll the voice
    3. neither exists  — synthesize a fixed calibration sentence once with
       the (LoRA-adapted) model, save it as the anchor
  If anchoring fails for any reason the server falls back to zero-shot and
  logs a warning — it never refuses to serve.

LoRA — self-trained checkpoint via VOXCPM_LORA_PATH (package-native format:
lora_config.json + lora_weights.safetensors). The config MUST be passed
explicitly (r/alpha from the checkpoint — the package default r=8/alpha=16
silently skips mismatched weights). Loading is verified: zero loaded keys or
any skipped key aborts startup (a silently-disabled LoRA is worse than a
crash). Set VOXCPM_LORA_PATH="" to run the bare base model.

voice_desc (text voice design) is accepted for old clients but IGNORED in
cloning mode — the anchor reference wins.

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
     VOXCPM_LORA_PATH (default the self-trained step_0000500 checkpoint),
     VOXCPM_REF_WAV / VOXCPM_ANCHOR_WAV (anchor resolution, see above),
     VOXCPM_VOICE_DESC (only used by the zero-shot fallback).
Startup note: optimize=True (torch.compile) makes the first launch take
2-4 minutes; /health reports loaded=true once ready.
"""

import argparse
import json
import os
import queue
import re
import threading
import time

import numpy as np
import soxr
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

SOULX = "/home/research/krisl/soulx"
MODEL_PATH = os.environ.get("VOXCPM_MODEL_PATH", "/mnt/disk3/krisl/soulx/VoxCPM2")
LORA_PATH = os.environ.get(
    "VOXCPM_LORA_PATH", "/vepfs-cnbjc4e12c6d7e1f/krisl/step_0000500"
).strip()
REF_WAV = os.environ.get("VOXCPM_REF_WAV", "").strip()
ANCHOR_WAV = os.environ.get("VOXCPM_ANCHOR_WAV", f"{SOULX}/data/voxcpm_anchor.wav")
VOICE_DESC = os.environ.get("VOXCPM_VOICE_DESC", "")  # zero-shot fallback only
OUT_RATE = 24000  # web app / frontend contract — do not change unilaterally

# Fixed sentence used once to sample the anchor voice. Neutral, mid-pitch,
# long enough (~4 s) for a clean ECAPA/VAE embedding of the timbre.
CALIB_TEXT = "你好呀，很高兴认识你。今天天气不错，我们随便聊聊天吧。"

MODEL = None
MODEL_SR = None   # native sample rate of the loaded model (VoxCPM2: 48000)
PROMPT_CACHE = None  # reference-mode cache; None => zero-shot fallback
LORA_LOADED = False
GEN_LOCK = threading.Lock()  # serializes GPU generation; producer-only

_DONE = object()  # sentinel: producer finished (after error or normally)

app = FastAPI(title="voxcpm-tts")


class SynthRequest(BaseModel):
    input: str
    # Live-tunable per-request knobs (web UI settings panel). All optional;
    # missing/None falls back to the server defaults.
    cfg_value: float | None = None          # expressiveness, ~0.5-5
    inference_timesteps: int | None = None  # quality/speed tradeoff, ~4-40
    voice_desc: str | None = None           # legacy voice design — IGNORED
                                             # in cloning mode (anchor wins)


def _clamp(value, lo, hi, default):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v))


def _lora_config_from_checkpoint(path):
    """LoRAConfig built from the checkpoint's lora_config.json.

    The json's lora_config block maps 1:1 onto the package's LoRAConfig
    fields (enable_lm/enable_dit/enable_proj/r/alpha/dropout/target_*).
    Passing the real r/alpha matters: from_pretrained auto-creates a
    default r=8/alpha=16 config when none is given, and mismatched shapes
    make load_lora_weights silently skip every key.
    """
    from voxcpm.core import LoRAConfig

    cfg_path = os.path.join(path, "lora_config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"{path} has no lora_config.json — cannot infer LoRA shape"
        )
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = json.load(f).get("lora_config", {})
    return LoRAConfig(**raw)


def _write_wav(path, wav, sample_rate):
    import soundfile as sf

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    sf.write(path, np.asarray(wav, dtype=np.float32), sample_rate)


def _generate_anchor():
    """Sample the anchor voice once with the (LoRA-adapted) model."""
    wav = MODEL.generate(CALIB_TEXT, max_len=500)
    _write_wav(ANCHOR_WAV, wav, MODEL_SR)
    print(
        f"[voxcpm-tts] anchor voice sampled ({len(wav)/MODEL_SR:.1f}s) "
        f"-> {ANCHOR_WAV}",
        flush=True,
    )


def _build_prompt_cache():
    """Resolve the anchor reference and encode it once (reference mode)."""
    global PROMPT_CACHE

    ref = REF_WAV if REF_WAV and os.path.exists(REF_WAV) else None
    if ref is None and os.path.exists(ANCHOR_WAV):
        ref = ANCHOR_WAV
    try:
        if ref is None:
            _generate_anchor()
            ref = ANCHOR_WAV
        PROMPT_CACHE = MODEL.tts_model.build_prompt_cache(reference_wav_path=ref)
        print(
            f"[voxcpm-tts] reference voice locked from {ref} "
            f"({'explicit VOXCPM_REF_WAV' if ref == REF_WAV else 'anchor'})",
            flush=True,
        )
    except Exception as exc:
        # anchorless is degraded (drifty voice) but serviceable — never
        # refuse to serve over this.
        PROMPT_CACHE = None
        print(
            f"[voxcpm-tts] WARNING: voice anchoring failed "
            f"({type(exc).__name__}: {exc}) — falling back to zero-shot, "
            f"cross-segment drift will return",
            flush=True,
        )


def load_model():
    """Load VoxCPM2 (+ LoRA) and lock the anchor voice at startup."""
    global MODEL, MODEL_SR, LORA_LOADED
    from voxcpm import VoxCPM

    t0 = time.time()
    lora_config = None
    if LORA_PATH and os.path.isdir(LORA_PATH):
        lora_config = _lora_config_from_checkpoint(LORA_PATH)
        print(
            f"[voxcpm-tts] LoRA config from {LORA_PATH}: "
            f"r={lora_config.r} alpha={lora_config.alpha} "
            f"lm={lora_config.enable_lm} dit={lora_config.enable_dit} "
            f"proj={lora_config.enable_proj}",
            flush=True,
        )
    elif LORA_PATH:
        raise FileNotFoundError(
            f"VOXCPM_LORA_PATH={LORA_PATH} does not exist — fix the path "
            f"or set it empty to run the bare base model"
        )

    # lora_config only (not lora_weights_path): from_pretrained would load
    # the weights internally and only PRINT the key counts — we load them
    # ourselves to verify programmatically. Fresh LoRALinear weights are
    # zero-init, so the __init__ warm-up generate runs the bare base model.
    MODEL = VoxCPM.from_pretrained(
        MODEL_PATH,
        load_denoiser=False,  # no reference-audio denoising needed here
        optimize=True,        # torch.compile: slow first start, fast steady state
        lora_config=lora_config,
    )
    MODEL_SR = MODEL.tts_model.sample_rate

    if lora_config is not None:
        loaded, skipped = MODEL.load_lora(LORA_PATH)
        if not loaded or skipped:
            raise RuntimeError(
                f"LoRA load failed verification: {len(loaded)} keys loaded, "
                f"{len(skipped)} skipped (first: {skipped[:3]})"
            )
        LORA_LOADED = True
        print(
            f"[voxcpm-tts] LoRA active: {len(loaded)} parameters loaded, "
            f"0 skipped",
            flush=True,
        )

    print(
        f"[voxcpm-tts] model loaded from {MODEL_PATH} "
        f"(sr={MODEL_SR}, load+compile {time.time()-t0:.1f}s, "
        f"lora={'on' if LORA_LOADED else 'off'})",
        flush=True,
    )

    _build_prompt_cache()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "loaded": MODEL is not None,
        "sample_rate": OUT_RATE,
        "model_sample_rate": MODEL_SR,
        "lora": LORA_LOADED,
        "anchored": PROMPT_CACHE is not None,
    }


@app.post("/synthesize")
def synthesize(req: SynthRequest):
    text = (req.input or "").strip()
    if not text or MODEL is None:
        return StreamingResponse(iter(()), media_type="audio/L16")

    # per-request tunables (clamped; None -> server defaults)
    cfg = _clamp(req.cfg_value, 0.5, 5.0, 2.0) if req.cfg_value is not None else 2.0
    steps = (
        int(_clamp(req.inference_timesteps, 4, 40, 10))
        if req.inference_timesteps is not None
        else 10
    )

    # parity with the facade's _generate text cleanup
    text = re.sub(r"\s+", " ", text.replace("\n", " "))

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
                if PROMPT_CACHE is not None:
                    # reference mode: every segment reuses the anchor cache
                    # (the endpoint core._generate funnels into this same
                    # private call — pinned voxcpm 2.0.3, re-check on upgrade)
                    gen = MODEL.tts_model._generate_with_prompt_cache(
                        target_text=text,
                        prompt_cache=PROMPT_CACHE,
                        cfg_value=cfg,
                        inference_timesteps=steps,
                        streaming=True,
                    )
                else:
                    # zero-shot fallback (anchor failed): voice_desc prefix
                    # is the only timbre steering we have left
                    voice = (req.voice_desc or "").strip()[:200] or VOICE_DESC
                    gen = MODEL.generate_streaming(
                        f"{voice}{text}",
                        cfg_value=cfg,
                        inference_timesteps=steps,
                    )
                # reference mode yields (wav, new_tokens, audio_feats)
                # tuples per chunk (the facade's streaming loop unpacks the
                # same way); the zero-shot fallback yields bare arrays.
                for item in gen:
                    if cancel.is_set():
                        break
                    wav = item[0] if isinstance(item, tuple) else item
                    if hasattr(wav, "cpu"):
                        wav = wav.squeeze(0).cpu().numpy()
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
