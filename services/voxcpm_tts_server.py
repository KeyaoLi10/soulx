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

LoRA — TWO self-trained checkpoints loaded side by side (2026-11-09),
selected per request via SynthRequest.lora ("step_0000500" | "step_0000400",
default 500; unknown names clamp to the default with a warning). Each
checkpoint gets its OWN model instance and its OWN anchor wav — the anchor
cache is built from audio synthesized by that same LoRA-adapted model, and
reusing a 500-generated anchor under the 400 adapter breaks the voice
stability the anchor exists for (each voice must stay consistent within
itself; the two checkpoints are intentionally allowed to sound different).
Checkpoint resolution:
    VOXCPM_LORA_PATH_500 (default .../step_0000500) -> "step_0000500"
    VOXCPM_LORA_PATH_400 (default .../step_0000400) -> "step_0000400"
  Set either to "" to drop that checkpoint; both empty runs the bare base
  model (single shared instance). Legacy VOXCPM_LORA_PATH still works and
  overrides the 500 entry. Package-native format per checkpoint:
  lora_config.json + lora_weights.safetensors. The config MUST be passed
  explicitly (r/alpha from the checkpoint — the package default r=8/alpha=16
  silently skips mismatched weights). Loading is verified: zero loaded keys
  or any skipped key aborts startup (a silently-disabled LoRA is worse than
  a crash).
  Per-LoRA anchors: 500 keeps $SOULX/data/voxcpm_anchor.wav (long-standing,
  DO NOT delete casually — it IS the Elva voice); 400 uses
  $SOULX/data/voxcpm_anchor_step_0000400.wav, synthesized on first boot.
  VOXCPM_REF_WAV overrides the anchor for every checkpoint (explicit
  reference audio, e.g. A/B against a fixed voice).

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
     VOXCPM_LORA_PATH_500 / VOXCPM_LORA_PATH_400 (default the self-trained
       step_0000500 / step_0000400 checkpoints; "" drops that checkpoint),
     VOXCPM_LORA_PATH (legacy single-checkpoint override of the 500 entry),
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

# Dual-checkpoint registry (2026-11-09): name -> checkpoint dir. Explicit ""
# drops the entry; both empty (or no lora_config.json anywhere) runs the bare
# base model as a single shared instance under the DEFAULT_LORA name.
LORA_SPECS = [
    (
        "step_0000500",
        os.environ.get(
            "VOXCPM_LORA_PATH", ""  # legacy single-checkpoint override, if set
        ).strip()
        or os.environ.get(
            "VOXCPM_LORA_PATH_500", "/vepfs-cnbjc4e12c6d7e1f/krisl/step_0000500"
        ).strip(),
    ),
    (
        "step_0000400",
        os.environ.get(
            "VOXCPM_LORA_PATH_400", "/vepfs-cnbjc4e12c6d7e1f/krisl/step_0000400"
        ).strip(),
    ),
]
LORA_SPECS = [(name, path) for name, path in LORA_SPECS if path]
DEFAULT_LORA = "step_0000500"  # unknown request names clamp here
REF_WAV = os.environ.get("VOXCPM_REF_WAV", "").strip()
ANCHOR_WAV = os.environ.get("VOXCPM_ANCHOR_WAV", f"{SOULX}/data/voxcpm_anchor.wav")
VOICE_DESC = os.environ.get("VOXCPM_VOICE_DESC", "")  # zero-shot fallback only
OUT_RATE = 24000  # web app / frontend contract — do not change unilaterally
# Text normalization (2026-10-30): VoxCPM2's facade generate() has
# normalize=True but the prompt-cache fast path we call does NOT normalize —
# digits/symbols reached the acoustic model raw ("23.5度" read as digits,
# "0.6B" mangled). We run the same TextNormalizer the facade uses, before
# generation, for both the reference and zero-shot paths.
TEXT_NORM = os.environ.get("VOXCPM_TEXT_NORM", "1").strip() != "0"

# Fixed sentence used once to sample the anchor voice. Neutral, mid-pitch,
# long enough (~4 s) for a clean ECAPA/VAE embedding of the timbre.
CALIB_TEXT = "你好呀，很高兴认识你。今天天气不错，我们随便聊聊天吧。"

MODEL = None           # DEFAULT_LORA's instance (or the bare base model)
MODEL_SR = None        # native sample rate of the loaded model (VoxCPM2: 48000)
MODELS = {}            # lora name -> VoxCPM instance (single entry in bare mode)
PROMPT_CACHES = {}     # lora name -> reference-mode cache; None => zero-shot
GEN_LOCK = threading.Lock()  # serializes GPU generation; producer-only

TEXT_NORMALIZER = None  # lazy singleton — loads WeTextProcessing (slow, ~10s)
TN_LOCK = threading.Lock()


def _get_normalizer():
    """Lazy-init the TextNormalizer the facade's normalize=True path uses.

    Imports are deferred: WeTextProcessing pulls in a pynini/OpenFst stack
    (~10s) that we only pay on the first synthesized request (or never, if
    the default text never contains digits/symbols and TEXT_NORM=0).
    Init failures degrade to no-op (raw text) — TTS must not go down over
    a normalization hiccup.
    """
    global TEXT_NORMALIZER
    if TEXT_NORMALIZER is not None:
        return TEXT_NORMALIZER
    if not TEXT_NORM:
        return None
    with TN_LOCK:
        if TEXT_NORMALIZER is None:
            try:
                from voxcpm.utils.text_normalize import TextNormalizer
                t0 = time.time()
                TEXT_NORMALIZER = TextNormalizer()
                print(
                    f"[voxcpm-tts] text normalization enabled "
                    f"(WeTextProcessing loaded in {time.time()-t0:.1f}s)",
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"[voxcpm-tts] WARNING: TextNormalizer init failed "
                    f"({type(exc).__name__}: {exc}) — synthesizing raw text",
                    flush=True,
                )
    return TEXT_NORMALIZER


def _normalize_text(text):
    tn = _get_normalizer()
    if tn is None:
        return text
    try:
        return tn.normalize(text)
    except Exception as exc:
        # a malformed string must not kill the request — raw is degraded but
        # speakable
        print(
            f"[voxcpm-tts] WARNING: normalize({text[:40]!r}) failed "
            f"({type(exc).__name__}: {exc}) — using raw text",
            flush=True,
        )
        return text

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
    lora: str | None = None                 # checkpoint name, e.g.
                                             # "step_0000500"/"step_0000400";
                                             # unknown -> default checkpoint


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


def _anchor_path(name):
    """Per-LoRA anchor wav. 500 keeps the long-standing data/voxcpm_anchor.wav
    (it IS the Elva voice — deleting it re-rolls the default voice); every
    other checkpoint gets its own <name> sibling, synthesized by that same
    LoRA-adapted model on first boot."""
    if name == "step_0000500":
        return ANCHOR_WAV
    root, ext = os.path.splitext(ANCHOR_WAV)
    return f"{root}_{name}{ext or '.wav'}"


def _generate_anchor(model, anchor_path):
    """Sample the anchor voice once with the (LoRA-adapted) model."""
    # model-local sr, NOT the global MODEL_SR: load_model() only sets the
    # global after ALL checkpoints load, so during boot it is still None
    # (this exact bug made the first 400-anchor synthesis crash with
    # "samplerate must be specified")
    sr = model.tts_model.sample_rate
    wav = model.generate(CALIB_TEXT, max_len=500)
    _write_wav(anchor_path, wav, sr)
    print(
        f"[voxcpm-tts] anchor voice sampled ({len(wav)/sr:.1f}s) "
        f"-> {anchor_path}",
        flush=True,
    )


def _build_prompt_cache(name, model):
    """Resolve this checkpoint's anchor reference and encode it once."""
    anchor = _anchor_path(name)

    ref = REF_WAV if REF_WAV and os.path.exists(REF_WAV) else None
    if ref is None and os.path.exists(anchor):
        ref = anchor
    try:
        if ref is None:
            _generate_anchor(model, anchor)
            ref = anchor
        PROMPT_CACHES[name] = model.tts_model.build_prompt_cache(
            reference_wav_path=ref
        )
        print(
            f"[voxcpm-tts] reference voice locked for {name} from {ref} "
            f"({'explicit VOXCPM_REF_WAV' if ref == REF_WAV else 'anchor'})",
            flush=True,
        )
    except Exception as exc:
        # anchorless is degraded (drifty voice) but serviceable — never
        # refuse to serve over this.
        PROMPT_CACHES[name] = None
        print(
            f"[voxcpm-tts] WARNING: voice anchoring failed for {name} "
            f"({type(exc).__name__}: {exc}) — falling back to zero-shot, "
            f"cross-segment drift will return",
            flush=True,
        )


def _load_one(name, lora_path):
    """Load one VoxCPM2 instance (+ its LoRA if a dir is given) into
    MODELS[name]. Returns (loaded_key_count, lora_on)."""
    from voxcpm import VoxCPM

    t0 = time.time()
    lora_config = None
    if lora_path:
        if not os.path.isdir(lora_path):
            raise FileNotFoundError(
                f"LoRA path for {name} ({lora_path}) does not exist — fix "
                f"the path or set it empty to run the bare base model"
            )
        lora_config = _lora_config_from_checkpoint(lora_path)
        print(
            f"[voxcpm-tts] LoRA config for {name} from {lora_path}: "
            f"r={lora_config.r} alpha={lora_config.alpha} "
            f"lm={lora_config.enable_lm} dit={lora_config.enable_dit} "
            f"proj={lora_config.enable_proj}",
            flush=True,
        )

    # lora_config only (not lora_weights_path): from_pretrained would load
    # the weights internally and only PRINT the key counts — we load them
    # ourselves to verify programmatically. Fresh LoRALinear weights are
    # zero-init, so the __init__ warm-up generate runs the bare base model.
    model = VoxCPM.from_pretrained(
        MODEL_PATH,
        load_denoiser=False,  # no reference-audio denoising needed here
        optimize=True,        # torch.compile: slow first start, fast steady state
        lora_config=lora_config,
    )
    loaded = []
    if lora_config is not None:
        loaded, skipped = model.load_lora(lora_path)
        if not loaded or skipped:
            raise RuntimeError(
                f"LoRA load failed verification for {name}: "
                f"{len(loaded)} keys loaded, {len(skipped)} skipped "
                f"(first: {skipped[:3]})"
            )
        print(
            f"[voxcpm-tts] LoRA active for {name}: {len(loaded)} parameters "
            f"loaded, 0 skipped",
            flush=True,
        )
    MODELS[name] = model
    print(
        f"[voxcpm-tts] model for {name} loaded from {MODEL_PATH} "
        f"(load+compile {time.time()-t0:.1f}s, "
        f"lora={'on' if lora_config is not None else 'off'})",
        flush=True,
    )
    _build_prompt_cache(name, model)
    return loaded


def load_model():
    """Load every configured checkpoint (+ LoRA) and lock anchor voices."""
    global MODEL, MODEL_SR

    if not LORA_SPECS:
        # bare base model — one shared instance, addressable under the
        # default name so request routing is uniform
        LORA_SPECS.append((DEFAULT_LORA, ""))
    for name, path in LORA_SPECS:
        _load_one(name, path)

    MODEL = MODELS[DEFAULT_LORA] if DEFAULT_LORA in MODELS else next(iter(MODELS))
    MODEL_SR = MODEL.tts_model.sample_rate
    print(
        f"[voxcpm-tts] ready: {len(MODELS)} checkpoint(s) "
        f"({', '.join(MODELS)}), default={DEFAULT_LORA}, sr={MODEL_SR}",
        flush=True,
    )


def _resolve_lora(name):
    """Request lora name -> a name actually in MODELS. Unknown/missing names
    clamp to the default (which itself falls back to whatever is loaded when
    the default checkpoint was dropped via env)."""
    if name in MODELS:
        return name
    if DEFAULT_LORA in MODELS:
        return DEFAULT_LORA
    return next(iter(MODELS))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "loaded": bool(MODELS),
        "sample_rate": OUT_RATE,
        "model_sample_rate": MODEL_SR,
        "lora": {
            name: {"anchored": cache is not None}
            for name, cache in PROMPT_CACHES.items()
        },
        "default_lora": DEFAULT_LORA if DEFAULT_LORA in MODELS
        else next(iter(MODELS), None),
        "text_norm": TEXT_NORM,
    }


@app.post("/synthesize")
def synthesize(req: SynthRequest):
    text = (req.input or "").strip()
    if not text or not MODELS:
        return StreamingResponse(iter(()), media_type="audio/L16")

    # per-request checkpoint routing (unknown names clamp to the default)
    lora_name = _resolve_lora(req.lora)
    if req.lora is not None and req.lora != lora_name:
        print(
            f"[voxcpm-tts] WARNING: unknown lora {req.lora!r} — "
            f"using {lora_name!r}",
            flush=True,
        )
    model = MODELS[lora_name]
    prompt_cache = PROMPT_CACHES.get(lora_name)

    # per-request tunables (clamped; None -> server defaults)
    cfg = _clamp(req.cfg_value, 0.5, 5.0, 2.0) if req.cfg_value is not None else 2.0
    steps = (
        int(_clamp(req.inference_timesteps, 4, 40, 10))
        if req.inference_timesteps is not None
        else 10
    )

    # parity with the facade's _generate text cleanup
    text = re.sub(r"\s+", " ", text.replace("\n", " "))

    # text normalization (digits/symbols/units -> speakable words). The
    # facade's generate(normalize=True) runs this exact class; the
    # prompt-cache fast path has no such knob, so we do it ourselves —
    # covers both reference mode and the zero-shot fallback below.
    if TEXT_NORM:
        text = _normalize_text(text)

    # queue carries raw pcm bytes, ("err", msg) tuples, or _DONE.
    # Unbounded on purpose: bounded by one short LLM segment (~few hundred
    # KB); a bounded queue could block the producer on a dead reader.
    q = queue.Queue()
    cancel = threading.Event()

    def producer():
        gen = None
        try:
            with GEN_LOCK:
                if cancel.is_set() or not MODELS:
                    return
                rs = soxr.ResampleStream(MODEL_SR, OUT_RATE, 1, dtype="float32")
                if prompt_cache is not None:
                    # reference mode: every segment reuses the anchor cache
                    # (the endpoint core._generate funnels into this same
                    # private call — pinned voxcpm 2.0.3, re-check on upgrade)
                    gen = model.tts_model._generate_with_prompt_cache(
                        target_text=text,
                        prompt_cache=prompt_cache,
                        cfg_value=cfg,
                        inference_timesteps=steps,
                        streaming=True,
                    )
                else:
                    # zero-shot fallback (anchor failed): voice_desc prefix
                    # is the only timbre steering we have left
                    voice = (req.voice_desc or "").strip()[:200] or VOICE_DESC
                    gen = model.generate_streaming(
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
