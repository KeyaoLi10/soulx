"""PVAD (personalized VAD) microservice for the SoulX web dialogue app.

Wraps FireRedChat's fireredchat-pvad model (ONNX + ECAPA speaker embedding)
as an HTTP sidecar. Runs from full_duplex/venv-frc — the only env with the
livekit-agents fork + speechbrain + onnxruntime the plugin needs (same
cross-venv pattern as ultraVAD running from eot_bench/venv). CPU only.

Purpose in the SoulX architecture: an *advisory* "is this the main speaker?"
signal. The turn controller (SoulX-Duplug :8000) keeps consuming the raw mic
stream untouched; the web app feeds a COPY of each chunk here asynchronously
and uses the returned probability to gate turn/barge-in *decisions*. Nothing
in this service sits on the turn-controller path.

Wire contract:
  GET  /health   -> {"status": "ok", "loaded": bool}
  POST /feed     {"session_id", "audio": base64 float32 LE, "sample_rate"}
                 -> {"prob": float | None, "enrolled": bool}
                    prob = personal-VAD speech probability of the LAST 10 ms
                    frame in this batch (caller keeps the rolling window).
  POST /enroll   {"session_id", "audio", "sample_rate"}
                 -> {"ok": bool, "enrolled": bool, "duration_s": float}
                    Computes an ECAPA embedding from the tail of `audio`
                    (last 5 s, mirroring the plugin's ecapa_window_size)
                    and personalizes the session. Needs >= 2 s of audio.
  POST /score    {"session_id", "audio", "sample_rate"}
                 -> {"cos": float | None, "enrolled": bool}
                    Cosine similarity between the enrolled speaker and the
                    ECAPA embedding of the tail of `audio` (last 5 s).
                    None when unenrolled / too little audio. This is the
                    identity signal the app gates on: measured separation
                    same-speaker ~0.78-0.84 vs different ~0.27-0.52, while
                    the pvad ONNX prob fails hard negatives (podcast
                    channels: diff-speaker speech-mean 0.54 vs main 0.60).
  POST /reset    {"session_id"} -> drops the session state

Concurrency model (same time-slicing as the turn server's engine.py): one
VadProcessor instance shared by all sessions; per-session state is the
(mel_buffer, gru_buffer, spkemb) numpy arrays the ONNX model takes as
inputs, swapped in/out around each call under STATE_LOCK. The ONNX session
itself is stateless, so this is exact — no hidden state to lose.

Usage (see scripts/start_services.sh, tmux window "pvad"):
  HF_HUB_OFFLINE=1 /home/research/krisl/full_duplex/venv-frc/bin/python \
      services/pvad_server.py [--port 8086]
"""

import argparse
import base64
import threading
import time

import numpy as np
import torch
import torchaudio.transforms as T
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

SAMPLE_RATE = 16000
FRAME = 160                 # 10 ms @ 16 kHz — VadProcessor's fixed window
ENROLL_MIN_SAMPLES = 32000  # 2 s minimum for a usable ECAPA embedding
SCORE_MIN_SAMPLES = 8000    # 0.5 s minimum for a scoring embedding — the
                            # barge-in gate scores at the first hysteresis
                            # crossing (~0.5s of speech); requiring 1s there
                            # fail-opens exactly when echo/noise starts
ENROLL_WINDOW = 80000       # 5 s tail, mirrors the plugin's ecapa_window_size
SESSION_IDLE_S = 120        # janitor: drop sessions idle longer than this

PROCESSOR = None            # VadProcessor, loaded at startup
STATE_LOCK = threading.Lock()
SESSIONS = {}               # sid -> {"mel", "gru", "spkemb", "enrolled", "last_seen"}

_resamplers: dict[int, T.Resample] = {}

app = FastAPI(title="pvad")


def load_model():
    """Load the pvad ONNX + ECAPA extractor at startup (from the plugin's
    in-tree resources; weights also mirrored on /mnt/disk3)."""
    global PROCESSOR
    from livekit.plugins.fireredchat_pvad.vad import VadProcessor

    t0 = time.time()
    PROCESSOR = VadProcessor()
    print(
        f"[pvad] model loaded (onnx + ecapa, {time.time()-t0:.1f}s)", flush=True
    )


def _resample(wav: np.ndarray, sr: int) -> np.ndarray:
    if sr not in _resamplers:
        _resamplers[sr] = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
    return _resamplers[sr](torch.from_numpy(wav)).numpy()


def _decode(audio_b64: str, sr: int) -> np.ndarray:
    raw = base64.b64decode(audio_b64)
    wav = np.frombuffer(raw, dtype="<f4").astype(np.float32)
    if sr != SAMPLE_RATE:
        wav = _resample(wav, sr)
    return np.clip(wav, -1.0, 1.0)


def _session(sid: str) -> dict:
    """Get or create per-session ONNX state (zero spkemb = unpersonalized)."""
    st = SESSIONS.get(sid)
    if st is None:
        st = {
            "mel": np.zeros((1, 80, 15), dtype=np.float32),
            "gru": np.zeros((2, 1, 256), dtype=np.float32),
            "spkemb": np.zeros((1, 192), dtype=np.float32),
            "enrolled": False,
            "last_seen": time.time(),
        }
        SESSIONS[sid] = st
    st["last_seen"] = time.time()
    return st


class FeedRequest(BaseModel):
    session_id: str
    audio: str            # base64 float32 LE
    sample_rate: int = SAMPLE_RATE


class EnrollRequest(FeedRequest):
    pass


class ResetRequest(BaseModel):
    session_id: str


def _ecapa_tail(wav: np.ndarray):
    """L2-normalized ECAPA embedding of the last 5 s, as (192,) numpy."""
    window = np.ascontiguousarray(wav[-ENROLL_WINDOW:])
    audio = torch.from_numpy(window).unsqueeze(0)
    with STATE_LOCK:
        emb = PROCESSOR._spk_extractor.get_embedding(audio)  # (1, 192)
    return emb.reshape(-1)


@app.get("/health")
def health():
    return {"status": "ok", "loaded": PROCESSOR is not None, "sessions": len(SESSIONS)}


@app.post("/feed")
def feed(req: FeedRequest):
    if PROCESSOR is None:
        return {"prob": None, "enrolled": False}
    wav = _decode(req.audio, req.sample_rate)
    st = _session(req.session_id)

    n = (wav.shape[0] // FRAME) * FRAME
    if n == 0:
        return {"prob": None, "enrolled": st["enrolled"]}

    prob = None
    with STATE_LOCK:
        p = PROCESSOR
        # snapshot/restore: swap this session's state into the shared
        # processor, run the frames, save the evolved buffers back
        p.mel_buffer = st["mel"]
        p.gru_buffer = st["gru"]
        p.spkemb = st["spkemb"]
        for i in range(0, n, FRAME):
            prob = p(wav[i:i + FRAME])
        st["mel"] = p.mel_buffer
        st["gru"] = p.gru_buffer

    return {"prob": float(prob), "enrolled": st["enrolled"]}


@app.post("/enroll")
def enroll(req: EnrollRequest):
    if PROCESSOR is None:
        return {"ok": False, "error": "model not loaded"}
    wav = _decode(req.audio, req.sample_rate)
    if wav.shape[0] < ENROLL_MIN_SAMPLES:
        return {
            "ok": False,
            "error": f"need >= {ENROLL_MIN_SAMPLES // SAMPLE_RATE}s of audio",
        }

    # Mirror the plugin's VADStream.update_speaker: float32/1.0 waveform,
    # torch tensor (1, N), ECAPA embedding L2-normalized by the extractor
    st = _session(req.session_id)
    st["spkemb"] = _ecapa_tail(wav).reshape(1, 192)
    st["enrolled"] = True

    return {"ok": True, "enrolled": True, "duration_s": wav[-ENROLL_WINDOW:].shape[0] / SAMPLE_RATE}


@app.post("/score")
def score(req: EnrollRequest):
    """Identity check: cosine(enrolled embedding, embedding of audio tail).
    Both embeddings are L2-normalized, so the dot product is the cosine."""
    st = SESSIONS.get(req.session_id)
    if st is None or not st["enrolled"] or PROCESSOR is None:
        return {"cos": None, "enrolled": bool(st and st["enrolled"])}
    wav = _decode(req.audio, req.sample_rate)
    if wav.shape[0] < SCORE_MIN_SAMPLES:
        return {"cos": None, "enrolled": True}
    cos = float(np.dot(st["spkemb"].reshape(-1), _ecapa_tail(wav)))
    return {"cos": cos, "enrolled": True}


@app.post("/reset")
def reset(req: ResetRequest):
    SESSIONS.pop(req.session_id, None)
    return {"ok": True}


def _janitor():
    while True:
        time.sleep(30)
        now = time.time()
        stale = [sid for sid, st in SESSIONS.items()
                 if now - st["last_seen"] > SESSION_IDLE_S]
        for sid in stale:
            SESSIONS.pop(sid, None)
        if stale:
            print(f"[pvad] janitor dropped {len(stale)} idle session(s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8086)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    load_model()
    threading.Thread(target=_janitor, daemon=True).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
