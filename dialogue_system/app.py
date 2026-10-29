import re
import os
import time
import soxr
import queue
import base64
import faulthandler
import logging
import signal
import threading
import uuid
import json
import asyncio
import numpy as np
from typing import Dict, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState

# SIGUSR1 dumps all thread tracebacks to stderr (web.log) — use when the
# app looks wedged:  kill -USR1 <pid>
faulthandler.register(signal.SIGUSR1)


def dump_asyncio_tasks(*_args):
    """SIGUSR2: log where every asyncio task is suspended (cr_frame chain).

    faulthandler cannot see suspended coroutines, but the coroutine objects
    can be walked: task.coro -> cr_frame (suspended line) -> cr_await (what
    it awaits, following Tasks/Futures). Run:  kill -USR2 <pid>
    """
    try:
        tasks = asyncio.all_tasks()
    except RuntimeError:
        return
    logger.warning(f"==== asyncio task dump ({len(tasks)} tasks) ====")

    def walk(awaitable, depth=0):
        chain = []
        node = awaitable
        while node is not None and depth < 20:
            if isinstance(node, asyncio.Task):
                chain.append(f"<Task {node.get_name()}>")
                node = node.get_coro()
            elif isinstance(node, asyncio.Future):
                chain.append(f"<Future done={node.done()}>")
                node = None
            else:
                frame = getattr(node, "cr_frame", None)
                if frame is not None:
                    chain.append(
                        f"{frame.f_code.co_name}"
                        f"@{frame.f_code.co_filename.rsplit('/', 1)[-1]}"
                        f":{frame.f_lineno}"
                    )
                node = getattr(node, "cr_await", None)
            depth += 1
        return chain

    for task in tasks:
        chain = walk(task.get_coro())
        logger.warning(
            f"TASK {task.get_name()} done={task.done()} "
            f"suspended: {' <- '.join(chain) if chain else '?'}"
        )


signal.signal(signal.SIGUSR2, dump_asyncio_tasks)

from clients.tts_client import (
    IndexTTS_VLLM,
    Cosyvoice_Streaming_VLLM,
    FireRedTTS,
    VoxCPM_TTS,
)
from clients.llm_client import QwenLLM_stream
from clients.vad_client import TurnTaking
from clients.pvad_client import PVADSession
from modules.utils.backchannel_utils import check_backchannel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI()


# Headers required for SharedArrayBuffer
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    return response


# Global event loop reference for thread-safe websocket sending
main_loop = None


@app.on_event("startup")
async def startup_event():
    global main_loop
    main_loop = asyncio.get_running_loop()


class Config:
    """Global configuration constants."""

    SAMPLE_RATE = 16000
    VAD_POOL_SIZE = 10
    PORT = 8090


class ChatSession:
    """Manages the full lifecycle of a single user session."""

    def __init__(self, client_id, vad_instance, websocket: WebSocket):
        self.client_id = client_id
        self.vad = vad_instance
        self.websocket = websocket
        self.lock = threading.Lock()
        self._stop_event = threading.Event()  # Internal event to signal interruption
        self.is_active = True

        # Audio config
        self.input_sample_rate = Config.SAMPLE_RATE

        # Live settings (web UI panel) — see DEFAULT_SETTINGS
        self.settings = dict(DEFAULT_SETTINGS)

        # State for pending message commitment (handling interruption)
        self.pending_message = None
        self.pending_audio_duration = 0.0
        self.pending_start_time = None
        self.interruption_time = None

        # Main-speaker gate (pvad sidecar). The feeder thread inside
        # PVADSession is fully async — feed() is a queue put, so the turn
        # controller below never waits on the pvad service.
        self.pvad = PVADSession(client_id)
        self.pvad_checking = False           # barge-in identity check in flight
        self.pvad_enroll_pending = False     # re-try enrollment on next turn
        self._pvad_last_status = 0.0
        # Mic ring for enrollment audio (float32 @16k chunks). Own lock:
        # session.lock is held across vad.process and must not be contended.
        self.ring_lock = threading.Lock()
        self.mic_ring = []                 # [(ts, np.ndarray)]

    @property
    def stop_event(self):
        return self._stop_event

    def interrupt(self):
        """Interrupts current inference or audio playback."""
        self._stop_event.set()
        emit_to_room(self.client_id, "stop_audio", {"message": "interrupt"})
        emit_to_room(self.client_id, "circle_status", {"status": "LISTENING"})

    def pause(self):
        """Pauses audio playback."""
        emit_to_room(self.client_id, "pause_audio", {"message": "pause"})
        emit_to_room(self.client_id, "circle_status", {"status": "LISTENING"})

    def reset_interrupt(self):
        """Creates a new stop event for the next processing cycle, leaving the old one set."""
        self._stop_event = threading.Event()


class SessionManager:
    """Thread-safe manager for active chat sessions."""

    def __init__(self):
        self.sessions: Dict[str, ChatSession] = {}
        self._lock = threading.Lock()

    def create_session(self, client_id, vad_instance, websocket: WebSocket):
        with self._lock:
            session = ChatSession(client_id, vad_instance, websocket)
            self.sessions[client_id] = session
            return session

    def get_session(self, client_id) -> Optional[ChatSession]:
        return self.sessions.get(client_id)

    def remove_session(self, client_id):
        with self._lock:
            return self.sessions.pop(client_id, None)


class VADModelPool:
    """Object pool for VAD instances to optimize memory and startup time."""

    def __init__(self, model_cls, size=Config.VAD_POOL_SIZE):
        self.pool = queue.Queue(maxsize=size)
        logger.info(f"Initializing VAD Pool with {size} instances...")
        for _ in range(size):
            # Initialize instances without specific callbacks (bound during acquisition)
            instance = model_cls(
                status_callback=None, transcription_callback=None, circle_callback=None
            )
            self.pool.put(instance)

    def acquire(self):
        """Retrieve a VAD instance from the pool."""
        return self.pool.get(block=True)

    def release(self, instance):
        """Reset and return the instance back to the pool."""
        if hasattr(instance, "reset"):
            try:
                instance.reset()
            except Exception as exc:
                # reset() itself tolerates a failed reconnect, but never let
                # a release-time error drop the instance: the pool would
                # silently shrink until acquire() blocks forever.
                logger.error(f"VAD pool reset failed: {exc}")
        # Clear callbacks to prevent memory leaks or stale context
        instance.status_callback = None
        instance.transcription_callback = None
        instance.circle_callback = None
        self.pool.put(instance)


# ==== Global Singleton Initialization ====
vad_pool = VADModelPool(TurnTaking)
session_manager = SessionManager()
llm = QwenLLM_stream()
# Both TTS engines instantiated — the active one is chosen per session from
# the web UI settings panel. Engine choice on the command line (TTS_ENGINE
# env) only sets the DEFAULT_SETTINGS default below.
TTS_ENGINE = os.environ.get("TTS_ENGINE", "voxcpm")
TTS_CLIENTS = {
    "voxcpm": VoxCPM_TTS(),   # VoxCPM2 service on :8085
    "firered": FireRedTTS(),  # FireRedTTS on :8081 (shared with full_duplex)
}
asr = None  # Placeholder for ASR client if transcription isn't handled within VAD

# Per-session live settings (web UI "Settings" panel, applied via the
# update_config WS event; no restart needed). Mirrored by frontend
# DEFAULT_PERSONA / currentSettings() in script.js — keep in sync.
DEFAULT_SETTINGS = {
    "system_prompt": QwenLLM_stream.SYSTEM_PROMPT,
    "max_history": 5,          # LLM memory turns
    "min_seg_len": 10,         # chars before a segment is sent to TTS
    "tts_engine": TTS_ENGINE,  # voxcpm | firered
    "tts_cfg": 2.0,            # VoxCPM2 expressiveness
    "tts_timesteps": 10,       # VoxCPM2 quality steps
    "tts_voice_desc": "",      # "" = server default voice
    # Main-speaker gate (pvad sidecar :8086). off = ignore; shadow = evaluate
    # and report but never block; enforce = reject non-main-speaker turns.
    "pvad_mode": "enforce",
    "pvad_threshold": 0.50,    # ECAPA cosine vs enrolled speaker must be
                               # above this (measured: same 0.59-0.80,
                               # different 0.06-0.44)
}

print("System initialized: VAD Pool, LLM client, TTS clients ready.")


def emit_to_room(client_id, event, data):
    """Helper function to safely emit WebSocket messages to a specific client."""
    session = session_manager.get_session(client_id)
    if not session or not main_loop:
        return

    ws = session.websocket

    if ws.client_state != WebSocketState.CONNECTED:
        return

    # Helper wrapper to run async send in the main loop
    async def _send():
        try:
            if event == "audio_chunk":
                # Audio data: send raw bytes
                await ws.send_bytes(data)
            else:
                # Text/JSON data
                message = json.dumps({"event": event, "data": data})
                await ws.send_text(message)
        except Exception as e:
            logger.error(f"Failed to send to {client_id}: {e}")

    asyncio.run_coroutine_threadsafe(_send(), main_loop)


# ==== Main-speaker gate (pvad sidecar) ====
# Advisory only: the turn controller still runs byte-identically; the gate
# is consulted at the two decision points (barge-in, turn completion) and
# can only *veto* acting on a decision, never delay or alter the stream.
# Identity signal = ECAPA cosine vs the enrolled embedding (measured
# separation: same speaker 0.59-0.80 vs different 0.06-0.44; the pvad ONNX
# prob itself is kept as live activity telemetry only). Unknown signal
# (not enrolled / service down / too little audio) => fail open.

_PVAD_RING_MAX_S = 30.0   # mic ring retention
_PVAD_SEARCH_S = 8.0      # how far back to look for speech
_PVAD_CHUNK_RMS = 0.015   # a 160 ms chunk counts as speech above this
_PVAD_MAX_SPEECH_S = 5.0  # cap on the speech audio sent for embedding
_PVAD_MIN_SPEECH_S = 1.0  # less speech than this => nothing to identify


def _pvad_ring_append(session, wav) -> None:
    with session.ring_lock:
        session.mic_ring.append((time.time(), wav))
        # Trim to the retention window (chunk count ~ duration/160ms).
        max_chunks = int(_PVAD_RING_MAX_S / 0.16) + 8
        if len(session.mic_ring) > max_chunks:
            del session.mic_ring[: len(session.mic_ring) - max_chunks]


def _pvad_ring_speech(session):
    """Speech-only audio from the mic ring: the 160 ms chunks above the
    RMS floor within the last 8 s, concatenated (up to ~5 s). Returns
    (wav, speech_seconds); (None, 0.0) when the ring holds no speech.

    A plain max-RMS window is not enough: a decision landing in a silence
    gap yields a window of 4.5 s zeros + a speech sliver — RMS passes any
    sane floor, but ECAPA on silence-dominated audio is garbage (measured:
    the main speaker scored 0.16). Speech-chunk concatenation gives the
    embedding real voice or nothing."""
    with session.ring_lock:
        if not session.mic_ring:
            return None, 0.0
        wav = np.concatenate([c for _, c in session.mic_ring])
    recent = wav[-int(_PVAD_SEARCH_S * Config.SAMPLE_RATE):]
    n = (recent.shape[0] // 2560) * 2560
    if n == 0:
        return None, 0.0
    chunks = recent[:n].reshape(-1, 2560)
    rms = np.sqrt(np.mean(chunks * chunks, axis=1))
    speech = chunks[rms >= _PVAD_CHUNK_RMS]
    if speech.shape[0] == 0:
        return None, 0.0
    keep = int(_PVAD_MAX_SPEECH_S / 0.16)
    return np.concatenate(speech[-keep:]), speech.shape[0] * 0.16


def _pvad_score(session):
    """Blocking ECAPA identity check — worker threads only. Returns
    (cos, threshold); cos is None when unusable (fail-open): unenrolled,
    service down, or less than ~1 s of speech in the recent ring."""
    threshold = session.settings.get(
        "pvad_threshold", DEFAULT_SETTINGS["pvad_threshold"]
    )
    if not session.pvad.enrolled:
        return None, threshold
    wav, speech_s = _pvad_ring_speech(session)
    if wav is None or speech_s < _PVAD_MIN_SPEECH_S:
        return None, threshold
    return session.pvad.score(wav), threshold


def _pvad_turn_gate(session) -> bool:
    """First thing in pipeline_worker: a rejected turn must not touch the
    ongoing playback or LLM history. Runs in the worker thread, so the
    ~150 ms ECAPA forward never blocks the audio receive loop."""
    mode = session.settings.get("pvad_mode", DEFAULT_SETTINGS["pvad_mode"])
    if mode == "off":
        return True
    cos, threshold = _pvad_score(session)
    if cos is not None and cos < threshold:
        emit_to_room(
            session.client_id,
            "pvad_reject",
            {
                "kind": "turn",
                "cos": round(cos, 3),
                "threshold": threshold,
                "shadow": mode == "shadow",
            },
        )
        logger.info(
            f"[{session.client_id}] pvad rejected turn: "
            f"cos={cos:.3f} < {threshold} (mode={mode})"
        )
        return mode == "shadow"  # shadow reports but proceeds
    return True


def _pvad_do_bargein(session) -> None:
    if session.interruption_time is None:
        session.interruption_time = time.time()
    session.interrupt()


def _pvad_handle_bargein(session) -> None:
    """Barge-in decision point. Fail-open paths interrupt inline; when
    enrolled, the identity check runs in a daemon thread (the receive loop
    must not wait ~150 ms per barge-in signal — they can fire many times
    per second during overlapping speech). An in-flight check absorbs
    duplicate signals: whichever way it resolves, it acts once."""
    mode = session.settings.get("pvad_mode", DEFAULT_SETTINGS["pvad_mode"])
    if mode == "off" or not session.pvad.enrolled:
        _pvad_do_bargein(session)
        return
    if session.pvad_checking:
        return
    session.pvad_checking = True

    def _worker():
        try:
            cos, threshold = _pvad_score(session)
            allow = cos is None or cos >= threshold or mode == "shadow"
            if cos is not None and cos < threshold:
                emit_to_room(
                    session.client_id,
                    "pvad_reject",
                    {
                        "kind": "bargein",
                        "cos": round(cos, 3),
                        "threshold": threshold,
                        "shadow": mode == "shadow",
                    },
                )
                logger.info(
                    f"[{session.client_id}] pvad rejected barge-in: "
                    f"cos={cos:.3f} < {threshold} (mode={mode})"
                )
            if allow:
                _pvad_do_bargein(session)
        finally:
            session.pvad_checking = False

    threading.Thread(target=_worker, daemon=True).start()


def _pvad_try_enroll(session, reason: str) -> None:
    """Enroll the main speaker from the mic ring tail. Blocking HTTP runs
    in a daemon thread; defers to the next gate-passed turn if the ring is
    still too short (<3 s) or the service errors."""

    def _worker():
        wav, speech_s = _pvad_ring_speech(session)
        if wav is None or speech_s < 3.0:
            session.pvad_enroll_pending = True
            logger.info(
                f"[{session.client_id}] pvad enroll deferred ({reason}): "
                f"ring holds {speech_s:.1f}s of speech (need 3s)"
            )
            return
        ok = session.pvad.enroll(wav)
        if ok:
            session.pvad_enroll_pending = False
            logger.info(
                f"[{session.client_id}] pvad enrolled ({reason}, "
                f"{speech_s:.1f}s speech)"
            )
        else:
            session.pvad_enroll_pending = True

    threading.Thread(target=_worker, daemon=True).start()


def _pvad_maybe_auto_enroll(session) -> None:
    """After a gate-passed turn: enroll if not yet enrolled (first turn
    wins — the person the system actually conversed with becomes the main
    speaker) or if a previous enrollment attempt was deferred."""
    if session.pvad.enrolled and not session.pvad_enroll_pending:
        return
    _pvad_try_enroll(session, "auto")


def _pvad_emit_status(session) -> None:
    """Throttled live status for the UI Speaker Gate panel (~1/1.5s)."""
    now = time.time()
    if now - session._pvad_last_status < 1.5:
        return
    session._pvad_last_status = now
    emit_to_room(
        session.client_id,
        "pvad_status",
        {
            "enrolled": session.pvad.enrolled,
            "prob": session.pvad.latest_prob(),
            "mode": session.settings.get("pvad_mode", DEFAULT_SETTINGS["pvad_mode"]),
            "threshold": session.settings.get(
                "pvad_threshold", DEFAULT_SETTINGS["pvad_threshold"]
            ),
        },
    )


def pipeline_worker(client_id, audio_segment, sample_rate):
    """
    Main processing pipeline: ASR -> LLM -> TTS.
    Runs in a background thread for each detected utterance.
    """
    session = session_manager.get_session(client_id)
    if not session:
        return

    # Main-speaker gate FIRST: a rejected turn must not interrupt the
    # ongoing playback, touch LLM history, or cost any LLM/TTS work.
    if not _pvad_turn_gate(session):
        return
    _pvad_maybe_auto_enroll(session)

    try:
        # 1. ASR Phase (Automatic Speech Recognition)
        if asr is None:
            # Fallback if ASR is handled by internal TurnTaking module
            asr_text = (
                audio_segment if isinstance(audio_segment, str) else "Voice Detected"
            )
        else:
            asr_text = asr.recognize(audio_segment, sample_rate)

        if not asr_text.strip():
            return

        # # 2. Backchannel Detection (vad already handles this, but double-check here)
        # # Checks for filler words or short backchannels that shouldn't trigger a full response
        # if check_backchannel(asr_text):
        #     emit_to_room(client_id, "resume_audio", {"message": "backchannel detected"})
        #     return

        # Process Pending Message from previous turn if any (before adding new user message)
        with session.lock:
            if session.pending_message:
                final_msg = session.pending_message
                # Check if an interruption occurred during the playback of the previous complete message
                if (
                    session.interruption_time
                    and session.pending_start_time
                    and session.pending_audio_duration > 0
                ):
                    elapsed = max(
                        session.interruption_time - session.pending_start_time, 0
                    )
                    ratio = min(elapsed / session.pending_audio_duration, 1.0)
                    cutoff = int(len(final_msg) * ratio)
                    final_msg = final_msg[:cutoff]
                    logger.info(
                        f"[{client_id}] Previous turn, Ratio: {ratio:.2f}, Truncated: {final_msg}"
                    )
                else:
                    logger.info(f"[{client_id}] Previous turn completed fully.")

                llm.add_message(client_id, "assistant", final_msg)

            # Clear pending state
            session.pending_message = None
            session.pending_audio_duration = 0.0
            session.pending_start_time = None
            session.interruption_time = None

        # 3. Preparation for Response
        session.interrupt()  # Signal previous threads to stop
        session.reset_interrupt()  # Create a fresh event for this new thread

        # Capture the specific stop_event for this execution cycle
        current_stop_event = session.stop_event

        if current_stop_event.is_set():
            return

        logger.info(f"[{client_id}] ASR result: {asr_text}")
        emit_to_room(client_id, "user_transcription", {"text": asr_text})
        # Let the frontend clear the previous assistant line
        emit_to_room(client_id, "turn_start", {})

        # Snapshot live settings for this pipeline run (user may tweak the
        # UI mid-generation; one consistent view per turn is enough)
        settings = dict(session.settings)
        tts_client = TTS_CLIENTS.get(
            settings.get("tts_engine"), TTS_CLIENTS[DEFAULT_SETTINGS["tts_engine"]]
        )
        tts_options = {
            "cfg_value": settings.get("tts_cfg", 2.0),
            "inference_timesteps": settings.get("tts_timesteps", 10),
            "voice_desc": settings.get("tts_voice_desc", ""),
        }

        llm.add_message(client_id, "user", asr_text)
        llm_reply_gen = llm.generate_with_history(
            client_id,
            stop_event=current_stop_event,
            system_prompt=settings.get("system_prompt") or None,
            max_history=settings.get("max_history"),
            min_seg_len=settings.get("min_seg_len"),
        )
        # logger.info(llm.get_session(client_id))

        # 4. LLM & TTS Streaming Processing
        # Iterate over LLM response chunks and synthesize audio on the fly
        message_to_add = ""
        interrupted = False
        first_emit_time = None
        total_audio_duration = 0.0

        for i, chunk in enumerate(
            llm_reply_gen
            if hasattr(llm_reply_gen, "__iter__") and not isinstance(llm_reply_gen, str)
            else [llm_reply_gen]
        ):
            if current_stop_event.is_set():
                interrupted = True
                break

            logger.info(f"[{client_id}] LLM Chunk: {chunk}")

            # Send LLM text chunk immediately
            emit_to_room(client_id, "text_response", {"text": chunk})

            # Accumulate text before TTS loop to ensure current chunk is considered
            message_to_add += chunk

            # Synthesize text chunk to speech (Iterate over int16 pcm chunks)
            for wav_chunk in tts_client.synthesize(
                chunk, streaming=True, options=tts_options
            ):
                if current_stop_event.is_set():
                    interrupted = True
                    break

                if first_emit_time is None:
                    first_emit_time = time.time()

                # Calculate audio duration: bytes / (sample_rate * channels * bytes_per_sample)
                # Assuming 24k sample rate, 1 channel, 16-bit (2 bytes) = 48000 bytes/sec
                total_audio_duration += len(wav_chunk) / 48000.0

                emit_to_room(client_id, "audio_chunk", wav_chunk)

            if current_stop_event.is_set():
                interrupted = True
                break

        if interrupted:
            # If interrupted mid-stream, calculate truncation immediately using current time
            if first_emit_time and total_audio_duration > 0:
                elapsed_time = max(session.interruption_time - first_emit_time, 0)
                ratio = min(elapsed_time / total_audio_duration, 1.0)
                cutoff_length = int(len(message_to_add) * ratio)
                message_to_add = message_to_add[:cutoff_length]
                logger.info(
                    f"[{client_id}] Interrupted mid-stream. Ratio: {ratio:.2f}. Truncated message: {message_to_add}"
                )
            else:
                message_to_add = ""

            # Commit immediately as this pipeline execution is dead
            llm.add_message(client_id, "assistant", message_to_add)

            # Ensure no pending state is left over
            with session.lock:
                session.pending_message = None
                session.pending_audio_duration = 0.0
                session.pending_start_time = None
                session.interruption_time = None

        else:
            # Pipeline finished successfully, but user might interrupt later while audio is playing.
            # Do NOT commit to LLM yet. Save to session pending state.
            with session.lock:
                session.pending_message = message_to_add
                session.pending_audio_duration = total_audio_duration
                session.pending_start_time = first_emit_time
                session.interruption_time = None

    except Exception as e:
        logger.error(f"Error in pipeline for {client_id}: {e}", exc_info=True)


# ==== WebSocket Endpoint & Processing ====

# Validation for update_config: (key, type, lo, hi) — ints/floats clamped to
# [lo, hi]; strings truncated to hi chars. Anything else in the payload is
# silently dropped so the UI can send a superset safely.
_SETTING_RULES = {
    "max_history": (int, 1, 20),
    "min_seg_len": (int, 4, 40),
    "tts_timesteps": (int, 4, 40),
    "tts_cfg": (float, 0.5, 5.0),
    "system_prompt": (str, 0, 4000),
    "tts_voice_desc": (str, 0, 200),
    "pvad_threshold": (float, 0.3, 0.9),
}
_TTS_ENGINES = ("voxcpm", "firered")
_PVAD_MODES = ("off", "shadow", "enforce")


def _merge_session_settings(session, data: dict) -> dict:
    """Validate + merge a partial settings update into session.settings."""
    if not isinstance(data, dict):
        return dict(session.settings)

    for key, (typ, lo, hi) in _SETTING_RULES.items():
        if key not in data:
            continue
        value = data[key]
        if typ in (int, float):
            try:
                value = typ(value)
            except (TypeError, ValueError):
                continue
            session.settings[key] = max(lo, min(hi, value))
        else:  # str
            if not isinstance(value, str):
                continue
            session.settings[key] = value[:hi]

    engine = data.get("tts_engine")
    if engine in _TTS_ENGINES:
        session.settings["tts_engine"] = engine

    mode = data.get("pvad_mode")
    if mode in _PVAD_MODES:
        session.settings["pvad_mode"] = mode

    return dict(session.settings)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_id = str(uuid.uuid4())
    logger.info(f"New connection: {client_id}")

    session = None

    try:
        # Initial Handshake / Setup
        emit_to_room(
            client_id,
            "vad_loading",
            {"state": "loading", "message": "Acquiring model resources..."},
        )

        # We need a session object to put in session_manager before emit_to_room works fully.
        loop = asyncio.get_running_loop()
        vad_instance = await loop.run_in_executor(None, vad_pool.acquire)

        # Bind Callbacks
        vad_instance.status_callback = lambda s, m: emit_to_room(
            client_id, "vad_status", {"state": s, "message": m}
        )
        vad_instance.transcription_callback = lambda t: emit_to_room(
            client_id, "user_transcription", {"text": t}
        )
        vad_instance.circle_callback = lambda s: emit_to_room(
            client_id, "circle_status", {"status": s}
        )

        session = session_manager.create_session(client_id, vad_instance, websocket)

        # Now emission works via session_manager lookups
        emit_to_room(client_id, "connect_ack", {"client_id": client_id})
        emit_to_room(
            client_id,
            "vad_loading",
            {"state": "ready", "message": "Model loaded, ready to experience"},
        )
        logger.info(f"Session initialized for {client_id}")

        n_chunks = 0
        while True:
            # Receive Message
            try:
                message = await websocket.receive()
            except WebSocketDisconnect:
                logger.info(f"Client disconnected: {client_id}")
                break

            n_chunks += 1
            if n_chunks % 50 == 0:
                logger.info(f"[{client_id}] received {n_chunks} messages")

            if "bytes" in message and message["bytes"]:
                # Binary Audio Data
                data = message["bytes"]
                # Convert buffer to float32
                audio_chunk = (
                    np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                )

                current_sr = session.input_sample_rate
                if current_sr != Config.SAMPLE_RATE:
                    audio_chunk = soxr.resample(
                        audio_chunk, current_sr, Config.SAMPLE_RATE, quality="VHQ"
                    )

                # PVAD sidecar: async copy of the mic chunk + enrollment ring.
                # Both are outside session.lock so the turn-controller path
                # below is untouched (feed() is a queue put; ring append is
                # an O(1) list append under its own short-lived lock).
                session.pvad.feed(audio_chunk)
                _pvad_ring_append(session, audio_chunk)
                _pvad_emit_status(session)

                t_proc = time.time()
                with session.lock:
                    segment = session.vad.process(audio_chunk)
                dt_proc = time.time() - t_proc
                if dt_proc > 0.3:
                    logger.warning(
                        f"[{client_id}] vad.process took {dt_proc:.2f}s "
                        f"(expected ~0.03s)"
                    )

                if segment is not None:
                    if isinstance(segment, list) and segment[0] is None:
                        # Barge-in — main-speaker gate runs in a worker
                        # thread (see _pvad_handle_bargein)
                        _pvad_handle_bargein(session)
                    else:
                        # Complete Utterance — pipeline_worker applies the
                        # main-speaker gate as its first step
                        threading.Thread(
                            target=pipeline_worker,
                            args=(client_id, segment, Config.SAMPLE_RATE),
                            daemon=True,
                        ).start()

            elif "text" in message and message["text"]:
                # JSON Control Message
                try:
                    payload = json.loads(message["text"])
                    event = payload.get("event")

                    if event == "duplex_stop":
                        if session.interruption_time is None:
                            session.interruption_time = time.time()
                        session.stop_event.set()
                        session.reset_interrupt()
                        logger.info(f"Session manually stopped by client: {client_id}")

                    elif event == "pvad_enroll":
                        # Manual (re-)enrollment from the UI "Speaker Gate"
                        # panel: uses the mic ring tail if it already has
                        # >= 3 s of audio, else defers to the next turn.
                        logger.info(f"[{client_id}] pvad re-enroll requested")
                        _pvad_try_enroll(session, "manual")

                    elif event == "ping":
                        # App-level heartbeat from the frontend: a dead or
                        # half-dead TCP link is invisible to the browser
                        # (send() succeeds, readyState stays OPEN), so the
                        # client times out if this reply stops arriving.
                        emit_to_room(client_id, "pong", {"ts": time.time()})

                    elif event == "config_audio":
                        # Client sending sample rate configuration
                        sr_data = payload.get("data", {})
                        sr = sr_data.get("sample_rate")
                        if sr:
                            session.input_sample_rate = int(sr)
                            logger.info(f"[{client_id}] Sample rate set to {sr}")

                    elif event == "update_config":
                        # Live settings from the web UI panel. Partial updates
                        # are merged; unknown keys ignored; values validated
                        # and clamped so a buggy client cannot break the
                        # pipeline. Echoes the merged settings back so the UI
                        # can confirm what actually landed.
                        data = payload.get("data", {})
                        merged = _merge_session_settings(session, data)
                        emit_to_room(client_id, "config_applied", {"settings": merged})
                        logger.info(f"[{client_id}] Settings updated: {merged}")

                except json.JSONDecodeError:
                    logger.warning(f"[{client_id}] Received invalid JSON")

    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {client_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        if session:
            session.is_active = False
            session.stop_event.set()
            session.pvad.stop()
            vad_pool.release(session.vad)
            session_manager.remove_session(client_id)
        logger.info(f"Session cleaned up: {client_id}")


# ==== Static Resource Routing ====
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    logger.info(f"Server starting on http://localhost:{Config.PORT}")
    uvicorn.run(app, host="0.0.0.0", port=Config.PORT)
