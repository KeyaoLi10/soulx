import re
import time
import soxr
import queue
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
from agent.runtime import AgentRuntime, AGENT_SYSTEM_ADDON, FAST_TOOLS
from frc_engine import FrcEngine, services_available as frc_services_available

# Refactor 2026-10-30: app.py was a ~1960-line monolith. The three cohesive
# layers it shared everything with now live in their own modules:
#   settings.py  — Config / DEFAULT_SETTINGS / settings validation
#   emitter.py   — thread-safe WebSocket push (emit_to_room)
#   pvad_gate.py — the main-speaker gate (all pvad_* entry points)
# The two hooks those modules cannot own (they must not import app back)
# are wired right after the singletons below.
import emitter
from emitter import emit_to_room
import pvad_gate
from pvad_gate import (
    pvad_bank_turn_audio,
    pvad_emit_status,
    pvad_enabled,
    pvad_handle_bargein,
    pvad_maybe_auto_enroll,
    pvad_ring_append,
    pvad_try_enroll,
    pvad_turn_gate,
)
from settings import Config, DEFAULT_SETTINGS, TTS_ENGINE, sanitize_updates

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


@app.on_event("startup")
async def startup_event():
    # the event loop reference lives in emitter.py (thread-safe sending)
    emitter.main_loop = asyncio.get_running_loop()


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
        # Enrollment speech bank: speech audio accumulated across turns.
        # The 8s mic ring alone never reaches the 3s enroll floor when the
        # user speaks in short bursts, so the session stays unenrolled and
        # the barge-in gate fail-opens forever (the choppiness bug of
        # 2026-10-29: every nonidle blip interrupted playback).
        self.pvad_speech_bank = None       # np.ndarray or None (ring_lock)
        # Whether THIS turn's gate pass was identity-VERIFIED (scored
        # cos >= threshold). Gate passes are not all equal: fail-open
        # passes (unenrolled / unscoreable window) are exactly the
        # unverified ones, and banking/re-enrolling them lets echo-class
        # audio displace the enrolled speaker — measured 2026-10-30
        # (session b63c185f): one fail-open phantom turn re-enrolled,
        # after which the real user scored 0.348 -> 0.013 -> -0.017 and
        # every later turn of theirs was rejected while phantoms passed.
        # Set by pvad_turn_gate, read by the bank/enroll calls of the
        # same pipeline_worker run.
        self.pvad_turn_verified = False
        # Gate-decision audio dumps written for this session (forensics
        # cap; see pvad_gate._pvad_dump_forensics)
        self.pvad_forensics_n = 0
        # Barge-in hysteresis: consecutive nonidle chunks. A 1-2 chunk
        # nonidle blip (noise / AEC residue the turn server misclassified)
        # must not stop playback — only sustained nonidle (real speech)
        # may interrupt. Any idle/speak result resets the streak.
        self.bargein_streak = 0
        # The nonidle audio itself (rolling ~2.5 s). The barge-in identity
        # check scores THIS, not the mic ring: the ring's recent-speech
        # window can still hold the user's just-finished turn, which made
        # the first echo/noise onset score as the main speaker.
        self.bargein_audio = []
        # One barge-in episode interrupts once: without this, continuous
        # overlapping speech re-triggers interrupt() every _BARGEIN_MIN_CHUNKS
        # chunks (a stop_audio storm measured at 4x per 2s barge-in).
        # Re-armed by reset_interrupt() at the start of each new turn.
        self.bargein_done = False
        # Contiguous nonidle chunks in the CURRENT voice episode — unlike
        # bargein_streak this is NOT reset when the barge-in trigger fires,
        # so at complete-turn time it measures the whole episode. Blip turns
        # (< _BARGEIN_MIN_CHUNKS of voice before "completing") are echo/noise
        # that paraformer hallucinates short fragments onto ("怎么样""平时"
        # class phantoms, 2026-10-30) — dropped in the websocket loop.
        self.voice_run = 0
        # Wall-clock of the last SUSTAINED voice (>= _BARGEIN_MIN_CHUNKS
        # consecutive nonidle chunks — the same evidence the barge-in
        # hysteresis trusts). The turn server often emits a complete turn's
        # text only AFTER the audio has gone idle (voice_run already reset
        # to 0; measured 2026-10-30: real first turns and split-utterance
        # trailing segments dropped as 0-2-chunk blips), so acceptance
        # falls back to "sustained voice happened moments ago". Echo/noise
        # blips never sustain 3 consecutive chunks, so they never open
        # this window.
        self.last_sustained_voice_ts = 0.0
        # Estimated wall-clock end of assistant playback (audio chunks are
        # buffered client-side ahead of real time). Used to suppress the
        # LISTENING circle flip while the assistant is still talking.
        self.playback_end = 0.0
        # Fast/slow brain: True while a speech generation (user turn OR
        # background-result delivery) is running in this session. The
        # delivery arbiter polls it to pick a speaking slot.
        self.gen_in_progress = False
        # Wall-clock of the last user voice activity (barge-in chunks or a
        # completed turn) — the delivery arbiter waits for a quiet gap.
        self.last_voice_ts = 0.0
        # AgentRuntime (task mgmt + delivery arbiter). Created in the
        # websocket endpoint once the session is registered.
        self.agent_runtime = None
        # FrcEngine (FireRedChat-architecture turn engine, only fed audio
        # while settings turn_engine == "frc"). Created in the websocket
        # endpoint alongside the agent runtime.
        self.frc = None
        # Recent assistant text segments (the LLM chunks that get spoken),
        # newest last, capped. The self-talk check compares a turn's ASR
        # text against these: echo turns transcribe OUR OWN words.
        self.assistant_chunks = []

    @property
    def stop_event(self):
        return self._stop_event

    def interrupt(self):
        """Interrupts current inference or audio playback."""
        self._stop_event.set()
        self.playback_end = 0.0
        emit_to_room(self.client_id, "stop_audio", {"message": "interrupt"})
        emit_to_room(self.client_id, "circle_status", {"status": "LISTENING"})

    def pause(self):
        """Pauses audio playback."""
        emit_to_room(self.client_id, "pause_audio", {"message": "pause"})
        emit_to_room(self.client_id, "circle_status", {"status": "LISTENING"})

    def reset_interrupt(self):
        """Creates a new stop event for the next processing cycle, leaving the old one set."""
        self._stop_event = threading.Event()
        self.bargein_done = False


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
# wire the modules that must not import app back (see the import note above)
emitter.set_session_lookup(session_manager.get_session)
llm = QwenLLM_stream()
pvad_gate.get_history = llm.get_session
# Both TTS engines instantiated — the active one is chosen per session from
# the web UI settings panel (defaults in settings.DEFAULT_SETTINGS; the
# TTS_ENGINE env only picks that default).
TTS_CLIENTS = {
    "voxcpm": VoxCPM_TTS(),   # VoxCPM2 service on :8085
    "firered": FireRedTTS(),  # FireRedTTS on :8081 (shared with full_duplex)
}
asr = None  # Placeholder for ASR client if transcription isn't handled within VAD

print("System initialized: VAD Pool, LLM client, TTS clients ready.")


# ==== Voice frontend / turn-control constants ====
# (extracted 2026-10-30 from the former pvad block; the pvad_* gate itself
# now lives in pvad_gate.py — these are the constants the receive loop and
# pipeline_worker below use directly)

_BARGEIN_MIN_CHUNKS = 3   # consecutive nonidle chunks before a barge-in
                          # acts (3 x 160 ms = 480 ms of sustained speech)
_CHUNK_MS = 160            # one upstream audio chunk (2560 samples @16k) —
                          # converts turn_wait_ms to max_wait_num chunks
# Premature-turn grace (2026-10-30 抢话修复): a "complete"-triggered turn
# boundary fires the reply instantly with zero silence confirmation, so
# mid-sentence pauses get misread as finished turns. The reply generation
# starts immediately but is HELD for this long — if the user resumes
# speech it is dropped silently and their continuation merges with the
# original user message (llm_client groupby). The LLM's first segment
# takes ~1s anyway, so the hold costs almost no perceived latency.
# "wait"-triggered turns need no grace (already silence-verified by
# max_wait_num idle chunks on the turn server).
# Live-tunable per session since 2026-10-30 (settings turn_grace_ms); this
# constant is only the fallback for sessions without the setting.
_TURN_GRACE_S = 0.9
# Mention gate for the Agent Runtime task view (2026-10-30 闲聊不汇报):
# the fast brain only sees "当前后台任务：…" when the user's turn actually
# references the background work. Deliberately broad — a false positive
# merely injects an (accurate) status note, while a false negative has the
# model answering "查得怎么样了" with no idea a task exists.
_TASK_MENTION_RE = re.compile(
    r"查|任务|进度|结果|好了吗|好了没|出来|多久|搞定|完成|办好|弄好|找到|怎么样"
)
_SUSTAINED_VOICE_WINDOW_S = 6.0
                          # a complete turn is accepted if sustained voice
                          # (>= _BARGEIN_MIN_CHUNKS consecutive chunks)
                          # happened within this window — the turn server
                          # often emits the turn text only after the audio
                          # went idle, so voice_run alone reads 0-2 for
                          # real speech
# 2026-10-30 晚：毛刺话轮过滤**整体下线**。实测（logs/web.log 01:31 会话）
# 两条判据全部失效：① 真人麦克风语音经浏览器 WebRTC 处理后，turn server
# 的状态逐块 nonidle/idle 交替（约 50% 占空比，永远没有 3 连续 chunk），
# 任何"连续/近期 chunk 计数"判据都会误杀真话轮（用户整句"喂，你怎么
# 不说话了？"被丢）；② 幻听碎片（"这个怎么样？""平时。"）前的 episode
# 与真实语音同形态（5-6 个 1-chunk / 秒），计数无法区分。防幻听须另找
# 判据（pvad 身份门控方向或音频取证），代码保留待判据成立后启用。
_BLIPTURN_FILTER = False


def _commit_pending_message(session, client_id):
    """Commit the previous generation's assistant message into LLM history,
    truncated to how much of it actually played. Shared by the user-turn
    path and the background-result delivery path."""
    with session.lock:
        if not session.pending_message:
            session.pending_message = None
            session.pending_audio_duration = 0.0
            session.pending_start_time = None
            session.interruption_time = None
            return
        final_msg = session.pending_message
        if (
            session.interruption_time
            and session.pending_start_time
            and session.pending_audio_duration > 0
        ):
            elapsed = max(session.interruption_time - session.pending_start_time, 0)
            ratio = min(elapsed / session.pending_audio_duration, 1.0)
            cutoff = int(len(final_msg) * ratio)
            final_msg = final_msg[:cutoff]
            logger.info(f"[{client_id}] Previous turn, Ratio: {ratio:.2f}, Truncated: {final_msg}")
        else:
            logger.info(f"[{client_id}] Previous turn completed fully.")
        session.pending_message = None
        session.pending_audio_duration = 0.0
        session.pending_start_time = None
        session.interruption_time = None
    llm.add_message(client_id, "assistant", final_msg)


def _run_speech_pipeline(session, client_id, llm_reply_gen, current_stop_event,
                         tts_client, tts_options, grace_until=None):
    """Stream one LLM generation to the user: text events + TTS audio +
    pending-message bookkeeping (interruption truncation). Returns True if
    the generation ran to completion, False if interrupted.

    Used by BOTH speech outlets — the user-turn pipeline and the
    background-result delivery — so the whole system has exactly one way
    of speaking (one turn_start, one pending-message lifecycle).

    grace_until: premature-turn hold deadline (monotonic wall clock,
    time.time()). A "complete"-triggered turn boundary fires the reply the
    instant the turn model sees fit — no silence confirmation — so a
    mid-sentence pause gets misread as a finished turn and the assistant
    barges in ("抢话"). While the grace window is open the LLM keeps
    streaming into a hold buffer (its first segment takes ~1s anyway, so
    the hold is nearly free) and NOTHING reaches the user. If the user
    resumes talking, the whole generation is dropped silently — their
    continuation arrives as another user message and llm_client's groupby
    merges the two into one utterance, so the next attempt sees the full
    sentence. Only when the window passes without resumed speech is the
    held text flushed through the normal emit path."""
    holding = grace_until is not None
    # last_voice_ts at hold start: any later bump means sustained voice
    # resumed (blips deliberately never move it — see the receive loop)
    grace_ref_ts = session.last_voice_ts if holding else 0.0
    premature = False
    # Segments held back while the grace window is open. Kept in a list —
    # NOT folded into message_to_add — so a flush speaks each one exactly
    # once through the normal emit path (an early version appended held
    # text to message_to_add directly, which meant a reply that finished
    # inside the window was never spoken yet still entered history).
    held_chunks = []

    if not holding:
        # Let the frontend clear the previous assistant line
        emit_to_room(client_id, "turn_start", {})

    message_to_add = ""
    interrupted = False
    first_emit_time = None
    total_audio_duration = 0.0

    def _speak_chunk(chunk):
        """Normal emit path for one LLM segment: text event + TTS audio.
        Returns True if a stop was requested mid-synthesis."""
        nonlocal message_to_add, first_emit_time, total_audio_duration
        logger.info(f"[{client_id}] LLM Chunk: {chunk}")
        emit_to_room(client_id, "text_response", {"text": chunk})
        message_to_add += chunk
        # Remember spoken segments for the self-talk (echo) check
        session.assistant_chunks.append(chunk)
        if len(session.assistant_chunks) > 12:
            del session.assistant_chunks[0]
        for wav_chunk in tts_client.synthesize(
            chunk, streaming=True, options=tts_options
        ):
            if current_stop_event.is_set():
                return True
            if first_emit_time is None:
                first_emit_time = time.time()
            # Calculate audio duration: bytes / (sample_rate * channels * bytes_per_sample)
            # Assuming 24k sample rate, 1 channel, 16-bit (2 bytes) = 48000 bytes/sec
            total_audio_duration += len(wav_chunk) / 48000.0
            emit_to_room(client_id, "audio_chunk", wav_chunk)
            # Estimated remaining buffered playback: chunks are emitted
            # as generated while the client plays them back in real
            # time, so what is left in the buffer is (duration emitted
            # so far) - (wall time since first chunk). The circle-status
            # gate uses this to keep blips from flipping the UI.
            session.playback_end = (
                time.time()
                + max(
                    total_audio_duration - (time.time() - first_emit_time),
                    0.0,
                )
                + 0.5
            )
        return current_stop_event.is_set()

    def _flush_held():
        """The grace window passed with no resumed speech: this was a real
        end of turn. Release every held segment through the normal emit
        path, then let the caller continue live. Returns True if a stop
        landed mid-flush."""
        nonlocal holding
        holding = False
        emit_to_room(client_id, "turn_start", {})
        for held in held_chunks:
            if current_stop_event.is_set():
                return True
            if _speak_chunk(held):
                return True
        return False

    for i, chunk in enumerate(
        llm_reply_gen
        if hasattr(llm_reply_gen, "__iter__") and not isinstance(llm_reply_gen, str)
        else [llm_reply_gen]
    ):
        if current_stop_event.is_set():
            interrupted = True
            break

        if holding:
            if session.last_voice_ts > grace_ref_ts:
                # Sustained voice during the grace window: the turn
                # boundary was premature, the user is still talking.
                premature = True
                break
            if time.time() >= grace_until:
                # Window passed with no resumed speech: this was a real
                # end of turn — flush the held segments and continue live
                if _flush_held():
                    interrupted = True
                    break
            else:
                # Still inside the window: keep the LLM streaming but
                # hold everything back from the user
                held_chunks.append(chunk)
                continue

        if _speak_chunk(chunk):
            interrupted = True
            break

    if holding and not premature and not interrupted:
        # The generation finished entirely inside the grace window (short
        # reply / fast LLM): the window still has to pass before anything
        # is spoken — the user may still be mid-sentence. Once it does,
        # the held segments MUST be flushed (they have not been spoken).
        while time.time() < grace_until:
            if current_stop_event.is_set():
                interrupted = True
                break
            if session.last_voice_ts > grace_ref_ts:
                premature = True
                break
            time.sleep(0.05)
        if not premature and not interrupted:
            if _flush_held():
                interrupted = True

    if premature:
        # Nothing was spoken, so nothing enters history — an unspoken
        # assistant turn would pollute the conversation the user never
        # heard. The user message is already in llm history and will
        # merge with their continuation (llm_client groupby).
        logger.info(
            f"[{client_id}] premature turn dropped (user resumed during "
            f"grace window); reply generation abandoned"
        )
        try:
            llm_reply_gen.close()  # runs the generator's finally -> closes HTTP
        except Exception:
            pass
        with session.lock:
            session.pending_message = None
            session.pending_audio_duration = 0.0
            session.pending_start_time = None
            session.interruption_time = None
        return False

    if interrupted:
        # If interrupted mid-stream, calculate truncation immediately using current time
        if first_emit_time and total_audio_duration > 0:
            # interruption_time can be None when a NEW TURN (not a
            # barge-in) stopped this stream — the new pipeline's pending
            # block cleared it before calling interrupt(). Truncate at
            # "now" in that case (the turn arrived just moments ago).
            elapsed_time = max(
                (session.interruption_time or time.time()) - first_emit_time, 0
            )
            ratio = min(elapsed_time / total_audio_duration, 1.0)
            cutoff_length = int(len(message_to_add) * ratio)
            message_to_add = message_to_add[:cutoff_length]
            logger.info(
                f"[{client_id}] Interrupted mid-stream. Ratio: {ratio:.2f}. Truncated message: {message_to_add}"
            )
        else:
            message_to_add = ""

        # Commit immediately as this pipeline execution is dead
        # (an empty message means nothing was ever spoken — e.g. a stop
        # that landed inside the grace hold — and an empty assistant
        # turn in history is pure noise for the next generation)
        if message_to_add:
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

    return not interrupted


# ---------------------------------------------------------------------------
# Fast/slow brain: delivery-timing hook + the delivery speech path
# ---------------------------------------------------------------------------

def _agent_can_deliver(session):
    """Is now a natural gap to speak a background result? The user must not
    be talking, nothing may be playing, and no generation may be running."""
    now = time.time()
    with session.lock:
        if session.gen_in_progress:
            return False
    if now < session.playback_end:
        return False
    if now - session.last_voice_ts < 1.2:
        return False
    return True


def _agent_deliver_entry(session, client_id, entry):
    """Speak one completed background task result through the normal speech
    pipeline. Returns True if fully spoken, False if interrupted."""
    _commit_pending_message(session, client_id)
    session.interrupt()  # stop anything still lingering
    session.reset_interrupt()
    current_stop_event = session.stop_event
    if current_stop_event.is_set():
        return False

    # The result enters history as a user-role system note (persisted: later
    # turns can still refer to it within the history window)
    if entry.get("error") is not None:
        note = (
            f"[系统提示：后台任务 {entry['task_id']}（{entry['goal']}）执行失败，"
            f"失败原因：{entry['error']}。\n"
            "——请用自然口语向用户致歉并简要说明这个任务没做成，可以主动提出换个方式再试，"
            "不要编造任何结果，控制在两三句话内。]"
        )
    else:
        note = (
            f"[系统提示：后台任务 {entry['task_id']}（{entry['goal']}）已完成，以下是真实结果——\n"
            f"{entry['result']}\n"
            "——请用自然口语向用户汇报要点：先说结论，关键数字和日期必须准确引用上面的结果，"
            "不要逐字朗读，不要使用列表格式，控制在几句话内。]"
        )
    llm.add_message(client_id, "user", note)
    logger.info(f"[{client_id}] Delivering background task {entry['task_id']}")

    settings = dict(session.settings)
    tts_client = TTS_CLIENTS.get(
        settings.get("tts_engine"), TTS_CLIENTS[DEFAULT_SETTINGS["tts_engine"]]
    )
    tts_options = {
        "cfg_value": settings.get("tts_cfg", 2.0),
        "inference_timesteps": settings.get("tts_timesteps", 10),
    }
    gen = llm.generate_with_history(
        client_id,
        stop_event=current_stop_event,
        system_prompt=settings.get("system_prompt") or None,
        max_history=settings.get("max_history"),
        min_seg_len=settings.get("min_seg_len"),
    )
    with session.lock:
        session.gen_in_progress = True
    try:
        return _run_speech_pipeline(
            session, client_id, gen, current_stop_event, tts_client, tts_options
        )
    finally:
        with session.lock:
            session.gen_in_progress = False


def pipeline_worker(client_id, audio_segment, sample_rate, turn_trigger=None):
    """
    Main processing pipeline: ASR -> LLM -> TTS.
    Runs in a background thread for each detected utterance.

    turn_trigger: how the turn boundary was reached on the turn server
    ("complete" = <|user_complete|>, instant, unverified; "wait" =
    silence-verified). Only "complete" turns get the premature-turn grace
    hold in _run_speech_pipeline — that is the 抢话 fix of 2026-10-30.
    """
    session = session_manager.get_session(client_id)
    if not session:
        return

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

        # Main-speaker gate, AFTER ASR: the self-talk check needs the
        # turn's text, and blip turns bail at the empty-text check above
        # for free. A rejected turn must not interrupt the ongoing
        # playback, touch LLM history, or cost any LLM/TTS work.
        if not pvad_turn_gate(session, asr_text):
            return
        if pvad_enabled(session):
            pvad_maybe_auto_enroll(session)

        # # 2. Backchannel Detection (vad already handles this, but double-check here)
        # # Checks for filler words or short backchannels that shouldn't trigger a full response
        # if check_backchannel(asr_text):
        #     emit_to_room(client_id, "resume_audio", {"message": "backchannel detected"})
        #     return

        # Process Pending Message from previous turn if any (before adding new user message)
        _commit_pending_message(session, client_id)

        # 3. Preparation for Response
        session.interrupt()  # Signal previous threads to stop
        session.reset_interrupt()  # Create a fresh event for this new thread

        # Capture the specific stop_event for this execution cycle
        current_stop_event = session.stop_event

        if current_stop_event.is_set():
            return

        logger.info(f"[{client_id}] ASR result: {asr_text}")
        emit_to_room(client_id, "user_transcription", {"text": asr_text})
        # Bank this turn's speech for cross-turn speaker enrollment (only
        # reaches here with non-empty ASR text and a gate pass, so noise
        # and rejected turns never enter the bank)
        if pvad_enabled(session):
            pvad_bank_turn_audio(session)

        # Snapshot live settings for this pipeline run (user may tweak the
        # UI mid-generation; one consistent view per turn is enough)
        settings = dict(session.settings)
        tts_client = TTS_CLIENTS.get(
            settings.get("tts_engine"), TTS_CLIENTS[DEFAULT_SETTINGS["tts_engine"]]
        )
        tts_options = {
            "cfg_value": settings.get("tts_cfg", 2.0),
            "inference_timesteps": settings.get("tts_timesteps", 10),
        }

        llm.add_message(client_id, "user", asr_text)

        # 4. Generation. With the agent enabled, the fast brain gets its
        # delegate/cancel tools (native tool calling via /agent_chat) plus
        # the Agent Runtime's current task view injected as context. Tool
        # rounds stay transient — only the final spoken text is persisted.
        agent_runtime = (
            session.agent_runtime
            if settings.get("agent_enabled") and session.agent_runtime
            else None
        )
        if agent_runtime:
            system_prompt = (
                settings.get("system_prompt") or QwenLLM_stream.SYSTEM_PROMPT
            ) + "\n\n" + AGENT_SYSTEM_ADDON
            # deterministic retraction handling: if the user's words clearly
            # cancel a running task, the Runtime cancels it NOW (the fast
            # brain understands "算了" but reliably fails to call the tool
            # itself — see runtime.auto_cancel) and the note grounds its
            # confirmation in fact.
            #
            # Task view is MENTION-GATED (2026-10-30): injected only when
            # the user's words actually reference the background work (or
            # a cancel note is being injected). Unconditional injection had
            # the fast brain narrating task status in every reply while
            # the slow brain ran — the user wants free chat on other
            # topics in the meantime, with progress synced only on request.
            auto_note = agent_runtime.auto_cancel(asr_text)
            # NOTE: no realtime MUST-delegate note is injected anymore —
            # an A/B against :6008 (2026-10-30, 5 trials/cell) showed any
            # trailing user-role note REDUCES the fast brain's own tool
            # trigger rate (4-5/5 without vs 0-4/5 with, three wordings).
            # The addon's realtime hard rule + runtime.auto_delegate
            # (post-turn safety net below) cover the failure mode instead.
            extra_context = None
            if auto_note or _TASK_MENTION_RE.search(asr_text):
                extra_context = agent_runtime.task_view()
            if auto_note:
                extra_context = (
                    (extra_context + "\n" + auto_note) if extra_context
                    else auto_note
                )
            llm_reply_gen = llm.generate_agent_with_history(
                client_id,
                stop_event=current_stop_event,
                system_prompt=system_prompt,
                max_history=settings.get("max_history"),
                min_seg_len=settings.get("min_seg_len"),
                extra_context=extra_context,
                tools=FAST_TOOLS,
                tool_handler=agent_runtime.handle_tool,
            )
            # pre-generation task-table snapshot for the auto_delegate
            # safety net below
            task_ids_before = set(agent_runtime.tasks.keys())
        else:
            llm_reply_gen = llm.generate_with_history(
                client_id,
                stop_event=current_stop_event,
                system_prompt=settings.get("system_prompt") or None,
                max_history=settings.get("max_history"),
                min_seg_len=settings.get("min_seg_len"),
            )

        # 5. LLM & TTS Streaming (shared speech outlet)
        # Grace window length is live-tunable (settings turn_grace_ms); the
        # module constant is only the fallback. 0 disables the window.
        grace_s = (
            settings.get("turn_grace_ms", int(_TURN_GRACE_S * 1000)) / 1000.0
        )
        grace_until = (
            time.time() + grace_s if turn_trigger == "complete" and grace_s > 0
            else None
        )
        with session.lock:
            session.gen_in_progress = True
        try:
            fully_spoken = _run_speech_pipeline(
                session, client_id, llm_reply_gen, current_stop_event,
                tts_client, tts_options, grace_until=grace_until,
            )
            # deterministic delegate safety net (runtime.auto_delegate):
            # the fast brain said "我去查" in speech but never called
            # delegate_task — measured 2026-10-30 14:51 (session
            # ec822c0a) with the realtime MUST-delegate note injected and
            # the call still missing; the model later fabricated the
            # answer. Create the task now so the acceptance speech is
            # true and the panel shows it.
            if agent_runtime and fully_spoken:
                reply_text = ""
                with session.lock:
                    reply_text = session.pending_message or ""
                new_tid = agent_runtime.auto_delegate(
                    asr_text, reply_text, task_ids_before)
                if new_tid:
                    logger.info(
                        "[%s] auto-delegated %s: fast brain accepted in "
                        "speech but never called delegate_task (%s)",
                        client_id[:8], new_tid, asr_text[:60],
                    )
        finally:
            with session.lock:
                session.gen_in_progress = False

    except Exception as e:
        logger.error(f"Error in pipeline for {client_id}: {e}", exc_info=True)


# ==== WebSocket Endpoint & Processing ====

# Settings validation lives in settings.py (sanitize_updates); this only
# applies a sanitized update to the session and runs the side effects.


def _merge_session_settings(session, data: dict) -> dict:
    """Validate + merge a partial settings update into session.settings."""
    clean = sanitize_updates(data)

    if clean.get("turn_engine") not in (None, session.settings.get("turn_engine")):
        # engine switch mid-session: drop any half-finished FRC
        # utterance so a stale pending commit can't fire later
        if session.frc is not None:
            session.frc.reset()
        # switching back to soulx: re-apply the wait override — it was
        # skipped while in frc mode, and the server session may have
        # been GC'd during the detour (recreated from session_config,
        # but only if a push ever happened)
        if clean["turn_engine"] == "soulx":
            _push_turn_wait_num(session)

    session.settings.update(clean)

    if "turn_wait_ms" in clean:
        _push_turn_wait_num(session)

    return dict(session.settings)


def _push_turn_wait_num(session):
    """Push the session's turn_wait_ms to the turn server as max_wait_num.

    Call after the setting changes and once at session init. Skipped in frc
    mode (the SoulX turn server sees no audio then — harmless either way,
    but the push would race the engine switch for nothing). The turn server
    keeps the override in a GC-surviving store, so it survives frc detours
    and reconnects.
    """
    if session.settings.get("turn_engine") == "frc":
        return
    wait_ms = session.settings.get("turn_wait_ms")
    if wait_ms is None:
        return
    n = int(wait_ms) // _CHUNK_MS
    if n < 1:
        n = 1
    try:
        session.vad.set_max_wait_num(n)
    except Exception as exc:
        logger.warning(f"set_max_wait_num({n}) failed for session: {exc}")


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

        # announce the wait-path override to the turn server now — the
        # server session is created on first audio, and the override store
        # must be populated before that (also survives server restarts via
        # the client's per-reconnect re-push)
        _push_turn_wait_num(session)

        # Rebind the circle callback now that the session exists: while the
        # assistant's buffered audio is still playing, a nonidle blip's
        # LISTENING must not flip the UI back and forth with the per-chunk
        # SPEAKING the frontend sets on every audio_chunk.
        def _circle_gated(status):
            if status == "LISTENING" and time.time() < session.playback_end:
                return
            emit_to_room(client_id, "circle_status", {"status": status})

        vad_instance.circle_callback = _circle_gated

        # Fast/slow brain runtime: background task management + result
        # delivery arbitration for this session
        session.agent_runtime = AgentRuntime(
            client_id,
            emit=lambda event, data: emit_to_room(client_id, event, data),
            can_deliver=lambda: _agent_can_deliver(session),
            deliver=lambda entry: _agent_deliver_entry(session, client_id, entry),
        )
        if session.settings.get("agent_enabled"):
            session.agent_runtime.start()

        # FireRedChat-architecture turn engine (only fed audio while
        # settings turn_engine == "frc"; started here so the UI switch is
        # live without a reconnect)
        def _frc_last_assistant_text():
            # what the user just heard: the pending (still-playing) reply
            # if any, else the last committed assistant message
            with session.lock:
                pending = session.pending_message
            if pending and pending.strip():
                return pending
            for m in reversed(llm.get_session(client_id)):
                if m["role"] == "assistant" and (m.get("content") or "").strip():
                    return m["content"]
            return ""

        def _frc_commit(transcript):
            # same speech pipeline as the SoulX path; no grace window —
            # the dual-EOT decision already guards against premature turns
            threading.Thread(
                target=pipeline_worker,
                args=(client_id, transcript, Config.SAMPLE_RATE, None),
                daemon=True,
            ).start()

        session.frc = FrcEngine(
            session, client_id,
            commit_cb=_frc_commit,
            context_cb=_frc_last_assistant_text,
        )
        session.frc.start()

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

                # FireRedChat frontend mode: the chunk goes to the FrcEngine
                # (pvad speech-activity endpointing → dual EOT → FireRedASR)
                # and the SoulX turn server is bypassed for this session.
                # Switching is per-session and live (Settings panel).
                if session.settings.get("turn_engine") == "frc":
                    session.frc.feed(audio_chunk)
                    continue

                # PVAD sidecar: async copy of the mic chunk + enrollment ring.
                # Both are outside session.lock so the turn-controller path
                # below is untouched (feed() is a queue put; ring append is
                # an O(1) list append under its own short-lived lock).
                # Skipped entirely when the gate is off — zero pvad work
                # on the direct-SoulX path.
                if pvad_enabled(session):
                    session.pvad.feed(audio_chunk)
                    pvad_ring_append(session, audio_chunk)
                    pvad_emit_status(session)

                t_proc = time.time()
                with session.lock:
                    segment = session.vad.process(audio_chunk)
                dt_proc = time.time() - t_proc
                if dt_proc > 0.3:
                    logger.warning(
                        f"[{client_id}] vad.process took {dt_proc:.2f}s "
                        f"(expected ~0.03s)"
                    )

                if segment is None:
                    if session.bargein_streak:
                        logger.info(
                            f"[{client_id}] transient barge-in suppressed "
                            f"({session.bargein_streak} chunk(s) < "
                            f"{_BARGEIN_MIN_CHUNKS})"
                        )
                    session.bargein_streak = 0
                    session.bargein_audio = []
                    session.voice_run = 0
                elif isinstance(segment, list) and segment[0] is None:
                    # Barge-in signal — hysteresis first: the turn server
                    # emits nonidle per chunk, and a 1-2 chunk blip (noise /
                    # echo residue) must not stop playback. Only sustained
                    # nonidle (real speech) reaches the main-speaker gate,
                    # which then runs in a worker thread
                    # (see pvad_handle_bargein).
                    # NOTE: a blip must also NOT push the delivery arbiter's
                    # quiet-gap boundary (last_voice_ts) — measured 2026-10-30,
                    # episodic 1-chunk echo blips every few seconds kept
                    # deferring background-task delivery past every gap.
                    # Sustained voice (below) and complete turns (line ~1524)
                    # still push it.
                    session.bargein_streak += 1
                    session.voice_run += 1
                    session.bargein_audio.append(audio_chunk)
                    if len(session.bargein_audio) > 16:  # ~2.5 s rolling
                        del session.bargein_audio[0]
                    if session.bargein_streak >= _BARGEIN_MIN_CHUNKS:
                        # sustained voice: trusted evidence a human is
                        # talking — also opens the blip-filter's acceptance
                        # window (the turn text may arrive a few seconds
                        # after the audio went idle)
                        session.last_sustained_voice_ts = time.time()
                        session.last_voice_ts = time.time()
                        session.bargein_streak = 0
                        pvad_handle_bargein(session)
                else:
                    # Voice activity (complete turn): push the delivery
                    # arbiter's quiet-gap boundary forward
                    session.last_voice_ts = time.time()
                    run, session.voice_run = session.voice_run, 0
                    session.bargein_streak = 0
                    session.bargein_audio = []
                    # Real-speech evidence for this turn: either sustained
                    # voice is STILL ongoing (run >= 3), or it happened
                    # moments ago — the turn server often emits the text
                    # only after the audio went idle (voice_run reads 0-2
                    # for real first turns and split-utterance trailing
                    # segments; measured 2026-10-30). Echo/noise blips
                    # never sustain 3 consecutive chunks, so a turn with
                    # neither is a hallucinated fragment.
                    sustained_recent = (
                        time.time() - session.last_sustained_voice_ts
                    ) < _SUSTAINED_VOICE_WINDOW_S
                    if (_BLIPTURN_FILTER
                            and run < _BARGEIN_MIN_CHUNKS
                            and not sustained_recent):
                        # Blip turn: no sustained voice anywhere near this
                        # "complete utterance" — it is echo/noise residue,
                        # and its text is a paraformer hallucination (short
                        # frequent fragments like "怎么样""是不是""平时").
                        # Never spoken, never banked, never enters history;
                        # the UI drops the streamed partial via the same
                        # path as a gate rejection.
                        logger.info(
                            f"[{client_id}] blip turn dropped "
                            f"({run} chunk(s) < {_BARGEIN_MIN_CHUNKS}): "
                            f"{str(segment)[:40]!r}"
                        )
                        emit_to_room(
                            client_id, "pvad_reject",
                            {"kind": "blip_turn", "chunks": run,
                             "text": str(segment)[:40]},
                        )
                        continue
                    # Complete Utterance — pipeline_worker applies the
                    # main-speaker gate as its first step. Capture the
                    # trigger NOW, in the receive loop: the worker thread
                    # may start later than the next vad.process() call,
                    # which could overwrite last_turn_trigger.
                    turn_trigger = getattr(
                        session.vad, "last_turn_trigger", None
                    )
                    threading.Thread(
                        target=pipeline_worker,
                        args=(client_id, segment, Config.SAMPLE_RATE,
                              turn_trigger),
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
                        if not pvad_enabled(session):
                            logger.info(
                                f"[{client_id}] pvad re-enroll ignored (gate off)"
                            )
                        else:
                            logger.info(f"[{client_id}] pvad re-enroll requested")
                            pvad_try_enroll(session, "manual")

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
                        # Fast/slow brain toggle: starting/stopping the
                        # runtime cancels running tasks on switch-off
                        if session.agent_runtime:
                            if merged.get("agent_enabled"):
                                session.agent_runtime.start()
                            else:
                                session.agent_runtime.shutdown()
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
            if session.agent_runtime:
                session.agent_runtime.shutdown()
            if session.frc:
                session.frc.stop()
            session.pvad.stop()
            vad_pool.release(session.vad)
            session_manager.remove_session(client_id)
        logger.info(f"Session cleaned up: {client_id}")


# ==== Static Resource Routing ====
# Browsers heuristic-cache index.html (StaticFiles sends only etag/last-modified),
# so an updated frontend can stay invisible in an already-open tab. Always
# revalidate the HTML shell; JS/CSS bust via the ?v= query in index.html.
@app.middleware("http")
async def no_cache_html(request, call_next):
    response = await call_next(request)
    if response.headers.get("content-type", "").startswith("text/html"):
        response.headers["Cache-Control"] = "no-cache"
    return response


app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    logger.info(f"Server starting on http://localhost:{Config.PORT}")
    uvicorn.run(app, host="0.0.0.0", port=Config.PORT)
