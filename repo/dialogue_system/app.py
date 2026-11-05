import hmac
import os
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
from collections import deque
from pathlib import Path
from typing import Dict, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse, StreamingResponse
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
    _PVAD_PLAYBACK_GRACE_S,
    _pvad_selftalk_text,
    pvad_bank_turn_audio,
    pvad_emit_status,
    pvad_enabled,
    pvad_handle_bargein,
    pvad_maybe_auto_enroll,
    pvad_ring_append,
    pvad_streak_append,
    pvad_try_enroll,
    pvad_turn_gate,
)
# Pre-ASR stream gate (2026-11-05): silences non-main-speaker audio
# before it reaches the turn server. Same sidecar discipline as the rest
# of the pvad path — the receive loop only reads a flag; the scorer
# thread does the blocking ECAPA work.
from pvad_stream import PvadStreamGate, stream_gate_active, stream_gate_filter
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

        # Timeline forensics recorders (best-effort; see _SessionAudio).
        # mic records what the turn server saw (post-resample 16 kHz),
        # tts records the 24 kHz stream the browser plays.
        self.audio_mic = _SessionAudio(client_id, "mic", Config.SAMPLE_RATE)
        self.audio_tts = _SessionAudio(client_id, "tts", 24000)

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
        # The cos behind that verdict (None on fail-open). Once enrolled,
        # only high-margin passes (>= threshold + _PVAD_REENROLL_MARGIN)
        # may re-enroll or bank — marginal passes are the user+background
        # MIXED windows that poisoned the reference on 2026-11-05.
        self.pvad_turn_cos = None
        # Pre-ASR stream gate (pvad_stream.PvadStreamGate), created lazily
        # by the receive loop only while pvad is enabled. None = inert.
        self.pvad_stream_gate = None
        # Gate-decision audio dumps written for this session (forensics
        # cap; see pvad_gate._pvad_dump_forensics)
        self.pvad_forensics_n = 0
        # Barge-in hysteresis: consecutive nonidle chunks. A 1-2 chunk
        # nonidle blip (noise / AEC residue the turn server misclassified)
        # must not stop playback — only sustained nonidle (real speech)
        # may interrupt. Any idle/speak result resets the streak.
        self.bargein_streak = 0
        # The nonidle audio itself as (ts, wav) pairs, rolling ~2.5 s of
        # audio / _PVAD_BARGEIN_RETAIN_S of wall time. The barge-in
        # identity check scores THIS, not the mic ring: the ring's
        # recent-speech window can still hold the user's just-finished
        # turn, which made the first echo/noise onset score as the main
        # speaker. GAP-TOLERANT since 2026-11-05: an idle chunk no longer
        # clears it (only a turn commit or age does) — with a background
        # video playing, the turn server's rms gate rewrites the quiet
        # video chunks to idle, and the old instant clear kept resetting
        # the user's interrupt evidence to zero mid-shout ("停停停" got
        # 10 consecutive unscoreable rejections in 1.7 s; report docs/
        # PVAD背景视频失效调查报告-20261105.md §4). Retention is bounded
        # so stale evidence cannot confirm a later barge-in either.
        self.bargein_audio = []
        # Timestamps of recent nonidle chunks (rolling _PLAYBACK_BARGEIN_
        # WINDOW_S) — duty-cycle evidence for the playback barge-in guard
        # (2026-11-02 幻影打断修复; see the constants block). Unlike
        # bargein_streak this survives across idle gaps, because browser-
        # processed real speech alternates nonidle/idle (~50% duty) and
        # never sustains a consecutive run.
        self.bargein_times = deque()
        # Nonidle chunks in the CURRENT voice burst (reset on each turn
        # commit, and lazily in the receive loop when nonidle resumes
        # after a > window-sized gap). Same duty-cycle idea as
        # bargein_times but covering the WHOLE burst — a blip that
        # completes a phantom turn (turn server commits, paraformer
        # hallucinates 1-3 characters onto the echo) reads 1-2 here,
        # while any real utterance long enough to be a turn reads 4-8
        # even at browser-WebRTC's ~50% duty cycle.
        self.episode_nonidle = 0
        self.last_nonidle_ts = 0.0
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
        # Playback-run bookkeeping for the self-talk echo window
        # (2026-11-05 吞句修复): serial / first-audio-emit wall-clock /
        # total emitted seconds of the CURRENT continuous playback run.
        # assistant_chunks entries carry (run, cum) so the echo check can
        # estimate the playback head pos = min(now - first, total) and
        # only match chunks whose audio is plausibly at the mic now.
        self.playback_run_serial = 0
        self.playback_run_first_ts = None
        self.playback_run_total = 0.0
        # Fast/slow brain: True while a speech generation (user turn OR
        # background-result delivery) is running in this session. The
        # delivery arbiter polls it to pick a speaking slot.
        self.gen_in_progress = False
        # generation serial: each _begin_generation() bump invalidates the
        # previous pipeline's finally-clear, so an interrupted old pipeline
        # can never clear gen_in_progress out from under the new one that
        # replaced it (the old thread's LLM loop only notices the stop at
        # its next chunk — its finally can land seconds into the new
        # generation)
        self.gen_serial = 0
        # Wall-clock of the last user voice activity (barge-in chunks or a
        # completed turn) — the delivery arbiter waits for a quiet gap.
        self.last_voice_ts = 0.0
        # UI continuation merge (2026-11-05): wall-clock of the last FINAL
        # user transcription, and wall-clock of the last assistant audio
        # emission. A new user turn whose last_user_turn_ts is recent and
        # predates any assistant audio is a continued utterance the turn
        # server split (LLM side already merges via groupby) — it gets
        # append: true so the frontend folds it into the previous bubble.
        self.last_user_turn_ts = 0.0
        self.assistant_audible_ts = 0.0
        # AgentRuntime (task mgmt + delivery arbiter). Created in the
        # websocket endpoint once the session is registered.
        self.agent_runtime = None
        # FrcEngine (FireRedChat-architecture turn engine, only fed audio
        # while settings turn_engine == "frc"). Created in the websocket
        # endpoint alongside the agent runtime.
        self.frc = None
        # Recent assistant text segments (the LLM chunks that get spoken),
        # newest last, capped. Each entry is {"text", "run", "cum", "ts"}:
        # run = playback_run_serial the chunk's audio belongs to, cum =
        # cumulative seconds of that run's audio when the chunk finished
        # emitting, ts = wall-clock at append. The self-talk check uses
        # these to match a turn's ASR text only against chunks that are
        # plausibly audible NOW (echo physics), never stale ones.
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

_BARGEIN_MIN_CHUNKS = 0   # consecutive nonidle chunks before a barge-in
                          # acts. 0 = no hysteresis: the FIRST nonidle chunk
                          # goes straight to the gate (2026-10-30 迟滞下线；
                          # previously 3 x 160 ms = 480 ms — the duplicate
                          # storm is still absorbed by bargein_done /
                          # pvad_checking). Blip-turn counting (below) reads
                          # it too, so 0 also disables that count-based
                          # branch (filter already off via _BLIPTURN_FILTER).
# Playback-period barge-in guard (2026-11-02 幻影打断修复): with the
# hysteresis off AND the pvad gate off (the default), a SINGLE nonidle
# chunk of speaker echo / AEC residue stops the TTS instantly — measured
# 2026-11-02 22:44 (external client, quiet room): several interrupts per
# session, each ~1 s into playback, plus a few hallucinated characters
# streamed to the UI. The consecutive-count hysteresis is NOT restorable
# (browser-processed real speech alternates nonidle/idle at ~50% duty and
# never sustains 3 consecutive chunks — the 2026-10-30 finding that
# retired the blip-turn filter), so the guard counts nonidle chunks in a
# SLIDING WINDOW instead: duty-cycle evidence, robust to that alternation.
# Echo residue is episodic (1 chunk every few seconds) and never reaches
# 3-in-window; real speech at 50% duty does within ~1 s. Only applies
# while assistant audio is actually playing (playback_end in the future)
# and the pvad gate is not in enforce mode (enforce has its own stronger
# fail-closed playback check); outside playback nothing needs protecting
# and the zero-latency first-chunk path is unchanged.
_PLAYBACK_BARGEIN_WINDOW_S = 1.6   # evidence window (~10 chunks)
_PLAYBACK_BARGEIN_NONIDLE = 3      # nonidle chunks needed inside it
# (bargein_audio's gap-tolerant retention lives in pvad_gate.py:
#  _PVAD_BARGEIN_RETAIN_S — see the 2026-11-05 background-video report)


def _playback_guard_active(session):
    """Master switch for the playback barge-in guard: the per-session
    `playback_guard` setting (default on, live-tunable from the Settings
    panel) AND not in pvad enforce mode (enforce has its own stronger
    fail-closed playback checks on both the barge-in and turn paths)."""
    if not session.settings.get("playback_guard", True):
        return False
    mode = session.settings.get("pvad_mode", DEFAULT_SETTINGS["pvad_mode"])
    return not (pvad_enabled(session) and mode == "enforce")


def _begin_generation(session) -> int:
    """Mark a reply as generating, IN THE RECEIVE LOOP at turn-commit time
    (2026-11-03): pipeline_worker only sets gen_in_progress after its
    gates and LLM-generator creation (~0.5 s with agent mode), and the
    01:26 test round showed a blip landing exactly in that gap — its
    barge-in killed the in-flight generation and the follow-up phantom
    turn '是。' then passed with in-flight already False. Setting the
    flag at commit closes the hole; the returned serial makes the
    worker's finally-clear a no-op once a newer generation began."""
    with session.lock:
        session.gen_serial += 1
        session.gen_in_progress = True
        return session.gen_serial


def _reply_in_flight(session):
    """Is a reply pending or playing? BOTH phases, not just playback:
    generation (turn committed, LLM/TTS streaming, gen_in_progress —
    nothing is audible yet, so playback_end is stale) and playback
    (playback_end + grace). frc_engine already uses this same pair for
    its EOT wait. The 2026-11-03 01:01 incident (session 338b1765): a
    1-2 chunk mic blip in the GENERATION window (a) confirmed a
    zero-evidence barge-in that killed the in-flight generation and
    (b) completed a phantom turn '是。' that replaced the user's real
    sentence in the UI and entered LLM history — both defenses below
    keyed on playback_end alone and were inactive until the first TTS
    audio chunk. The same duty-cycle discriminator applies: a real
    follow-up utterance reads 4-8 nonidle chunks, a breath/noise blip
    1-2."""
    return (session.gen_in_progress
            or time.time() < session.playback_end + _PVAD_PLAYBACK_GRACE_S)


def _playback_bargein_deferred(session, now=None):
    """True while a reply is in flight (generating or playing) but the
    nonidle evidence so far is too sparse to interrupt on — the deferred
    barge-in keeps counting (bargein_streak is NOT reset) and confirms
    on its own once the window fills, so a genuine talker only loses
    ~1 s, once."""
    now = now if now is not None else time.time()
    if not _reply_in_flight(session):
        return False
    if not _playback_guard_active(session):
        return False
    return len(session.bargein_times) < _PLAYBACK_BARGEIN_NONIDLE

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

# UI continuation-merge window (2026-11-05): a turn-server split of ONE
# user utterance lands its follow-up commit within a couple seconds of the
# previous one (the counting incident's three bubbles spanned ~2s). The
# LLM side already merges such turns (llm_client groupby joins consecutive
# user messages with "\n"); this window lets the UI mirror that semantics.
# Deliberately generous — a false positive only concatenates two adjacent
# user bubbles that the LLM saw as one turn anyway; a false negative
# re-splits an utterance the LLM merged.
_USER_TURN_MERGE_WINDOW_S = 4.0


def _is_continuation_turn(session):
    """Should this final user transcription be appended to the previous
    user bubble instead of opening a new one? Mirrors the LLM-side groupby
    merge: the previous user turn is recent (within
    _USER_TURN_MERGE_WINDOW_S) and NO assistant audio has been audible
    since it (assistant_audible_ts < last_user_turn_ts). With no assistant
    speech in between, two rapid user turns are one continued utterance
    the turn server sliced — or, at worst, a quick restatement, which the
    LLM already treats as one message."""
    now = time.time()
    return bool(
        session.last_user_turn_ts
        and now - session.last_user_turn_ts < _USER_TURN_MERGE_WINDOW_S
        and session.assistant_audible_ts < session.last_user_turn_ts
    )


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
    pending-message bookkeeping (interruption truncation). Returns a
    tri-state:

    - True  — the generation ran to completion;
    - False — interrupted AFTER audio was emitted (the user heard at least
      part of it; for the delivery arbiter this is a rejection — terminal,
      no retry: they will ask if they still want it);
    - None  — stopped before ANY audio was emitted (stop inside the LLM
      generator, or during the first synthesis, or a premature-turn drop):
      the user never heard a word, so the delivery arbiter must NOT count
      it as a rejection — it retries on a separate silent budget instead
      (and _agent_deliver_entry retracts the result note so the report
      has exactly one outlet at a time).

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
        # Remember spoken segments for the self-talk (echo) check, with
        # run/cum/ts bookkeeping so the check can bound its sources to
        # physically plausible playback (2026-11-05 吞句修复). cum is
        # updated as this chunk's audio streams; the serial is patched
        # onto the entry when the run's first audio actually emits.
        entry = {
            "text": chunk, "run": session.playback_run_serial,
            "cum": 0.0, "ts": time.time(),
        }
        session.assistant_chunks.append(entry)
        if len(session.assistant_chunks) > 12:
            del session.assistant_chunks[0]
        for wav_chunk in tts_client.synthesize(
            chunk, streaming=True, options=tts_options
        ):
            if current_stop_event.is_set():
                return True
            if first_emit_time is None:
                first_emit_time = time.time()
                # a new continuous playback run starts here (previous run
                # was interrupted or finished): re-seed its bookkeeping
                session.playback_run_serial += 1
                entry["run"] = session.playback_run_serial
                session.playback_run_first_ts = first_emit_time
                session.playback_run_total = 0.0
                # assistant audio is now audible (or about to be, within
                # client buffer latency): any user turn committed AFTER
                # this point is a new conversational turn, not a
                # continuation of the previous user utterance
                session.assistant_audible_ts = time.time()
                # forensic recording: a new pipeline means playback was
                # discontinuous (user turn or barge-in in between) — the
                # tts sidecar must not glue this reply onto the last run
                session.audio_tts.run_break()
            # Calculate audio duration: bytes / (sample_rate * channels * bytes_per_sample)
            # Assuming 24k sample rate, 1 channel, 16-bit (2 bytes) = 48000 bytes/sec
            total_audio_duration += len(wav_chunk) / 48000.0
            session.playback_run_total = total_audio_duration
            entry["cum"] = total_audio_duration
            emit_to_room(client_id, "audio_chunk", wav_chunk)
            session.audio_tts.write(bytes(wav_chunk))
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

    # Did the generator ever hand us a segment? Every interrupted=True
    # assignment below lives INSIDE the loop body — a stop that kills the
    # generator before its first yield (barge-in during the delivery
    # generation window, before any audio exists) exits the loop with
    # ZERO iterations, and the pipeline would otherwise report "fully
    # spoken" for a reply that never said a word (2026-11-05 静默事故:
    # delivery counted as success, task result never re-delivered).
    received_segment = False
    for i, chunk in enumerate(
        llm_reply_gen
        if hasattr(llm_reply_gen, "__iter__") and not isinstance(llm_reply_gen, str)
        else [llm_reply_gen]
    ):
        received_segment = True
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

    if not received_segment and current_stop_event.is_set():
        # The stop landed inside the generator before it yielded anything:
        # zero segments were spoken, so this is an interruption, not a
        # successful run. The delivery arbiter's retry path (one re-enqueue
        # via MAX_DELIVERY_ATTEMPTS) depends on seeing False here. Done as
        # interrupted=True (not a bare return) so the existing interruption
        # cleanup below runs: nothing entered history, pending state is
        # cleared, and first_emit_time is None so no truncation happens.
        interrupted = True

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
        return None  # nothing was ever audible (see tri-state docstring)

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
            # forensic recording: emission ran ahead of playback (TTS
            # RTF < 1), so the sidecar's last run holds bytes the browser
            # never played — mark where audible playback actually stopped
            session.audio_tts.mark_cut(
                session.interruption_time or time.time(), first_emit_time)
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

    if interrupted and first_emit_time is None:
        # silent kill: the stop landed before any audio was emitted, so
        # the user never heard a word — not a rejection (tri-state
        # docstring). Applies to the zero-segment stop (received_segment
        # check above) and to a stop during the first synthesis.
        return None

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
    pipeline. Tri-state return, passed straight through to the arbiter:
    True = fully spoken; False = interrupted after audio was audible (a
    rejection — terminal, no retry); None = stopped before any audio
    existed (the user never heard the report — the arbiter retries on its
    separate silent budget, and the result note is retracted so the
    user-turn pipeline cannot speak the report meanwhile)."""
    _commit_pending_message(session, client_id)
    session.interrupt()  # stop anything still lingering
    session.reset_interrupt()
    current_stop_event = session.stop_event
    if current_stop_event.is_set():
        return None  # stopped before starting: nothing audible yet

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
    gen_sn = _begin_generation(session)
    try:
        ret = _run_speech_pipeline(
            session, client_id, gen, current_stop_event, tts_client, tts_options
        )
        if ret is None:
            # Silent kill: retract the result note. While it sits in
            # history the user's next ordinary reply reads it and speaks
            # the result, and the re-enqueued retry would then say it
            # AGAIN (measured 2026-11-05 17:36: full report spoken by the
            # user-turn pipeline, then re-delivered in full). The retry
            # re-injects a fresh note.
            if llm.pop_message(client_id, "user", note):
                logger.info(
                    f"[{client_id}] delivery killed before audio: "
                    f"result note retracted (retry re-injects)"
                )
        return ret
    finally:
        with session.lock:
            # only clear if no newer generation began (a turn committed
            # mid-delivery and its pipeline replaced this one)
            if session.gen_serial == gen_sn:
                session.gen_in_progress = False


def pipeline_worker(client_id, audio_segment, sample_rate, turn_trigger=None,
                    gen_sn=None, commit_id=None):
    """
    Main processing pipeline: ASR -> LLM -> TTS.
    Runs in a background thread for each detected utterance.

    turn_trigger: how the turn boundary was reached on the turn server
    ("complete" = <|user_complete|>, instant, unverified; "wait" =
    silence-verified). Only "complete" turns get the premature-turn grace
    hold in _run_speech_pipeline — that is the 抢话 fix of 2026-10-30.

    commit_id: the turn server's counter for this commit — pairs the
    async turn_final (two-pass ASR correction) with this turn.
    """
    session = session_manager.get_session(client_id)
    if not session:
        return

    try:
        # 0. Two-pass ASR correction (2026-11-05 方案A): the turn server
        # re-recognizes the WHOLE turn offline right after committing and
        # pushes a turn_final. Hold briefly here (this background thread,
        # never the event loop) and prefer the corrected text for
        # everything downstream — bubble, LLM history, self-talk check,
        # agent cancel. Live text comes from a 3.2s rolling window with
        # speech chunks glued without pauses and LCS-diffed against
        # history; measured on 37 real turns, whole-turn recognition with
        # the same paraformer is markedly more accurate (typical
        # correction latency 0.1-0.3s). Timeout or empty correction falls
        # back to the live text — the correction is an enhancement, never
        # a dependency.
        if (commit_id is not None
                and isinstance(audio_segment, str)
                and session.settings.get("asr_correct", True)
                and callable(getattr(session.vad, "wait_turn_final", None))):
            wait_s = session.settings.get("asr_correct_wait_ms", 500) / 1000.0
            t_hold = time.time()
            corrected = session.vad.wait_turn_final(commit_id, timeout=wait_s)
            if corrected and corrected != audio_segment:
                logger.info(
                    f"[{client_id}] ASR corrected in {time.time()-t_hold:.2f}s "
                    f"({turn_trigger or 'wait'} turn): {corrected} "
                    f"(live: {audio_segment})"
                )
                audio_segment = corrected

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

        # Playback self-talk drop (2026-11-02 幻影话轮修复): with the pvad
        # gate off (the default), the barge-in guard stops the instant
        # chop, but an echo blip can still COMPLETE a turn on the turn
        # server — its transcription is a few characters of what we are
        # saying right now — and interrupt playback down at the
        # session.interrupt() below. The text check is the one
        # discriminator that needs no enrollment: a real user's words are
        # not a recall-≥0.6 garbled subsequence of our own recent output.
        # Scope: only while a reply is in flight — generating OR playing,
        # see _reply_in_flight (the 2026-11-03 01:01 incident's phantom
        # turn landed in the generation window) — or just after a
        # confirmed barge-in (interrupt zeroes playback_end and clears
        # gen_in_progress, but dense echo is exactly what confirms a
        # barge-in — its turn commits ~1 s later and must still be
        # caught) — outside these windows a user may legitimately quote
        # the assistant; and only with the guard on and the gate off —
        # enforce mode already rejects these turns fail-closed
        # (pvad_turn_gate below). The receive loop's episode-count drop
        # (path a) catches the sub-4-character hallucination fragments
        # this text check cannot see.
        # 2026-11-05 吞句修复: _pvad_selftalk_text now also bounds its
        # match sources to physically plausible playback (chunks at the
        # playback head ± echo tail — the mic cannot hear 40-second-old
        # text), so quoting what we said a minute ago is free; only
        # near-verbatim quotes of what is AT the mic now still drop.
        recent_interrupt = (
            session.interruption_time is not None
            and time.time() - session.interruption_time
            < _PVAD_PLAYBACK_GRACE_S
        )
        if (not pvad_enabled(session)
                and _playback_guard_active(session)
                and (_reply_in_flight(session) or recent_interrupt)
                and _pvad_selftalk_text(session, asr_text)):
            emit_to_room(
                client_id, "selftalk_drop",
                {"reason": "selftalk", "text": asr_text[:60]},
            )
            logger.info(
                f"[{client_id}] self-talk turn dropped during playback "
                f"(pvad off): {asr_text[:40]!r}"
            )
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

        logger.info(
            f"[{client_id}] ASR result ({turn_trigger or 'wait'} turn): "
            f"{asr_text}"
        )
        # final=True tells the frontend to FREEZE the live user bubble:
        # the turn committed, this text is final. Without it, a phantom
        # blip's partials/final arriving before the assistant bubble
        # opens OVERWROTE the real sentence live in place (2026-11-03
        # 01:01: "喂，你好，可以听到吗？" visibly became "是。" — the
        # swallow the user reported, distinct from the turn itself).
        # append=True marks a CONTINUATION of the previous user turn —
        # same semantics the LLM side applies via groupby (llm_client
        # add_message merges consecutive user messages): recent previous
        # user turn, no assistant audio audible since it, within the
        # merge window. The frontend folds it into the previous frozen
        # bubble instead of opening a third one (2026-11-05 counting
        # incident: one instruction surfaced as three user bubbles).
        append = _is_continuation_turn(session)
        emit_to_room(
            client_id, "user_transcription",
            {"text": asr_text, "final": True, "append": append},
        )
        session.last_user_turn_ts = time.time()
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
            # normally already True from _begin_generation at commit; set
            # again for direct calls without one
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
                # auto_delegate judges on the MERGED user utterance, not the
                # turn's raw ASR fragment: a query arrives fragmented across
                # turns ("帮我查一下。""…诺贝尔物理学奖得主是谁？""详细介绍
                # 他们。"), and the keyword check on the last fragment alone
                # misses queries whose 查/搜 landed earlier (measured
                # 2026-10-30 e2e: acceptance spoken, no delegate_task, the
                # backstop silently skipped on "详细介绍他们。"). llm_client's
                # groupby has already merged the fragments into the last
                # user history entry — and it makes a far better task goal.
                user_text = asr_text
                for m in reversed(llm.get_session(client_id)):
                    if m.get("role") == "user":
                        user_text = m.get("content") or asr_text
                        break
                new_tid = agent_runtime.auto_delegate(
                    user_text, reply_text, task_ids_before)
                if new_tid:
                    logger.info(
                        "[%s] auto-delegated %s: fast brain accepted in "
                        "speech but never called delegate_task (%s)",
                        client_id[:8], new_tid, asr_text[:60],
                    )
        finally:
            with session.lock:
                # only clear if no newer generation began (an interrupted
                # older pipeline's finally must not clear the flag out
                # from under its replacement — see gen_serial)
                if gen_sn is None or session.gen_serial == gen_sn:
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
    _audio_retention_cleanup()

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
        # Partial ASR text streamed from the turn server's asr_buffer.
        # Logged too (2026-11-04): blip partials like '刚才' flash in the
        # UI but left NO web.log trace — the forensics for them could not
        # even confirm WHEN they appeared. The turn server dedupes
        # unchanged buffers, so this is one line per CHANGED partial.
        def _on_partial_transcription(text):
            logger.info(f"[{client_id}] ASR partial: {text}")
            emit_to_room(client_id, "user_transcription", {"text": text})

        vad_instance.transcription_callback = _on_partial_transcription
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
            sn = _begin_generation(session)
            threading.Thread(
                target=pipeline_worker,
                args=(client_id, transcript, Config.SAMPLE_RATE, None, sn),
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
                    session.audio_mic.write(
                        (np.clip(audio_chunk, -1.0, 1.0) * 32767.0)
                        .astype(np.int16)
                        .tobytes()
                    )
                else:
                    session.audio_mic.write(data)   # already int16 @16k LE

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
                    if session.pvad_stream_gate is None and stream_gate_active(session):
                        session.pvad_stream_gate = PvadStreamGate(session)
                    # Pre-ASR stream gate: silence the chunk AFTER the ring /
                    # sidecar / feeder have seen the ORIGINAL audio (the
                    # scorer must keep scoring the real mic or the gate
                    # could never release), so background-voice audio never
                    # reaches ASR / live text / turn commits. Inert unless
                    # enrolled+enforce+actively-suppressing (pvad_stream.py).
                    audio_chunk = stream_gate_filter(session, audio_chunk)
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
                        # unconfirmed nonidle blips: either below the
                        # (now zero) hysteresis or deferred by the
                        # playback guard. Emitting the event lets the UI
                        # and the regression tests see the guard engage —
                        # a silent pass would be indistinguishable from
                        # "echo never classified as nonidle at all".
                        logger.info(
                            f"[{client_id}] transient barge-in suppressed "
                            f"({session.bargein_streak} unconfirmed chunk(s), "
                            f"{len(session.bargein_times)} in "
                            f"{_PLAYBACK_BARGEIN_WINDOW_S:.1f}s window)"
                        )
                        emit_to_room(
                            client_id, "bargein_deferred",
                            {"chunks": session.bargein_streak,
                             "window": len(session.bargein_times)},
                        )
                    session.bargein_streak = 0
                    # bargein_audio is NOT cleared here (gap-tolerant
                    # since 2026-11-05): idle chunks — including the
                    # quiet background-video ones the turn server's rms
                    # gate rewrites to idle — must not zero the user's
                    # interrupt evidence. It is bounded by age
                    # (_BARGEIN_AUDIO_RETAIN_S, evicted on append) and
                    # cleared on turn commit below.
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
                    now = time.time()
                    # (ts, wav) gap-tolerant accumulation — helper in
                    # pvad_gate.py (age-evicted, never cleared by idle)
                    pvad_streak_append(session, now, audio_chunk)
                    # duty-cycle evidence for the playback guard (the
                    # deque is pruned to the window; idle chunks do NOT
                    # clear it — see the Session field note)
                    # LAZY burst reset (2026-11-03): a voice burst is a
                    # run of nonidle chunks with <= window-sized gaps.
                    # Resetting here (on new nonidle after a long gap)
                    # instead of in the idle branch means the episode
                    # count at commit time is always "chunks of the
                    # current burst", so (a) repeated episodic blips
                    # (~2.2 s apart, measured) each read 1-2 and never
                    # accumulate to the threshold — an idle-branch reset
                    # at any gap < blip cadence let the 2nd+ blip's
                    # phantom turn through (2026-11-03 test round), and
                    # (b) a real turn's evidence survives ANY wait-path
                    # silence length (turn_wait_ms is tunable to 3200 ms;
                    # a time-based reset there dropped real turns).
                    if now - session.last_nonidle_ts > _PLAYBACK_BARGEIN_WINDOW_S:
                        session.episode_nonidle = 0
                    session.bargein_times.append(now)
                    session.episode_nonidle += 1
                    session.last_nonidle_ts = now
                    while (session.bargein_times
                           and session.bargein_times[0]
                           < now - _PLAYBACK_BARGEIN_WINDOW_S):
                        session.bargein_times.popleft()
                    if (session.bargein_streak >= _BARGEIN_MIN_CHUNKS
                            and not _playback_bargein_deferred(session, now)):
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
                    if session.bargein_streak:
                        # a deferred barge-in dissolved into a completed
                        # turn without ever confirming — same evidence
                        # signal as the suppression emit above (a blip
                        # that went straight to a turn commit never sees
                        # an idle chunk, so the other emit can't fire)
                        emit_to_room(
                            client_id, "bargein_deferred",
                            {"chunks": session.bargein_streak,
                             "window": len(session.bargein_times),
                             "resolved": "turn"},
                        )
                    run, session.voice_run = session.voice_run, 0
                    session.bargein_streak = 0
                    session.bargein_audio = []
                    # the committed utterance's nonidle evidence must not
                    # vouch for a LATER barge-in: without this clear, a
                    # blip arriving ~0.8 s after the user's last chunk
                    # rides their still-unexpired window count over the
                    # 3-in-1.6s threshold, confirms a barge-in that kills
                    # the just-started generation, and the follow-up
                    # phantom turn then passes with in-flight already
                    # False (2026-11-03 01:30 test round — and the same
                    # structure as the 01:01 live incident). Real
                    # overlapping speech re-accumulates 3 chunks in
                    # ~0.5 s, so only blips lose their free ride.
                    session.bargein_times.clear()
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
                    # In-flight phantom-turn drop (2026-11-02 幻影话轮修复,
                    # path (b) of the playback guard; window widened to the
                    # generation phase 2026-11-03 after session 338b1765):
                    # an echo/noise blip can COMPLETE a turn on the turn
                    # server even though the barge-in itself was deferred —
                    # 1-2 nonidle chunks of faint TTS bleed or breath
                    # residue, with paraformer hallucinating 1-3 characters
                    # onto them ("有。""这么。""是。" — measured in
                    # playback_echo_test phase 1 and live 01:01). The live
                    # case landed in the GENERATION window (reply not yet
                    # audible, playback_end stale): the phantom turn
                    # replaced the user's real sentence in the UI and
                    # entered history. Text below the 4-char floor of the
                    # self-talk check, so the pipeline-side text drop
                    # cannot catch these. episode_nonidle is the
                    # discriminator the retired blip-turn filter lacked:
                    # it counts the WHOLE episode without requiring
                    # consecutiveness, so real browser speech (~50% duty,
                    # 4-8 nonidle per utterance) passes while a blip
                    # episode (1-2) never does.
                    in_flight = _reply_in_flight(session)
                    if (in_flight and _playback_guard_active(session)
                            and session.episode_nonidle
                            < _PLAYBACK_BARGEIN_NONIDLE):
                        # NOTE: deliberately NO sustained-voice fallback
                        # here (unlike the retired filter below): a
                        # confirmed barge-in opens that window too, and a
                        # LOUD echo blip is exactly what confirms a
                        # barge-in — measured 2026-11-03 00:23, the 6 s
                        # fallback let every blip turn after one confirmed
                        # echo barge-in straight through. A real first
                        # turn passes on its own episode evidence; only
                        # post-commit trailing fragments ("……了吧") lose
                        # out, during playback only.
                        logger.info(
                            f"[{client_id}] in-flight blip turn dropped "
                            f"({session.episode_nonidle} nonidle chunk(s) "
                            f"in episode < {_PLAYBACK_BARGEIN_NONIDLE}"
                            f"{', generating' if session.gen_in_progress else ''}): "
                            f"{str(segment)[:40]!r}"
                        )
                        emit_to_room(
                            client_id, "selftalk_drop",
                            {"reason": "blip",
                             "chunks": session.episode_nonidle,
                             "text": str(segment)[:40]},
                        )
                        session.episode_nonidle = 0
                        continue
                    session.episode_nonidle = 0
                    # Complete Utterance — pipeline_worker applies the
                    # main-speaker gate as its first step. Capture the
                    # trigger NOW, in the receive loop: the worker thread
                    # may start later than the next vad.process() call,
                    # which could overwrite last_turn_trigger.
                    turn_trigger = getattr(
                        session.vad, "last_turn_trigger", None
                    )
                    # commit_id for the two-pass ASR correction: captured
                    # NOW for the same reason as turn_trigger — the next
                    # vad.process() overwrites last_commit_id
                    commit_id = getattr(
                        session.vad, "last_commit_id", None
                    )
                    # flag the reply as generating NOW, at commit: the
                    # worker thread's own set comes ~0.5 s later (gates +
                    # LLM connect) and a blip in that gap slipped both
                    # defenses (2026-11-03 01:26 test round)
                    gen_sn = _begin_generation(session)
                    threading.Thread(
                        target=pipeline_worker,
                        args=(client_id, segment, Config.SAMPLE_RATE,
                              turn_trigger, gen_sn, commit_id),
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
            if session.pvad_stream_gate is not None:
                session.pvad_stream_gate.stop()
            session.audio_mic.close()
            session.audio_tts.close()
            vad_pool.release(session.vad)
            session_manager.remove_session(client_id)
        logger.info(f"Session cleaned up: {client_id}")


# ==== Remote log viewer API (read-only, key-gated) ====
# The web port is reachable from outside (scanners hit it daily — see the 404s
# in the tmux panes), so the log endpoints require a shared key: env
# SOULX_LOGS_KEY, or a plain key file (workspace convention — gitignored).
# Served pages: frontend/logs.html.
LOGS_DIR = os.environ.get(
    "SOULX_LOGS_DIR", "/home/research/krisl/soulx/logs"
)
LOGS_KEY_FILE = os.environ.get(
    "SOULX_LOGS_KEY_FILE", "/home/research/krisl/soulx/logs_key.txt"
)
_LOG_TAIL_BYTES = 512 * 1024   # efficient tail: read the last 512 KiB only
_LOG_MAX_TAIL_LINES = 2000
_LOG_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+\.log$")

# ==== Session audio recording (timeline forensics, 2026-11-04) ====
# sessions.html overlays mic/TTS waveforms on the event timeline — to hear
# WHAT a blip actually was (breath? echo? real speech?) the server records
# both streams as raw int16 mono PCM under logs/audio/:
#   mic — post-resample 16 kHz, exactly what the turn server saw
#   tts — the 24 kHz stream emitted to the browser
# Serving is key-gated like the log API (the audio is personal data).
# Disable with SOULX_AUDIO_RECORD=0.
AUDIO_DIR = os.environ.get("SOULX_AUDIO_DIR", str(Path(LOGS_DIR) / "audio"))
_AUDIO_RECORD = os.environ.get("SOULX_AUDIO_RECORD", "1") != "0"
_AUDIO_MAX_SESSION_BYTES = 96 * 1024 * 1024   # per session per kind
_AUDIO_RETENTION_DAYS = 3
# A new recording run starts when a write lands this far beyond the point
# where the audio written so far would have FINISHED PLAYING. TTS chunks
# are written at generation speed (RTF ~0.36 measured 2026-11-05), not at
# playback speed, so "when it sounded" is the run epoch + audio duration
# — the same playback clock _run_speech_pipeline uses for playback_end.
# Mic is written in real time, so for it this reduces to a plain
# inter-write gap check (reconnects/drain stalls start new runs too).
_AUDIO_RUN_GAP_S = 0.5


class _SessionAudio:
    """Best-effort raw PCM recorder for one session/kind.

    Writes never block or raise into the caller: recording is forensics,
    not a feature the conversation depends on.

    Sidecar .json (rewritten on every run boundary):
      {"start": <epoch of byte 0>, "rate": <Hz>,
       "runs": [[byte_offset, epoch, cut_audio_s?], ...]}
    Runs partition the file: bytes [off_i, off_{i+1}) were written
    contiguously and map to session time epoch_i + (off - off_i)/(2*rate).
    A new run starts at an audible gap (see _AUDIO_RUN_GAP_S), at every
    reply pipeline's first emit (run_break), and — with a third element
    "cut" — when a barge-in truncated playback: bytes past the cut were
    emitted but never played, so the timeline must not present them as
    having sounded. Legacy sidecars without "runs" map the whole file as
    one run from "start" (time-compressed for old tts recordings).
    """

    def __init__(self, sid, kind, rate):
        self.rate = rate
        self.path = Path(AUDIO_DIR) / f"{sid}.{kind}.pcm"
        self.start = None
        self.runs = []
        self.n = 0                 # bytes written by this instance
        self._base = 0             # bytes already in the file at open
        self._f = None
        self._lock = threading.Lock()
        self._capped = False
        self._adopted_runs = False
        self._force_break = False
        if not _AUDIO_RECORD:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._f = open(self.path, "ab")
            self._base = self._f.tell()
        except OSError as e:
            logger.warning(f"audio record {kind} unavailable: {e}")
            return
        # Adopt an existing sidecar (sid reused across reconnects / app
        # restarts): keep its start/runs so the timeline stays one file
        # instead of silently re-dating old bytes.
        try:
            meta = json.loads(self.path.with_suffix(".json").read_text())
            if int(meta.get("rate", rate)) == rate and meta.get("start"):
                self.start = float(meta["start"])
                runs = meta.get("runs")
                if isinstance(runs, list):
                    for r in runs:
                        if (isinstance(r, (list, tuple)) and len(r) in (2, 3)
                                and isinstance(r[0], int)
                                and isinstance(r[1], (int, float))):
                            self.runs.append(
                                [r[0], float(r[1])] +
                                ([float(r[2])] if len(r) == 3 else []))
                    self._adopted_runs = bool(self.runs)
                if not self._adopted_runs:
                    # legacy sidecar (pre-2026-11-05): old bytes are one
                    # opaque run from "start"; fresh audio must not glue
                    # onto it (old tts files are time-compressed, so the
                    # gap check alone would not fire)
                    self.runs = [[0, self.start]]
                    self._force_break = True
        except (OSError, ValueError, TypeError):
            pass

    def _save(self):
        try:
            self.path.with_suffix(".json").write_text(json.dumps(
                {"start": self.start, "rate": self.rate, "runs": self.runs}))
        except OSError:
            pass

    def _start_run(self, pos, now):
        if self.runs and pos == self.runs[-1][0]:
            # previous run never got any bytes: re-date it instead of
            # stacking two runs at the same offset
            self.runs[-1][1] = now
        else:
            self.runs.append([pos, now])
        self._save()

    def write(self, pcm):
        if self._f is None:
            return
        with self._lock:
            if self._capped:
                return
            try:
                now = time.time()
                pos = self._base + self.n
                if self.start is None:
                    self.start = now
                    self.runs = [[0, now]]
                    self._save()
                gap = self._force_break
                if not gap and self.runs:
                    off, ep = self.runs[-1][0], self.runs[-1][1]
                    # audio seconds written into the current run so far
                    # = how far playback has progressed since it began
                    played = (pos - off) / (2.0 * self.rate)
                    gap = now > ep + played + _AUDIO_RUN_GAP_S
                if gap:
                    self._force_break = False
                    self._start_run(pos, now)
                self._f.write(pcm)
                self.n += len(pcm)
                if self._base + self.n >= _AUDIO_MAX_SESSION_BYTES:
                    self._capped = True
                    logger.warning(
                        f"audio record {self.path.name} capped at "
                        f"{_AUDIO_MAX_SESSION_BYTES >> 20} MiB"
                    )
            except OSError:
                try:
                    self._f.close()
                except OSError:
                    pass
                self._f = None

    def run_break(self):
        """Force the next write to start a new run. Called at each reply
        pipeline's first emit: a new pipeline means a user turn happened
        in between (gap or barge-in), so playback was discontinuous."""
        with self._lock:
            self._force_break = True

    def mark_cut(self, t, since=None):
        """Barge-in at session epoch t: audio in the last run beyond t was
        emitted but never played. `since` (the pipeline's first-emit time)
        guards the race where a newer pipeline already started its own
        run — only cut a run this pipeline wrote into."""
        with self._lock:
            if not self.runs:
                return
            off, ep = self.runs[-1][0], self.runs[-1][1]
            if since is not None and ep > since:
                return
            pos = self._base + self.n
            played = (pos - off) / (2.0 * self.rate)
            local = t - ep
            if local <= 0 or local >= played:
                return   # nothing audible was lost
            self.runs[-1] = [off, ep, float(local)]
            self._save()

    def close(self):
        with self._lock:
            if self._f is not None:
                try:
                    self._f.close()
                except OSError:
                    pass
                self._f = None


_audio_cleanup_ts = 0.0


def _audio_retention_cleanup():
    """Drop recordings past the retention window (throttled to hourly)."""
    global _audio_cleanup_ts
    now = time.time()
    if now - _audio_cleanup_ts < 3600:
        return
    _audio_cleanup_ts = now
    cutoff = now - _AUDIO_RETENTION_DAYS * 86400
    try:
        entries = list(Path(AUDIO_DIR).iterdir())
    except OSError:
        return
    for p in entries:
        try:
            if p.is_file() and p.stat().st_mtime < cutoff:
                p.unlink()
        except OSError:
            pass


def _logs_api_key() -> str:
    key = os.environ.get("SOULX_LOGS_KEY", "").strip()
    if key:
        return key
    try:
        return Path(LOGS_KEY_FILE).read_text().strip()
    except OSError:
        return ""


def _logs_key_error(request: Request):
    """None when the request may pass; otherwise a 401/503 JSONResponse."""
    expected = _logs_api_key()
    if not expected:
        return JSONResponse(
            status_code=503,
            content={"error": "logs API disabled: no key configured"},
        )
    supplied = request.query_params.get("key", "")
    if not hmac.compare_digest(expected, supplied):
        return JSONResponse(
            status_code=401, content={"error": "bad or missing key"}
        )
    return None


@app.get("/api/logs")
async def list_logs(request: Request):
    err = _logs_key_error(request)
    if err:
        return err
    files = []
    try:
        for p in sorted(Path(LOGS_DIR).glob("*.log"), key=lambda x: x.name):
            st = p.stat()
            files.append(
                {"name": p.name, "size": st.st_size, "mtime": st.st_mtime}
            )
    except OSError:
        pass
    return {"logs": files}


# ==== Structured session timeline API (frontend/sessions.html) ====
# The raw tail+grep viewer answers "which lines exist", not "what happened
# in this session, in what order" — the 2026-11-04 blip forensics
# ('对对对。' committed on 3-chunk breath residue; '刚才' partials that
# flash in the UI but left no log trace) had to be hand-reconstructed from
# a dozen greps. These endpoints parse web.log's tail into per-session
# event streams:
#   /api/logs/sessions        -> session summaries, most recent first
#   /api/logs/sessions/{id}   -> one session's full event list
# Key-gated like the endpoints above. NOTE: both must stay declared
# BEFORE /api/logs/{name} or FastAPI routes them into that catch-all.
_LOG_SESSION_TAIL_BYTES = 16 * 1024 * 1024   # parse window: last 16 MiB
_LOG_SESSION_MAX_EVENTS = 4000               # per-session event cap

_LOG_TS_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3}) \[\w+\] ?(.*)$"
)
_LOG_PREFIX_RE = re.compile(
    r"^\[([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\]"
    r" (.*)$"
)

_LOG_EVENT_PATTERNS = [
    (re.compile(r"^barge-in confirmed"), "bargein"),
    (re.compile(
        r"^transient barge-in suppressed \((\d+) unconfirmed chunk\(s\), "
        r"(\d+) in"), "suppressed"),
    (re.compile(
        r"^ASR (?:result|corrected in [\d.]+s)(?: \((\w+) turn\))?: (.*)"),
     "turn"),
    (re.compile(r"^ASR partial: (.*)"), "partial"),
    (re.compile(r"^LLM Chunk: (.*)"), "chunk"),
    (re.compile(r"^Previous turn, Ratio: ([\d.]+), Truncated: (.*)"),
     "prev_ratio"),
    (re.compile(
        r"^Interrupted mid-stream\. Ratio: ([\d.]+)\. Truncated message: "
        r"(.*)"), "interrupted"),
    (re.compile(r"^in-flight blip turn dropped \((\d+) nonidle.*?\): (.*)"),
     "drop_inflight"),
    (re.compile(r"^playback blip turn dropped \((\d+) nonidle.*?\): (.*)"),
     "drop_playback"),
    (re.compile(
        r"^self-talk turn dropped during playback \(pvad off\): (.*)"),
     "drop_selftalk"),
    (re.compile(r"^premature turn dropped"), "drop_premature"),
    (re.compile(r"^Delivering background task (\S+)"), "deliver"),
    (re.compile(r"^auto-delegated (\S+)"), "delegate"),
    (re.compile(r"^Session manually stopped"), "stopped"),
    (re.compile(r"^WebSocket error: (.*)"), "error"),
    (re.compile(r"^vad\.process took ([\d.]+)s"), "slow_warn"),
    (re.compile(r"^Settings updated: (.*)"), "settings"),
]

_session_log_cache = {"key": None, "result": None}


def _parse_log_sessions() -> dict:
    """web.log tail -> {sessions, truncated, records} (mtime+size cached)."""
    path = Path(LOGS_DIR) / "web.log"
    try:
        st = path.stat()
    except OSError:
        return {"sessions": [], "truncated": False, "records": {}}
    cache_key = (st.st_mtime_ns, st.st_size)
    if _session_log_cache["key"] == cache_key:
        return _session_log_cache["result"]

    import ast
    from datetime import datetime

    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - _LOG_SESSION_TAIL_BYTES))
        lines = f.read().decode("utf-8", "replace").splitlines()
    truncated = size > _LOG_SESSION_TAIL_BYTES
    if truncated and lines:
        lines = lines[1:]   # first line is a partial read

    records = {}     # full id -> {id, start, end, events, settings}
    open_ids = set()

    def _sess(full_id, ts):
        rec = records.get(full_id)
        if rec is None:
            rec = {"id": full_id, "start": ts, "end": ts,
                   "events": [], "settings": None}
            records[full_id] = rec
        if ts > rec["end"]:
            rec["end"] = ts
        return rec

    last_event = None   # multi-line continuations (LLM chunks, tracebacks)
    for line in lines:
        m = _LOG_TS_RE.match(line)
        if not m:
            # continuation of the previous event's text, or unattributable
            # service noise (VadClient prints, uvicorn access lines)
            if (last_event is not None
                    and not line.startswith(("[VadClient]", "INFO:"))):
                last_event["text"] = (
                    last_event.get("text", "") + "\n" + line
                ).strip()
            continue
        ts = (datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
              + int(m.group(2)) / 1000.0)
        rest = m.group(3)
        last_event = None

        pm = _LOG_PREFIX_RE.match(rest)
        if pm is not None:
            rec = _sess(pm.group(1), ts)
            msg = pm.group(2)
        elif rest.startswith("New connection: "):
            full = rest[len("New connection: "):].split()[0]
            open_ids.add(full)
            _sess(full, ts)
            continue
        elif rest.startswith("Session cleaned up: "):
            full = rest[len("Session cleaned up: "):].split()[0]
            open_ids.discard(full)
            _sess(full, ts)
            continue
        elif rest.startswith("[slowbrain") and len(open_ids) == 1:
            # slowbrain lines carry no session prefix; attribute them only
            # when exactly one session is open (multi-tab is ambiguous)
            rec = _sess(next(iter(open_ids)), ts)
            if len(rec["events"]) < _LOG_SESSION_MAX_EVENTS:
                ev = {"t": round(ts, 3), "k": "slowbrain", "text": rest}
                rec["events"].append(ev)
                last_event = ev
            continue
        else:
            continue   # uvicorn access lines, VadClient prints, ...

        for rx, kind in _LOG_EVENT_PATTERNS:
            em = rx.match(msg)
            if em is None:
                continue
            if kind == "settings":
                try:
                    d = ast.literal_eval(em.group(1))
                except (ValueError, SyntaxError):
                    break
                if isinstance(d, dict):
                    rec["settings"] = {
                        k: d.get(k) for k in (
                            "pvad_mode", "turn_engine", "playback_guard",
                            "turn_wait_ms", "turn_grace_ms", "tts_engine",
                            "agent_enabled",
                        )
                    }
                break
            ev = {"t": round(ts, 3), "k": kind}
            g = em.groups()
            if kind == "suppressed":
                ev["n"], ev["w"] = int(g[0]), int(g[1])
            elif kind in ("prev_ratio", "interrupted"):
                ev["ratio"], ev["text"] = float(g[0]), g[1]
            elif kind in ("drop_inflight", "drop_playback"):
                ev["n"], ev["text"] = int(g[0]), g[1]
            elif kind in ("turn", "partial", "chunk", "drop_selftalk",
                          "error"):
                # "turn" lines since 2026-11-05 carry the trigger path:
                # "ASR result (wait turn): ..." — old lines have no group
                ev["text"] = g[-1]
                if kind == "turn" and g[0]:
                    ev["trigger"] = g[0]
            elif kind in ("deliver", "delegate"):
                ev["task"] = g[0]
            elif kind == "slow_warn":
                ev["dt"] = float(g[0])
            if len(rec["events"]) < _LOG_SESSION_MAX_EVENTS:
                rec["events"].append(ev)
                last_event = ev
            break

    summaries = []
    for rec in records.values():
        counts = {}
        for e in rec["events"]:
            counts[e["k"]] = counts.get(e["k"], 0) + 1
        summaries.append({
            "id": rec["id"],
            "start": rec["start"],
            "end": rec["end"],
            "n_events": len(rec["events"]),
            "counts": counts,
            "settings": rec["settings"],
        })
    summaries.sort(key=lambda s: s["start"], reverse=True)
    result = {"sessions": summaries, "truncated": truncated,
              "records": records}
    _session_log_cache["key"] = cache_key
    _session_log_cache["result"] = result
    return result


def _audio_info(sid) -> dict:
    """Recording metadata for the timeline (absent kinds = not recorded)."""
    out = {}
    for kind, default_rate in (("mic", 16000), ("tts", 24000)):
        p = Path(AUDIO_DIR) / f"{sid}.{kind}.pcm"
        try:
            n = p.stat().st_size
        except OSError:
            continue
        rate, start, runs = default_rate, None, None
        try:
            meta = json.loads(p.with_suffix(".json").read_text())
            rate = int(meta.get("rate", default_rate))
            start = meta.get("start")
            if isinstance(meta.get("runs"), list):
                runs = meta["runs"]
        except (OSError, ValueError, TypeError):
            pass
        out[kind] = {"rate": rate, "start": start, "bytes": n, "runs": runs}
    return out


@app.get("/api/logs/sessions")
async def log_sessions(request: Request):
    err = _logs_key_error(request)
    if err:
        return err
    result = _parse_log_sessions()
    return {k: v for k, v in result.items() if k != "records"}


@app.get("/api/logs/sessions/{sid}")
async def log_session_detail(sid: str, request: Request):
    err = _logs_key_error(request)
    if err:
        return err
    rec = _parse_log_sessions()["records"].get(sid)
    if rec is None:
        return JSONResponse(status_code=404,
                            content={"error": "no such session"})
    return {
        "id": rec["id"],
        "start": rec["start"],
        "end": rec["end"],
        "settings": rec["settings"],
        "events": rec["events"],
        "audio": _audio_info(sid),
    }


def _wav_header(n_data: int, rate: int) -> bytes:
    """Canonical 44-byte RIFF/WAVE header, mono 16-bit PCM."""
    return (
        b"RIFF" + (36 + n_data).to_bytes(4, "little") + b"WAVE"
        + b"fmt " + (16).to_bytes(4, "little")
        + (1).to_bytes(2, "little")            # PCM
        + (1).to_bytes(2, "little")            # mono
        + rate.to_bytes(4, "little")
        + (rate * 2).to_bytes(4, "little")     # byte rate
        + (2).to_bytes(2, "little")            # block align
        + (16).to_bytes(2, "little")           # bits
        + b"data" + n_data.to_bytes(4, "little")
    )


_LOG_SID_RE = re.compile(r"^[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$")


@app.get("/api/logs/sessions/{sid}/audio/{kind}")
async def log_session_audio(sid: str, kind: str, request: Request):
    """Serve one session's recording as WAV (Range-capable for seeking)."""
    err = _logs_key_error(request)
    if err:
        return err
    if kind not in ("mic", "tts") or not _LOG_SID_RE.match(sid):
        return JSONResponse(status_code=404, content={"error": "not found"})
    pcm_path = Path(AUDIO_DIR) / f"{sid}.{kind}.pcm"
    try:
        n_pcm = pcm_path.stat().st_size
    except OSError:
        return JSONResponse(status_code=404, content={"error": "no recording"})
    rate = 16000 if kind == "mic" else 24000
    total = 44 + n_pcm
    header = _wav_header(n_pcm, rate)

    # The header always describes the FULL file so the browser can map
    # byte offsets to media time even on 206 slices of a live file.
    rng = request.headers.get("range", "").strip()
    m = re.match(r"^bytes=(\d+)-(\d*)$", rng)
    start, end = 0, total - 1
    if m:
        start = min(int(m.group(1)), total - 1)
        if m.group(2):
            end = min(int(m.group(2)), total - 1)

    def gen():
        pos = start
        if pos < 44:
            take = min(end + 1, 44) - pos
            yield header[pos:pos + take]
            pos += take
        if pos <= end:
            with open(pcm_path, "rb") as f:
                f.seek(pos - 44)
                remaining = end - pos + 1
                while remaining > 0:
                    blk = f.read(min(remaining, 1 << 18))
                    if not blk:
                        break
                    remaining -= len(blk)
                    yield blk

    headers = {
        "Accept-Ranges": "bytes",
        "Content-Type": "audio/wav",
        "Content-Length": str(end - start + 1),
        "Cache-Control": "no-store",
    }
    if m:
        headers["Content-Range"] = f"bytes {start}-{end}/{total}"
    return StreamingResponse(
        gen(), status_code=206 if m else 200, headers=headers
    )


@app.get("/api/logs/{name}")
async def read_log(
    name: str, request: Request, tail: int = 200, grep: str = ""
):
    err = _logs_key_error(request)
    if err:
        return err
    # strict name pattern + glob dir => no traversal, no symlink escape
    if not _LOG_NAME_RE.match(name):
        return JSONResponse(status_code=404, content={"error": "no such log"})
    path = Path(LOGS_DIR) / name
    try:
        st = path.stat()
        if not path.is_file():
            raise OSError
    except OSError:
        return JSONResponse(status_code=404, content={"error": "no such log"})
    tail = max(1, min(tail, _LOG_MAX_TAIL_LINES))
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - _LOG_TAIL_BYTES))
        data = f.read()
    lines = data.decode("utf-8", "replace").splitlines()
    truncated = size > _LOG_TAIL_BYTES
    if truncated and lines:
        lines = lines[1:]   # first line is a partial read
    if grep:
        g = grep.lower()
        lines = [l for l in lines if g in l.lower()]
    return {
        "name": name,
        "size": size,
        "mtime": st.st_mtime,
        "truncated": truncated,
        "lines": lines[-tail:],
    }


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
