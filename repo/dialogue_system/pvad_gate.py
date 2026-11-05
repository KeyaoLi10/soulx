"""Main-speaker gate (pvad sidecar) — extracted from app.py 2026-10-30.

Advisory only: the turn controller still runs byte-identically; the gate
is consulted at the two decision points (barge-in, turn completion) and
can only *veto* acting on a decision, never delay or alter the stream.
Identity signal = ECAPA cosine vs the enrolled embedding (measured
separation: same speaker 0.59-0.80 vs different 0.06-0.44; the pvad ONNX
prob itself is kept as live activity telemetry only). Unknown signal
(not enrolled / service down / too little audio) => fail open.

Public entry points (called from app.py's pipeline_worker / receive loop):
  pvad_enabled / pvad_ring_append / pvad_emit_status   — per audio chunk
  pvad_turn_gate / pvad_maybe_auto_enroll / pvad_bank_turn_audio — per turn
  pvad_handle_bargein / pvad_try_enroll                — on demand
Everything else is internal to this module.

Two hooks are wired by app.py after its singletons exist (importing app
from here would be circular): emit_to_room (from emitter.py).
"""
import difflib
import json
import logging
import threading
import time
import wave
from pathlib import Path

import numpy as np

from emitter import emit_to_room
from settings import Config, DEFAULT_SETTINGS

logger = logging.getLogger(__name__)


def pvad_enabled(session) -> bool:
    """Single switch for the whole pvad path (feed / ring / enrollment /
    bank / telemetry / both gates). mode=off => the app runs purely on
    SoulX turn decisions and does ZERO pvad work per audio chunk; the
    code stays reachable so Settings can flip it back live per session."""
    return session.settings.get("pvad_mode", DEFAULT_SETTINGS["pvad_mode"]) != "off"


_PVAD_RING_MAX_S = 30.0   # mic ring retention
# All ECAPA windows below use RAW CONTIGUOUS audio (ring tail / streak
# chunks), never speech-chunk concatenation. Measured (scripts/
# ecapa_window_calib.py, 2026-10-29): with a natural 6s enrollment the
# same speaker scores 0.55-0.70 on 1s windows vs different-speaker
# 0.30-0.47 and assistant-echo 0.26-0.47 — threshold 0.5 sits in that
# gap. Chunk-concatenated enrollment (the old bank format) collapses the
# same-speaker band to 0.17-0.56 and overlaps everything. 0.5s windows
# are useless (same 0.20-0.37); 1s is the minimum usable window.
# gap (measured, /tmp probes 2026-10-29): a 2.5s window of pure speech
# scores 0.57, but 1.0s speech + 1.5s trailing silence scores 0.23 and
# pure silence -0.08 — silence-PADDED windows garbage-score. Hence
# _pvad_ring_tail trims silence from the EDGES (interior pauses stay).
_PVAD_SIL_RMS = 0.005     # 160 ms chunk RMS floor for that edge-trim.
                          # Sits below echo/AEC-residual level (~0.01) so
                          # echo stays visible to the gate.
_PVAD_TURN_TAIL_S = 2.5   # speech-bounded ring tail scored at turn
                          # decisions (longer tails let similar-voiced
                          # strangers drift over threshold)
_PVAD_TURN_MIN_S = 1.2    # turn gate: shorter tail => fail-open
_PVAD_BARGEIN_MIN_S = 0.9 # barge-in gate: streak audio below this is
                          # unscoreable (1s is where ECAPA separates;
                          # 0.5s does not separate at all)
_PVAD_BARGEIN_RETAIN_S = 3.0
                          # Wall-clock retention of bargein_audio (gap-
                          # tolerant accumulation, 2026-11-05): nonidle
                          # chunks survive idle gaps up to this long. The
                          # turn server's pre-latch rms gate (config
                          # nonidle_rms_gate=0.01) rewrites quiet
                          # background-video chunks to idle, so real
                          # interleaved speech+video bursts read as
                          # alternating nonidle/idle — with the old clear-
                          # on-any-idle, interrupt evidence never reached
                          # the 0.9 s ECAPA floor while a video played
                          # (report docs/PVAD背景视频失效调查报告
                          # -20261105.md §4). 3 s spans a 停停停-style
                          # burst (measured bursts: 1.9-3.7 s of nonidle
                          # audio over ~3 s wall) without letting one
                          # burst vouch for the next.
_PVAD_ENROLL_TAIL_S = 6.0 # speech-bounded ring tail banked per turn
_PVAD_ENROLL_MIN_S = 2.5  # less banked audio than this => defer enroll
_PVAD_BANK_MAX_S = 12.0   # enrollment bank retention
_PVAD_REENROLL_MARGIN = 0.15
                          # Once enrolled, a turn must score threshold+this
                          # to re-enroll or bank. Verified-but-marginal
                          # passes (0.50-0.65) are exactly the user+video
                          # MIXED windows of the 2026-11-05 background-
                          # video session: they passed the gate at 0.52-0.69
                          # and re-enrolled the reference from their last-5s
                          # (video-contaminated) tail, after which the real
                          # user decayed 0.235 -> 0.098 and a video-dominant
                          # window scored 0.50-0.55 AGAINST the polluted
                          # reference (report docs/PVAD背景视频失效调查报告
                          # -20261105.md §6). High-margin only keeps the
                          # reference clean; a thin enrollment loses rolling
                          # adaptation, which is the cheaper failure.
_PVAD_FORENSICS_MAX = 200 # gate-decision audio dumps per session (disk
                          # guard; ~50KB/s of wav each)
_PVAD_FORENSICS_CONTEXT_S = 6.0
                          # raw mic context dumped around a gate decision
# (turn-control constants shared with the app's receive loop —
# _BARGEIN_MIN_CHUNKS, _CHUNK_MS, _TURN_GRACE_S, _TASK_MENTION_RE,
# _SUSTAINED_VOICE_WINDOW_S, _BLIPTURN_FILTER — stayed in app.py)
_PVAD_PLAYBACK_GRACE_S = 1.5  # turn-gate playback leniency extends this
# Self-talk echo window (2026-11-05 吞句修复): a chunk's audio can only be
# at the mic around the moment it PLAYS (playback-head estimate ± this
# tail), so older chunks of a long run are not echo sources — matching
# them ate short user corrections quoting minutes-old assistant words.
_SELF_TALK_ECHO_TAIL_S = 4.0
                          # far past playback_end: the turn decision lags
                          # the speech by ~0.8 s, so a turn completing just
                          # after playback ended was still spoken INTO the
                          # AEC-active window


def pvad_ring_append(session, wav) -> None:
    with session.ring_lock:
        session.mic_ring.append((time.time(), wav))
        # Trim to the retention window (chunk count ~ duration/160ms).
        max_chunks = int(_PVAD_RING_MAX_S / 0.16) + 8
        if len(session.mic_ring) > max_chunks:
            del session.mic_ring[: len(session.mic_ring) - max_chunks]


def _pvad_ring_tail(session, seconds, search_s=None, split_gap_s=0.6):
    """Speech-bounded raw mic audio for ECAPA: take the last `search_s`
    (default seconds + 1.5) of the ring, trim silence from the EDGES, cut
    at the last interior silence gap >= `split_gap_s` (keeping the final
    segment = the CURRENT utterance — without this, a turn completing
    right after the user's previous utterance scores a window mixing both,
    which let the echo following a user turn pass AS the user and poison
    the enrollment), then keep the last `seconds`. Natural contiguous
    audio (interior pauses < split_gap_s included) embeds far better than
    concatenated speech chunks; edge-trimming keeps ECAPA off
    silence-padded windows, which garbage-score (see the calibration
    notes above). Returns None when the window holds nothing above the
    RMS floor (turn gate fail-opens; such turns are blips whose ASR text
    is empty anyway)."""
    with session.ring_lock:
        if not session.mic_ring:
            return None
        wav = np.concatenate([c for _, c in session.mic_ring])
    search = seconds + 1.5 if search_s is None else search_s
    n = int(search * Config.SAMPLE_RATE)
    wav = wav[-n:] if wav.shape[0] > n else wav
    n = (wav.shape[0] // 2560) * 2560
    if n == 0:
        return None
    chunks = wav[:n].reshape(-1, 2560)
    rms = np.sqrt(np.mean(chunks * chunks, axis=1))
    loud = rms >= _PVAD_SIL_RMS
    idx = np.where(loud)[0]
    if idx.size == 0:
        return None
    wav = wav[idx[0] * 2560: (idx[-1] + 1) * 2560]
    if split_gap_s:
        # cut at the last interior silence run >= split_gap_s: everything
        # before it belongs to a previous utterance
        gap_chunks = max(2, int(-(-split_gap_s // 0.16)))  # ceil, not 0.16 short
        cut = 0
        run = 0
        for i, is_loud in enumerate(loud[idx[0]: idx[-1] + 1]):
            if is_loud:
                if run >= gap_chunks:
                    cut = i
                run = 0
            else:
                run += 1
        if cut:
            wav = wav[cut * 2560:]
    n = int(seconds * Config.SAMPLE_RATE)
    return wav[-n:] if wav.shape[0] > n else wav


def pvad_bank_turn_audio(session) -> None:
    """Bank the raw audio of a gate-passed, ASR-confirmed turn (called from
    pipeline_worker after the text check, so noise-only turns never enter
    the bank). The enrollment reference must be natural contiguous audio;
    short turns accumulate across turns so enrollment still completes —
    which arms the ECAPA barge-in gate, the actual defense against
    noise/echo interruptions.

    Once enrolled, only identity-VERIFIED turns bank (scored cos >=
    threshold): a fail-open pass is by definition unverified, and banking
    it is how echo-class audio displaces the enrolled speaker (measured
    2026-10-30 — see the pvad_turn_verified field note).

    Once enrolled, only HIGH-MARGIN verified turns bank (cos >= threshold
    + _PVAD_REENROLL_MARGIN): the bank feeds re-enrollment, and marginal
    passes are the mixed windows that poisoned the reference on
    2026-11-05 (see _PVAD_REENROLL_MARGIN)."""
    threshold = session.settings.get(
        "pvad_threshold", DEFAULT_SETTINGS["pvad_threshold"]
    )
    if session.pvad.enrolled:
        if not session.pvad_turn_verified:
            logger.info(
                f"[{session.client_id}] pvad bank skipped: gate pass was not "
                f"identity-verified (fail-open) — reference stays put"
            )
            return
        cos = getattr(session, "pvad_turn_cos", None)
        if cos is not None and cos < threshold + _PVAD_REENROLL_MARGIN:
            logger.info(
                f"[{session.client_id}] pvad bank skipped: verified pass was "
                f"marginal (cos={cos:.3f} < {threshold + _PVAD_REENROLL_MARGIN:.2f}) "
                f"— mixed-audio turns must not feed the reference"
            )
            return
    wav = _pvad_ring_tail(session, _PVAD_ENROLL_TAIL_S)
    if wav is None or wav.shape[0] < int(1.0 * Config.SAMPLE_RATE):
        return
    with session.ring_lock:
        bank = session.pvad_speech_bank
        bank = wav if bank is None else np.concatenate([bank, wav])
        max_n = int(_PVAD_BANK_MAX_S * Config.SAMPLE_RATE)
        if bank.shape[0] > max_n:
            bank = bank[-max_n:]
        session.pvad_speech_bank = bank


def _pvad_enroll_audio(session):
    """Enrollment audio: banked turn audio plus the current raw tail (bank
    first, tail last, so the sidecar's tail-5s embed is mostly the most
    recent turn). Returns (wav, seconds)."""
    tail = _pvad_ring_tail(session, _PVAD_ENROLL_TAIL_S)
    with session.ring_lock:
        bank = session.pvad_speech_bank
    parts, total = [], 0.0
    if bank is not None and bank.shape[0] >= Config.SAMPLE_RATE:
        parts.append(bank)
        total += bank.shape[0] / Config.SAMPLE_RATE
    if tail is not None and tail.shape[0] >= Config.SAMPLE_RATE:
        parts.append(tail)
        total += tail.shape[0] / Config.SAMPLE_RATE
    if not parts:
        return None, 0.0
    return (np.concatenate(parts) if len(parts) > 1 else parts[0], total)


def _pvad_score(session):
    """Turn-gate identity check — worker threads only. Scores the raw ring
    tail; returns (cos, threshold), cos None when unusable (fail-open):
    unenrolled, service down, or too little audio."""
    threshold = session.settings.get(
        "pvad_threshold", DEFAULT_SETTINGS["pvad_threshold"]
    )
    if not session.pvad.enrolled:
        return None, threshold
    wav = _pvad_ring_tail(session, _PVAD_TURN_TAIL_S)
    if wav is None or wav.shape[0] < int(_PVAD_TURN_MIN_S * Config.SAMPLE_RATE):
        return None, threshold
    return session.pvad.score(wav), threshold


def _pvad_forensics_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "logs" / "pvad_forensics"


def _pvad_write_wav(path: Path, wav: np.ndarray) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(Config.SAMPLE_RATE)
        w.writeframes((np.clip(wav, -1.0, 1.0) * 32767.0).astype("<i2").tobytes())


def _pvad_dump_forensics(session, asr_text, cos, reason) -> None:
    """Audio forensics for gate decisions (2026-10-30): dump the mic audio
    around every turn-gate REJECT plus suspicious passes, so phantom
    fragments can be LISTENED to instead of guessed at. Each event writes
    to logs/pvad_forensics/<session8>/<HHMMSSmmm>_<reason>/:
      context.wav — last _PVAD_FORENSICS_CONTEXT_S of raw mic ring (what
                    was actually in the room, untrimmed)
      window.wav  — the exact ECAPA scoring window, when one existed
      meta.json   — text, cos, threshold, mode, playing, enrolled, bank
    Purely diagnostic: capped per session, every failure swallowed, never
    blocks the pipeline."""
    try:
        if session.pvad_forensics_n >= _PVAD_FORENSICS_MAX:
            return
        session.pvad_forensics_n += 1
        ts = time.time()
        name = time.strftime("%H%M%S", time.localtime(ts)) + f"{int(ts * 1000) % 1000:03d}"
        edir = (_pvad_forensics_dir() / session.client_id[:8]
                / f"{name}_{reason}")
        edir.mkdir(parents=True, exist_ok=True)
        # raw context: copy the ring under its lock, keep the tail
        with session.ring_lock:
            ctx = (np.concatenate([c for _, c in session.mic_ring])
                   if session.mic_ring else None)
        if ctx is not None:
            n = int(_PVAD_FORENSICS_CONTEXT_S * Config.SAMPLE_RATE)
            _pvad_write_wav(edir / "context.wav",
                            ctx[-n:] if ctx.shape[0] > n else ctx)
        # the window the gate actually scored (recomputed — same ring,
        # microseconds later, deterministic trim/split)
        win = _pvad_ring_tail(session, _PVAD_TURN_TAIL_S)
        if win is not None and win.shape[0] > 0:
            _pvad_write_wav(edir / "window.wav", win)
        with session.ring_lock:
            bank_s = (0.0 if session.pvad_speech_bank is None
                      else session.pvad_speech_bank.shape[0] / Config.SAMPLE_RATE)
        (edir / "meta.json").write_text(json.dumps({
            "ts": ts,
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)),
            "session": session.client_id,
            "reason": reason,
            "asr_text": asr_text,
            "cos": None if cos is None else round(cos, 4),
            "threshold": session.settings.get(
                "pvad_threshold", DEFAULT_SETTINGS["pvad_threshold"]),
            "mode": session.settings.get(
                "pvad_mode", DEFAULT_SETTINGS["pvad_mode"]),
            "playing": time.time() < session.playback_end + _PVAD_PLAYBACK_GRACE_S,
            "enrolled": bool(session.pvad.enrolled),
            "bank_s": round(bank_s, 2),
        }, ensure_ascii=False, indent=1))
    except Exception as e:  # diagnostics must never break the pipeline
        logger.warning(f"pvad forensics dump failed: {type(e).__name__}: {e}")


def _pvad_norm_text(s: str) -> str:
    """Lowercase, punctuation/space-stripped text for fuzzy comparison."""
    return "".join(c for c in s.lower() if c.isalnum())


def _pvad_selftalk_text(session, asr_text: str) -> bool:
    """Does this turn's ASR text match what the assistant recently SPOKE?

    The echo/self-talk loop's turns are, by definition, transcriptions of
    our own output — while a real user's words are not. This is the one
    discriminator that separates assistant echo from an AEC-mangled main
    speaker during playback (their ECAPA cosines overlap: echo turns
    0.18-0.39 vs mangled user 0.26-0.46, live-measured 2026-10-29).

    Physics window (2026-11-05 吞句修复): the microphone can only pick up
    audio that is playing NOW or ended moments ago — never text from 40
    seconds ago. Before this fix the sources spanned 12 chunks + pending +
    the last 6 history messages (minutes), and short corrections quoting
    the assistant's topic words died on scattered 3-of-5-char 0.6 recalls
    (session 9e727c1f: five consecutive user corrections eaten while the
    matched source was 40-60 s old). Sources are now only chunks whose
    audio is plausibly at the mic: current-run chunks near the playback
    head (pos = min(now - first_emit, total_emitted), the real-time
    playback model the circle-status gate already uses), or older-run
    chunks within the just-interrupted tail. History/pending are never
    echo sources — everything ever spoken passed _speak_chunk."""
    q = _pvad_norm_text(asr_text)
    if len(q) < 4:
        return False  # too short to attribute to anyone
    now = time.time()
    playing = now < session.playback_end + _PVAD_PLAYBACK_GRACE_S
    recent_interrupt = (
        session.interruption_time is not None
        and now - session.interruption_time < _PVAD_PLAYBACK_GRACE_S
    )
    if not (playing or recent_interrupt):
        return False  # nothing audible: an echo is physically impossible
    run = getattr(session, "playback_run_serial", 0)
    first = getattr(session, "playback_run_first_ts", None)
    total = getattr(session, "playback_run_total", 0.0)
    # playback-head estimate; min() freezes it through inter-chunk gen gaps
    pos = None if first is None else min(now - first, total)
    sources = []
    for e in session.assistant_chunks:
        if e.get("run") == run and pos is not None:
            # current run: only chunks at/after the playback head minus the
            # echo tail. Early chunks of a long run are long gone.
            if e["cum"] < pos - _SELF_TALK_ECHO_TAIL_S:
                continue
        elif now - e.get("ts", 0.0) > _PVAD_PLAYBACK_GRACE_S + _SELF_TALK_ECHO_TAIL_S:
            # older run: plausible only just after its playback was cut
            continue
        sources.append(e["text"])
    for src in sources:
        s = _pvad_norm_text(src)
        if len(s) < 4:
            continue
        if q in s or s in q:
            return True
        # TTS->ASR roundtrip garbles a few chars AND often catches only
        # part of a segment ("怎么又喊我婉儿呀" <- "怎么又喊我“我儿”呀，
        # 是不是喊累了"): substring fails on one garbled char and a whole-
        # string ratio is diluted by the unmatched rest. Score instead the
        # RECALL of q in s — what fraction of the turn's text appears in
        # order in our recent output.
        sm = difflib.SequenceMatcher(None, q, s)
        blocks = sm.get_matching_blocks()
        matched = sum(b.size for b in blocks)
        if matched / len(q) < 0.6:
            continue
        # Span efficiency: an echo is a transcription of ONE consecutive
        # stretch of the playing audio, so its matched blocks hug each
        # other in the source too (only garbled chars fall in the gaps).
        # A correction interleaves the user's own words with quoted topic
        # words ("查今天的北京天气" vs "…查一下北京最新的天气情况":
        # 5 matched chars across a 10-char source span = 0.5) — that
        # scattered-in-source shape is a user, not an echo.
        real = [b for b in blocks if b.size]  # drop the (0,0)-size sentinel
        span = real[-1].b + real[-1].size - real[0].b
        if matched / span < 0.6:
            continue
        # Short-turn contiguity: echo transcribes a CONSECUTIVE stretch of
        # the playing audio, so its largest match block is nearly all of q.
        # A short correction quoting topic words scatters singles for the
        # same 0.6 recall ("就查今天的" <- 查/天/的 of "让我查一下星期天
        # 的天气", 3 blocks of size 1) — that shape is a user, not an echo.
        if len(q) < 8 and max(b.size for b in blocks) < len(q) - 2:
            continue
        return True
    return False


def pvad_turn_gate(session, asr_text: str) -> bool:
    """First thing in pipeline_worker after ASR: a rejected turn must not
    touch the ongoing playback or LLM history. Runs in the worker thread,
    so the ~150 ms ECAPA forward never blocks the audio receive loop.

    The gate applies the SAME strict threshold whether or not the
    assistant is playing: below threshold -> reject, unverifiable
    (unenrolled / too little audio / service down) -> fail-open (a
    falsely rejected user is the gate's worst failure) — EXCEPT while
    playing: an unscoreable turn during playback is REJECTED (fail-
    closed, same policy as the barge-in gate since 2026-10-29).
    Playback-onset echo is exactly a short unscoreable fragment; a real
    user talking over playback keeps producing speech and their next
    turn scores and passes. The self-talk text check survives only as
    the reject-reason label.

    (History: 2026-10-29 briefly had a playback-leniency branch that
    passed any non-self-talk below-threshold turn while playing, on the
    theory that browser AEC mangles the enrolled user mid-playback.
    Removed the same day by user decision — tests run with headphones,
    where AEC has nothing to cancel, yet playback-period scores still
    dropped; the real cause was reference drift toward other voices, and
    leniency let those voices through: impostor speech during playback
    got answered by the LLM.)"""
    mode = session.settings.get("pvad_mode", DEFAULT_SETTINGS["pvad_mode"])
    if not pvad_enabled(session):
        # gate off (the default since 2026-10-30): nothing downstream
        # consumes pvad_turn_verified while off, but keep it True so a
        # live mid-session switch to shadow/enforce starts from a clean
        # "last turn trusted" state
        session.pvad_turn_verified = True
        return True
    cos, threshold = _pvad_score(session)
    # stash for the bank/re-enroll calls later in this pipeline run: only
    # a SCORED pass may move the enrollment reference (fail-open passes
    # must not bank — that is the pollution vector of 2026-10-30), and
    # once enrolled only a HIGH-MARGIN pass may (_PVAD_REENROLL_MARGIN,
    # the mixed-window pollution vector of 2026-11-05)
    session.pvad_turn_verified = cos is not None and cos >= threshold
    session.pvad_turn_cos = cos
    if cos is not None and cos >= threshold:
        return True  # strict pass — clean capture of the enrolled speaker
    playing = time.time() < session.playback_end + _PVAD_PLAYBACK_GRACE_S
    if playing and cos is None and session.pvad.enrolled:
        # ENROLLED but unscoreable while playing: REJECT (fail-closed) —
        # the same policy the barge-in gate has used since 2026-10-29.
        # Playback onset produces exactly this signature: the ring-tail
        # split at the last interior silence keeps only the ~0.3 s onset
        # fragment, ECAPA can't score it, and the old fail-open here is
        # what let onset-echo turns through (2026-10-30: a TTS-onset
        # fragment passed as a "turn" one second after the user's real
        # one, and the LLM answered it). A real user talking over
        # playback keeps producing speech — their next turn scores
        # properly and passes ~0.5 s later; an echo blip produces
        # exactly one unscoreable fragment and stops. NOT applied while
        # unenrolled: enrollment bootstraps from playback-period turns
        # (users talk over replies), and an unscoreable onset fragment
        # is too short to bank anyway, so it can't pollute the bank.
        # The self-talk text check stays only as the reject-reason label.
        selftalk = _pvad_selftalk_text(session, asr_text)
        _pvad_dump_forensics(
            session, asr_text, cos,
            "selftalk" if selftalk else "unscoreable-playing")
        emit_to_room(
            session.client_id,
            "pvad_reject",
            {
                "kind": "turn",
                "cos": None,
                "threshold": threshold,
                "shadow": mode == "shadow",
                "reason": "selftalk" if selftalk else "unscoreable-playing",
            },
        )
        logger.info(
            f"[{session.client_id}] pvad rejected turn (unscoreable while "
            f"playing{'/self-talk' if selftalk else ''}): (mode={mode})"
        )
        return mode == "shadow"  # shadow reports but proceeds
    if cos is None:
        if playing:
            # unenrolled (or service-blind) while playing: suspicious
            # pass — capture it for forensics, but let it through
            # (enrollment bootstraps from playback-period turns)
            _pvad_dump_forensics(session, asr_text, cos,
                                 "pass-unenrolled-playing")
        elif session.pvad.enrolled:
            # enrolled + unscoreable + NOT playing: the fail-open short-
            # window channel. On 2026-11-05 this is how a pure video
            # phrase (「嗯，刚洗完，你这怎么天天晚？」, <1.2 s window)
            # reached the LLM. Fail-open stays (a falsely rejected user
            # is still the worst failure, and short confirmations like
            # 「好的」 are exactly this signature) — but it must be
            # VISIBLE in forensics so leaks can be tuned, not silent.
            _pvad_dump_forensics(session, asr_text, cos,
                                 "pass-unscoreable-short")
        return True  # fail-open (unenrolled / unscoreable / no service)
    _pvad_dump_forensics(session, asr_text, cos, "below-threshold")
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


def _pvad_do_bargein(session) -> None:
    # One interrupt per episode: while the user keeps talking, the streak
    # re-reaches the threshold every ~0.5 s — everything after the first
    # interrupt is a no-op that only spams stop_audio to the frontend.
    if session.bargein_done:
        return
    session.bargein_done = True
    if session.interruption_time is None:
        session.interruption_time = time.time()
    logger.info(
        f"[{session.client_id}] barge-in confirmed: stopping playback/generation"
    )
    session.interrupt()
    # forensic recording: the browser drops its buffer at this stop, but
    # the tts sidecar may hold an emitted-but-unplayed tail (TTS emits
    # faster than real time) — cut where audible playback actually ended.
    # No-op unless a fully-emitted reply is pending; an in-flight
    # pipeline cuts its own run in its interrupted branch.
    session.cut_pending_audio()


def pvad_streak_append(session, now, wav) -> None:
    """Receive-loop hook: accumulate a nonidle chunk into bargein_audio.
    GAP-TOLERANT (2026-11-05): entries are (ts, wav) and survive idle
    chunks — evicted only by age (_PVAD_BARGEIN_RETAIN_S above) or the
    16-chunk audio cap, and cleared on turn commit. With a background
    video playing, the turn server's pre-latch rms gate rewrites the
    quiet video chunks to idle, and the old clear-on-any-idle kept
    zeroing the user's interrupt evidence mid-shout (10 consecutive
    "unscoreable" rejections in 1.7 s on 2026-11-05; report docs/
    PVAD背景视频失效调查报告-20261105.md §4)."""
    session.bargein_audio.append((now, wav))
    while (session.bargein_audio
           and now - session.bargein_audio[0][0] > _PVAD_BARGEIN_RETAIN_S):
        del session.bargein_audio[0]
    if len(session.bargein_audio) > 16:  # ~2.5 s of chunks: score window
        del session.bargein_audio[0]


def pvad_handle_bargein(session) -> None:
    """Barge-in decision point. Runs in a daemon thread when enrolled (the
    receive loop must not wait ~150 ms per barge-in signal — they can fire
    many times per second during overlapping speech). An in-flight check
    absorbs duplicate signals: whichever way it resolves, it acts once.

    Identity is scored on session.bargein_audio — the nonidle chunks that
    actually triggered the barge-in — NOT on the mic ring: the ring's
    recent-speech window can still hold the user's just-finished turn,
    which made the first echo/noise onset score as the main speaker.

    Fail-open is conditional on playback: with nothing playing there is
    nothing to protect, so unknown signals interrupt as before. While the
    assistant is playing, an unverifiable barge-in (unenrolled, or less
    than ~0.9 s of streak audio — ECAPA does not separate speakers on
    shorter windows, and a short unverifiable onset is exactly what the
    first blip of echo/noise looks like) is REJECTED: allowing it is the
    choppiness bug. A genuine user keeps talking, the streak re-crosses
    the threshold with more audio, and the check passes (~0.5 s later)."""
    mode = session.settings.get("pvad_mode", DEFAULT_SETTINGS["pvad_mode"])
    if not pvad_enabled(session):
        _pvad_do_bargein(session)
        return
    playing = time.time() < session.playback_end

    if not session.pvad.enrolled:
        if playing:
            emit_to_room(
                session.client_id,
                "pvad_reject",
                {
                    "kind": "bargein",
                    "cos": None,
                    "threshold": session.settings.get(
                        "pvad_threshold", DEFAULT_SETTINGS["pvad_threshold"]
                    ),
                    "shadow": mode == "shadow",
                    "reason": "unenrolled",
                },
            )
            logger.info(
                f"[{session.client_id}] pvad rejected barge-in: "
                f"unenrolled while playing (mode={mode})"
            )
            return
        _pvad_do_bargein(session)
        return

    if session.pvad_checking:
        return
    session.pvad_checking = True

    # snapshot the streak audio under the same loop that appends it.
    # (ts, wav) pairs since 2026-11-05 — gap-tolerant accumulation: the
    # chunks can span idle gaps up to _PVAD_BARGEIN_RETAIN_S, so the
    # snapshot may hold nonidle audio from both sides of a quiet
    # background-video patch. That is the point: the user's interrupt
    # evidence must survive the video's interleaved quiet chunks.
    streak_wav = (
        np.concatenate([c for _, c in session.bargein_audio])
        if session.bargein_audio
        else None
    )

    def _worker():
        try:
            threshold = session.settings.get(
                "pvad_threshold", DEFAULT_SETTINGS["pvad_threshold"]
            )
            if (
                streak_wav is not None
                and streak_wav.shape[0] >= _PVAD_BARGEIN_MIN_S * Config.SAMPLE_RATE
            ):
                cos = session.pvad.score(streak_wav)
            else:
                cos = None
            if cos is None:
                allow = not playing or mode == "shadow"
                if not allow:
                    emit_to_room(
                        session.client_id,
                        "pvad_reject",
                        {
                            "kind": "bargein",
                            "cos": None,
                            "threshold": threshold,
                            "shadow": mode == "shadow",
                            "reason": "unscoreable",
                        },
                    )
                    logger.info(
                        f"[{session.client_id}] pvad rejected barge-in: "
                        f"unscoreable while playing (mode={mode})"
                    )
            else:
                allow = cos >= threshold or mode == "shadow"
                if not allow:
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


def pvad_try_enroll(session, reason: str) -> None:
    """Enroll the main speaker from banked turn audio plus the raw ring
    tail (natural contiguous speech — chunk-concatenated enrollment
    collapses ECAPA's same-speaker scores, see the calibration note at
    the constants). Blocking HTTP runs in a daemon thread; defers to the
    next gate-passed turn if there is too little audio (<2.5 s) or the
    service errors."""

    def _worker():
        wav, total_s = _pvad_enroll_audio(session)
        if wav is None or total_s < _PVAD_ENROLL_MIN_S:
            session.pvad_enroll_pending = True
            logger.info(
                f"[{session.client_id}] pvad enroll deferred ({reason}): "
                f"bank+tail hold {total_s:.1f}s of audio (need {_PVAD_ENROLL_MIN_S}s)"
            )
            return
        ok = session.pvad.enroll(wav)
        if ok:
            session.pvad_enroll_pending = False
            logger.info(
                f"[{session.client_id}] pvad enrolled ({reason}, "
                f"{total_s:.1f}s audio)"
            )
        else:
            session.pvad_enroll_pending = True

    threading.Thread(target=_worker, daemon=True).start()


def pvad_maybe_auto_enroll(session) -> None:
    """After every gate-passed turn: enroll (or RE-enroll) the main
    speaker. First turn wins the initial reference; later turns keep it
    rolling — measured same-speaker drift (vocal register, distance to
    the mic) drops later windows to 0.33-0.50 against a stale first-turn
    reference, right at the reject boundary.

    Once enrolled, re-enrollment requires an identity-VERIFIED pass this
    turn (scored cos >= threshold). The old claim that "an impostor must
    already score >= threshold to move the reference" was FALSE for
    fail-open passes: an unscoreable phantom turn passed the gate, and
    its echo-class ring tail re-enrolled the reference — after which the
    real user scored ~0 and was rejected on every later turn while
    phantoms kept passing (2026-10-30, session b63c185f).

    2026-11-05: verified is no longer enough — the pass must also be
    HIGH-MARGIN (cos >= threshold + _PVAD_REENROLL_MARGIN). The
    background-video session showed verified-but-marginal turns (0.52-
    0.69) were user+video MIXED windows whose last-5s enroll tail
    dragged the reference toward the video voice."""
    if session.pvad.enrolled:
        if not session.pvad_turn_verified:
            return  # unverified (fail-open) pass: never move the reference
        cos = getattr(session, "pvad_turn_cos", None)
        threshold = session.settings.get(
            "pvad_threshold", DEFAULT_SETTINGS["pvad_threshold"]
        )
        if cos is not None and cos < threshold + _PVAD_REENROLL_MARGIN:
            return  # marginal pass: not clean enough to move the reference
    pvad_try_enroll(session, "auto")


def pvad_emit_status(session) -> None:
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
            # pre-ASR stream gate (pvad_stream.py): True while mic->ASR is
            # being silenced as non-main-speaker audio (UI badge + the
            # regression tests read this)
            "gated": bool(
                session.pvad_stream_gate is not None
                and session.pvad_stream_gate.suppressing()
            ),
            # shadow separation observation (speaker_sep switch, default
            # off): latest separated-stream cos vs the enrolled reference
            "sep_cos": (
                session.pvad_stream_gate.sep_status()["sep_cos"]
                if session.pvad_stream_gate is not None
                and session.pvad_stream_gate.sep_status() is not None
                else None
            ),
        },
    )

