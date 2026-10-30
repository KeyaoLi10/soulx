"""FireRedChat-architecture frontend mode (settings turn_engine="frc").

Optional second voice-frontend processing path, ported from
full_duplex/scripts/frc_worker_integrated.py (the FireRedChat dual-EOT FDB
worker). For a session in "frc" mode the SoulX turn server is bypassed
entirely and turn-taking follows the FireRedChat recipe instead:

  mic chunk (160 ms @16k)
    → pvad sidecar (:8086, UNPERSONALIZED — zero speaker embedding) as the
      speech-activity VAD; sustained prob >= activation_threshold = voice
    → end-of-speech: MIN_SILENCE_S of consecutive low-prob chunks
    → dual end-of-turn decision, both channels in parallel:
        ultraVAD  (:8082 /predict) — 20 s rolling audio window + the last
                                     assistant reply as turns context
        Qwen3.5   (:6007 /v1/score) — first-token 是/否 scoring of an EOT
                                     judgment prompt over the utterance
                                     transcript (FireRedASR, :8010) + the
                                     last assistant reply
      fused with MIN: a fast commit needs BOTH channels above threshold
    → P(eot) >= threshold: commit (silence already >= min_endpointing_delay
      by construction); else hold until max_endpointing_delay after the
      end-of-speech, committing anyway at the cap; the user resuming speech
      cancels the pending commit and the utterance keeps growing
  commit → FireRedASR transcribes the whole utterance → the SAME
    pipeline_worker as the SoulX path (speaker gate / history / fast-slow
    brains / TTS), with no extra grace window — the dual-EOT decision IS
    the anti-premature-turn machinery here.

TTS (VoxCPM2 :8085) and the LLMs (:6007 / :6008) are exactly the same
services as the SoulX path — only turn-taking differs. The SoulX path
itself is untouched: this module is only fed audio when the session's
turn_engine setting says "frc" (default "soulx").

Known behavioral differences vs the SoulX engine (inherent to the FRC
architecture, by design):
  - no live partial transcripts (FireRedASR is utterance-final; the user
    bubble fills only when the turn commits)
  - barge-in uses the pvad prob instead of the turn server's nonidle
    state, with the same 3-chunk hysteresis
"""
import base64
import io
import queue
import threading
import time
import wave
from collections import deque

import numpy as np
import requests

# ---------------------------------------------------------------------------
# services & tuning (mirrors frc_worker_integrated.py defaults)
# ---------------------------------------------------------------------------

ULTRAVAD_BASE = "http://127.0.0.1:8082"
FRC_ASR_BASE = "http://127.0.0.1:8010"
LLM_SCORE_URL = "http://127.0.0.1:6007/v1/score"

FRC_ACTIVATION_THRESHOLD = 0.5   # pvad prob above this = voice
FRC_ONSET_CHUNKS = 3             # sustained chunks before speech is real
                                 # (same hysteresis as the SoulX barge-in)
FRC_MIN_SILENCE_S = 0.40         # silence before end-of-speech
FRC_MIN_SPEECH_CHUNKS = 3        # < this much speech in an utterance = noise
FRC_EOT_THRESHOLD = 0.2          # fused P(eot) for a fast commit
FRC_MAX_ENDPOINTING_DELAY_S = 6.0  # commit anyway this long after EOS
FRC_RING_S = 20.0                # ultraVAD rolling window
FRC_UTT_MAX_S = 60.0             # force an EOS on an endless monologue
FRC_PREROLL_S = 2.5              # audio kept before onset (anti-clipping)

CHUNK_SAMPLES = 2560             # 160 ms @ 16 kHz
CHUNK_S = CHUNK_SAMPLES / 16000.0

HTTP_TIMEOUT_S = 8.0
ASR_TIMEOUT_S = 20.0

# Verbatim from frc_worker_integrated.py — forbids punctuation-based
# judgment (ASR punctuation is unreliable) and forces a one-token 是/否
# answer, which is what /v1/score's first-token logits measure.
EOT_SYSTEM_PROMPT = """你是一个实时语音对话中的话轮结束(end-of-turn)判定器。给你当前对话上下文和用户最新一句话的语音识别(ASR)转写,判断用户是否已经说完了这一轮话,还是在停顿后还会继续说。

判断规则:
- 转写来自ASR,其中的标点完全不可靠:可能整句没有任何标点,也可能在句中任意位置插入逗号句号。绝对不要基于句尾有没有标点、标点是什么类型、或者转写"看起来像不像一句完整的话"来做判断。
- 只根据语义和语用判断:这句话在语义上是否已经构成一个完整的提问、请求或陈述,是否在语义上等待对方回应;还是明显是半句话,后面还会继续(例如:"我想问一下"、"然后"、"就是说"这类悬置开头,列举到一半,话题还没展开等)。
- 如果用户的话语义上已经说完了,即使转写没有任何标点,也应该回答"是"。
- 如果用户的话语义上是半句话,即使转写以句号结尾,也应该回答"否"。

用户是否已经说完了这一轮话?只回答"是"或"否\""""

import logging
logger = logging.getLogger(__name__)


def services_available():
    """Cheap liveness probe for the two FRC-only services (ultraVAD, ASR)."""
    for url in (f"{ULTRAVAD_BASE}/health", f"{FRC_ASR_BASE}/"):
        try:
            requests.get(url, timeout=2).raise_for_status()
        except requests.RequestException:
            return False
    return True


class FrcEngine:
    """Per-session FireRedChat-style turn engine.

    feed(chunk) is called from the websocket receive loop (non-blocking
    queue put); a single worker thread owns all state and does every HTTP
    call off the audio path. commit_cb(transcript) fires when a turn is
    committed (the app launches the shared pipeline_worker); context_cb()
    returns the last assistant reply text for the EOT prompts.
    """

    def __init__(self, session, client_id, commit_cb, context_cb):
        self.session = session
        self.client_id = client_id
        self.commit_cb = commit_cb
        self.context_cb = context_cb
        self._q = queue.SimpleQueue()
        self._stop = threading.Event()
        self._http = requests.Session()
        # engine state (touched by the worker thread + the decision thread;
        # _lock guards the utterance list and the decision bookkeeping)
        self._lock = threading.Lock()
        self._ring = []          # [np.ndarray] rolling ALL-audio window
        self._ring_n = 0
        self._pre = deque(maxlen=int(FRC_PREROLL_S / CHUNK_S))
        # monotonic decision generation: bumped on every end-of-speech so a
        # late HTTP result from a superseded decision (speech resumed, new
        # EOS fired) can never act on stale state
        self.decision_gen = 0
        self._reset_state()
        self._thread = None

    # ------------------------------------------------------------------
    def _reset_state(self):
        self.state = "idle"       # idle | speech | deciding
        self.onset_streak = 0
        self.speech_chunks = 0
        self.silence_chunks = 0
        self.utt = None           # current utterance chunks (list) or None
        self.utt_n = 0
        self.eos_ts = None
        self.commit_at = None     # max-delay commit deadline (wall clock)
        self.pending_text = None  # transcript held for a delayed commit
        self.interrupted_episode = False

    # ------------------------------------------------------------------
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run, daemon=True,
            name=f"frc-engine-{self.client_id[:8]}",
        )
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._q.put(np.zeros(0, dtype=np.float32))

    def feed(self, chunk):
        """Queue one mic chunk (float32 @16k). Never blocks."""
        self._q.put(chunk.copy())

    def reset(self):
        """Drop any half-finished utterance / pending commit (engine
        switch mid-session). Safe to call from any thread."""
        self._q.put(None)  # control sentinel handled by the worker thread

    def available(self):
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # worker thread: consumes chunks at real-time cadence
    # ------------------------------------------------------------------
    def _run(self):
        while not self._stop.is_set():
            try:
                chunk = self._q.get(timeout=0.25)
            except queue.Empty:
                continue
            if chunk is None:
                # reset sentinel (engine switch): abandon the current
                # utterance and any pending commit
                with self._lock:
                    self.decision_gen += 1
                    self._reset_state()
                logger.info("[%s] frc: engine state reset",
                            self.client_id[:8])
                continue
            if chunk.size == 0:
                break
            try:
                self._on_chunk(chunk)
            except Exception:
                logger.exception(
                    "[%s] frc engine chunk error", self.client_id[:8]
                )

    def _on_chunk(self, chunk):
        now = time.time()
        self._ring_push(chunk)
        self.session.pvad.feed(chunk)
        prob = self.session.pvad.latest_prob()
        active = prob is not None and prob >= FRC_ACTIVATION_THRESHOLD

        if self.state == "idle":
            self._pre.append(chunk)
            if active:
                self.onset_streak += 1
                if self.onset_streak >= FRC_ONSET_CHUNKS:
                    # sustained voice onset: real speech (blips never
                    # sustain 3 chunks — same evidence class as the SoulX
                    # barge-in hysteresis)
                    self.onset_streak = 0
                    self._emit_circle("LISTENING")
                    self._emit_status("listening", "Voice detected")
                    self._maybe_interrupt_playback()
                    with self._lock:
                        self.state = "speech"
                        self.utt = list(self._pre)  # pre-roll: keep onsets
                        self.utt_n = sum(c.size for c in self.utt)
                        self.speech_chunks = 0
                        self.silence_chunks = 0
            else:
                self.onset_streak = 0
            return

        # speech / deciding: the utterance keeps collecting, silence
        # included (the trailing pause is part of the utterance audio and
        # FireRedASR trims it fine)
        with self._lock:
            self.utt.append(chunk)
            self.utt_n += chunk.size

        if active:
            with self._lock:
                if self.state == "deciding":
                    # the user resumed before the pending commit — cancel
                    # it, the utterance keeps growing (this is exactly the
                    # "user paused mid-sentence" case the dual-EOT exists
                    # to catch)
                    logger.info(
                        "[%s] frc: speech resumed, pending commit canceled",
                        self.client_id[:8],
                    )
                    self.commit_at = None
                    self.pending_text = None
                    self.state = "speech"
                self.silence_chunks = 0
                self.speech_chunks += 1
            self.session.last_voice_ts = now
            self._maybe_interrupt_playback()
        else:
            with self._lock:
                self.silence_chunks += 1
                if self.state == "speech":
                    if self.speech_chunks < FRC_MIN_SPEECH_CHUNKS and \
                            self.silence_chunks * CHUNK_S >= FRC_MIN_SILENCE_S:
                        # too little actual speech for a real utterance —
                        # noise blip the pvad prob briefly cleared
                        logger.info(
                            "[%s] frc: noise blip dropped (%d speech chunks)",
                            self.client_id[:8], self.speech_chunks,
                        )
                        self._reset_state()
                        return
                    if (self.silence_chunks * CHUNK_S >= FRC_MIN_SILENCE_S
                            or self.utt_n >= FRC_UTT_MAX_S * 16000):
                        self._end_of_speech(now)
                elif self.state == "deciding":
                    if self.commit_at is not None and now >= self.commit_at:
                        # held turn: EOT said "probably not done" but the
                        # max-delay cap passed without resumed speech —
                        # commit anyway (same as AgentSession semantics)
                        self._commit_locked()

    # ------------------------------------------------------------------
    def _end_of_speech(self, now):
        """Called with _lock held, state == speech."""
        self.state = "deciding"
        self.eos_ts = now
        self.commit_at = None
        self.pending_text = None
        self.decision_gen += 1
        gen = self.decision_gen
        threading.Thread(
            target=self._decide, args=(gen,), daemon=True,
            name=f"frc-eot-{self.client_id[:8]}",
        ).start()

    def _decide(self, gen):
        """Dual-EOT decision, off the chunk path. Snapshots the utterance,
        runs ultraVAD + FireRedASR in parallel, then the Qwen first-token
        score on the transcript, fuses with min. `gen` must still be the
        live decision generation when the results land, else the results
        are discarded (speech resumed / a newer decision superseded)."""
        with self._lock:
            audio = np.concatenate(self.utt)
        assistant_text = ""
        try:
            assistant_text = (self.context_cb() or "").strip()
        except Exception:
            pass

        uv_result = {}
        asr_result = {}

        def uv_job():
            uv_result["p"] = self._ultravad_prob(assistant_text)

        def asr_job():
            asr_result["text"] = self._transcribe(audio)

        t_uv = threading.Thread(target=uv_job, daemon=True)
        t_asr = threading.Thread(target=asr_job, daemon=True)
        t_uv.start(); t_asr.start(); t_uv.join(); t_asr.join()

        p_uv = uv_result.get("p")
        text = (asr_result.get("text") or "").strip()

        with self._lock:
            if gen != self.decision_gen or self.state != "deciding":
                return  # superseded: speech resumed / newer decision
            if not text:
                # ASR found nothing speakable — noise, not a turn
                logger.info(
                    "[%s] frc: empty transcript, utterance dropped",
                    self.client_id[:8],
                )
                self._reset_state()
                return

        p_llm = self._llm_prob(text, assistant_text)

        with self._lock:
            if gen != self.decision_gen or self.state != "deciding":
                return
            # an errored channel reads as 0.0 (frc_worker semantics): a
            # fast commit then needs the OTHER channel to carry it alone,
            # which min() enforces naturally
            combined = min(
                p_uv if p_uv is not None else 0.0,
                p_llm if p_llm is not None else 0.0,
            )
            logger.info(
                "[%s] frc P(eot): uv=%s llm=%s -> %.3f (%s) user=%r",
                self.client_id[:8],
                "-" if p_uv is None else f"{p_uv:.3f}",
                "-" if p_llm is None else f"{p_llm:.3f}",
                combined,
                "fast-commit" if combined >= FRC_EOT_THRESHOLD else "hold",
                text[:60],
            )
            if combined >= FRC_EOT_THRESHOLD:
                # the decision itself (ASR + both channels) took well over
                # min_endpointing_delay since the end-of-speech — commit now
                self._commit_locked(text=text)
            else:
                # hold: cap-bounded commit unless speech resumes
                self.pending_text = text
                self.commit_at = (self.eos_ts or time.time()) + \
                    FRC_MAX_ENDPOINTING_DELAY_S

    def _commit_locked(self, text=None):
        """Commit the turn; _lock must be held. Launches the shared
        pipeline (which does its own user_transcription emission)."""
        if text is None:
            text = self.pending_text
        self._reset_state()
        if not text or not text.strip():
            return
        logger.info("[%s] frc turn committed: %r",
                    self.client_id[:8], text[:80])
        self._emit_status("taking turn", "User finished, taking turn")
        try:
            self.commit_cb(text.strip())
        except Exception:
            logger.exception("[%s] frc commit callback failed",
                             self.client_id[:8])

    # ------------------------------------------------------------------
    # service calls (all off the chunk path)
    # ------------------------------------------------------------------

    def _ultravad_prob(self, assistant_text):
        try:
            with self._lock:
                window = np.concatenate(self._ring) if self._ring else \
                    np.zeros(0, dtype=np.float32)
            if window.size < 1600:  # < 0.1 s
                return 0.0
            turns = (
                [{"role": "assistant", "content": assistant_text}]
                if assistant_text else []
            )
            r = self._http.post(
                f"{ULTRAVAD_BASE}/predict",
                json={
                    "audio": base64.b64encode(
                        (np.clip(window, -1, 1) * 32767)
                        .astype("<i2").tobytes()
                    ).decode(),
                    "sample_rate": 16000,
                    "turns": turns,
                },
                timeout=HTTP_TIMEOUT_S,
            )
            r.raise_for_status()
            return float(r.json()["eot_probability"])
        except Exception as e:
            logger.warning("[%s] ultravad error: %s", self.client_id[:8], e)
            return None

    def _llm_prob(self, user_text, assistant_text):
        try:
            parts = []
            if assistant_text:
                parts.append(f"之前的助手回复:{assistant_text}")
            parts.append(f"用户最新话语(ASR转写):{user_text}")
            r = self._http.post(
                LLM_SCORE_URL,
                json={
                    "messages": [
                        {"role": "system", "content": EOT_SYSTEM_PROMPT},
                        {"role": "user", "content": "\n".join(parts)},
                    ],
                    "candidates": ["是", "否"],
                },
                timeout=HTTP_TIMEOUT_S,
            )
            r.raise_for_status()
            return float(r.json()["probabilities"]["是"])
        except Exception as e:
            logger.warning("[%s] llm eot score error: %s",
                           self.client_id[:8], e)
            return None

    def _transcribe(self, audio_f32):
        try:
            pcm16 = (np.clip(audio_f32, -1, 1) * 32767).astype("<i2")
            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(pcm16.tobytes())
            r = self._http.post(
                f"{FRC_ASR_BASE}/audio/transcriptions",
                files={"file": ("utt.wav", buf.getvalue(), "audio/wav")},
                timeout=ASR_TIMEOUT_S,
            )
            r.raise_for_status()
            sents = r.json().get("sentences") or []
            return " ".join(s.get("text", "") for s in sents).strip()
        except Exception as e:
            logger.warning("[%s] fireredasr error: %s", self.client_id[:8], e)
            return ""

    # ------------------------------------------------------------------
    # small helpers
    # ------------------------------------------------------------------

    def _ring_push(self, chunk):
        self._ring.append(chunk)
        self._ring_n += chunk.size
        while self._ring_n > int(FRC_RING_S * 16000) and len(self._ring) > 1:
            self._ring_n -= self._ring[0].size
            self._ring.pop(0)

    def _emit_circle(self, status):
        # same playback-window gating as the websocket endpoint's
        # _circle_gated wrapper: LISTENING must not flip the ball while
        # buffered assistant audio is still playing
        if status == "LISTENING" and time.time() < self.session.playback_end:
            return
        self._emit("circle_status", {"status": status})

    def _emit_status(self, state, message):
        self._emit("vad_status", {"state": state, "message": message})

    def _emit(self, event, data):
        try:
            from emitter import emit_to_room
            emit_to_room(self.client_id, event, data)
        except Exception:
            pass

    def _maybe_interrupt_playback(self):
        """Sustained voice during playback/generation = barge-in. One
        interrupt per voice episode (re-armed when the state machine resets
        to idle after a commit)."""
        now = time.time()
        if self.interrupted_episode:
            return
        if now < self.session.playback_end or self.session.gen_in_progress:
            self.interrupted_episode = True
            logger.info(
                "[%s] frc: barge-in, stopping playback/generation",
                self.client_id[:8],
            )
            self.session.interrupt()
