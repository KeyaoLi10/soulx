"""VoxLock caller isolation (2026-11-10, replaces the MossFormer2
separation scheme — user decision: pvad+MossFormer 说话人分离效果不行).

What changed: the old scheme (MossFormer2_SS sidecar :8087 separating a
window into two streams + ECAPA picking the better one, speaker_sep.py,
retired) never influenced any decision — offline validation showed the
separated streams do NOT widen the ECAPA bands (docs/
说话人分离验证报告-20261106.md: intra-speaker variance, not mixture
contamination). VoxLock (arbazsiddiqui/voxlock, Apache-2.0) attacks the
problem from the other end: a 27M causal band-split RNN target-speaker
extractor that removes every voice but the caller's BEFORE ASR hears
the audio, streaming, 24 ms algorithmic latency, ONNX int8 in-process
(measured 0.60 RTF at 32 ms chunks on this box, one thread).

Pipeline position — the turn server is the ONLY consumer whose audio
changes:

    mic -> audio_mic recording / pvad ring / pvad sidecar feed /
           barge-in evidence                (all see the ORIGINAL mic)
        -> voxlock_asr_filter()             (this module, per session)
        -> session.vad.process()            (ASR / EOT / turn commits
                                             hear the isolated caller)

The pvad stream gate keeps scoring the original ring, so it stays
independently correct; the two layers compose (gate zeros pass through
the isolator as silence).

Lock acquisition (who is "the caller"):
  - pvad enrolled          -> seeded: VoxLock stream(sample) over the
                              banked VERIFIED turn audio (same natural
                              contiguous speech the ECAPA reference uses;
                              seeded once — a mid-call restart from an
                              empty LSTM state can latch onto background,
                              see LiveStream's own docstring).
  - pvad off / not enrolled-> live(): pass-through until 3 s of speech,
                              auto-locks, isolates from then on. The
                              lock push internally warms the stream over
                              the last ≤30 s (output dropped), an ~18 s
                              worker stall during which the filter fails
                              open — bounded, sample-conserving (see
                              filter()), and the reason the seeded path
                              is preferred whenever pvad is enrolled.

Scope guards (the "no collateral damage when unused" invariant):
  - `speaker_sep` setting (default False) — off means the filter is a
    one-line passthrough and this module never spawns a thread or
    imports voxlock (the worker imports it lazily on first chunk).
  - Every failure mode fails OPEN: the receive loop always has audio to
    send the turn server. A transiently slow worker sends the ORIGINAL
    chunk once and debits the duplicate from later drain — exact sample
    conservation, the ASR stream never stretches, repeats or replays.
  - sep_shadow_round (the observability surface the MossFormer shadow
    used to provide) ECAPA-scores what ASR actually heard (the isolated
    ring tail) against the enrolled reference: with a good isolator the
    user-speech windows score high and background windows collapse —
    the band separation MossFormer could not deliver. Reported as
    pvad_status.sep_cos; never changes any decision.
"""
import logging
import queue
import threading
import time

import numpy as np

from settings import DEFAULT_SETTINGS

logger = logging.getLogger(__name__)

_SR = 16000
_SEED_MIN_S = 3.0          # voxlock stream(sample) needs >= 3 s of caller
_PULL_TIMEOUT_S = 0.03     # filter's bounded wait for ready output
_STEADY_GRACE_S = 120.0    # no degrade-checks until this long after lock
_STEADY_BACKLOG_S = 2.0    # steady-state backlog that trips bypass
_RING_S = 8.0              # isolated-audio ring kept for shadow scoring


def voxlock_enabled(session) -> bool:
    return bool(session.settings.get(
        "speaker_sep", DEFAULT_SETTINGS["speaker_sep"]))


class VoxLockIsolator:
    """Per-session streaming isolator: one worker thread owns the voxlock
    stream (its LSTM state must see every chunk in order), the receive
    loop feeds it and drains the isolated output. Created lazily by the
    receive loop only while `speaker_sep` is on."""

    def __init__(self, session, sample=None):
        self.session = session
        self._in_q = queue.SimpleQueue()    # original chunks -> worker
        self._out_q = queue.SimpleQueue()   # isolated chunks -> filter
        self._lock = threading.Lock()
        self._alive = True
        self._degraded = False              # permanent fail-open bypass
        self._isolating = False             # worker's stream locked?
        self._locked_at = None              # lock transition time
        self._seeded = sample is not None   # seeded beats live()-lock
        self._reseed = None                 # sample the worker will apply
        self._drop = 0                      # isolated samples yet to drop
        self._ring = []                     # isolated audio (shadow)
        self._ring_n = 0
        self._initial_sample = sample
        self._worker = threading.Thread(
            target=self._run,
            name=f"voxlock-{session.client_id[:8]}",
            daemon=True,
        )
        self._worker.start()

    # ---- receive-loop side (never blocks > _PULL_TIMEOUT_S) ----

    def filter(self, wav: np.ndarray) -> np.ndarray:
        """Feed one mic chunk, return what the turn server should hear.
        Fails open (original audio) on any lag or failure. Sample
        conservation is exact: every fail-open send debits the same
        number of samples from later drain, so nothing is duplicated."""
        if not self._alive or self._degraded:
            return wav
        self._in_q.put(wav)                 # queue.SimpleQueue: never blocks
        parts = []
        try:
            parts.append(self._out_q.get(timeout=_PULL_TIMEOUT_S))
            while len(parts) < 32:          # burst catch-up after a lag
                parts.append(self._out_q.get_nowait())
        except queue.Empty:
            pass
        if not parts:
            # nothing ready (worker jitter / warm-up stall): send the
            # original chunk and debit its future isolated counterpart
            self._drop += int(wav.shape[0])
            return wav
        y = parts[0] if len(parts) == 1 else np.concatenate(parts)
        if self._drop:
            n = min(self._drop, y.shape[0])
            self._drop -= n
            y = y[n:]
            if y.shape[0] == 0:
                # the drain was entirely the debit for earlier
                # fail-open sends — this round fails open too
                self._drop += int(wav.shape[0])
                return wav
        y = np.clip(y, -1.0, 1.0).astype(np.float32, copy=False)
        self._ring_append(y)
        return y

    def _ring_append(self, y: np.ndarray) -> None:
        with self._lock:
            self._ring.append(y)
            self._ring_n += y.shape[0]
            max_n = int(_RING_S * _SR)
            while self._ring_n > max_n and self._ring:
                self._ring_n -= self._ring[0].shape[0]
                self._ring.pop(0)

    def isolated_tail(self, seconds: float):
        """Last `seconds` of the isolated stream (what ASR heard), for
        the shadow scorer. None when there is not enough yet."""
        with self._lock:
            need = int(seconds * _SR)
            if self._ring_n < need:
                return None
            parts, n = [], 0
            for chunk in reversed(self._ring):
                parts.append(chunk)
                n += chunk.shape[0]
                if n >= need:
                    break
            wav = np.concatenate(list(reversed(parts)))
            return wav[-need:]

    def seed(self, sample: np.ndarray) -> bool:
        """Upgrade a live()-locked (or not-yet-locked) isolator to a
        seeded stream over verified enrollment audio. Applied by the
        worker on its next chunk. One-way: a mid-call restart risks
        latching the LSTM onto background speech, so re-enrollments of
        the SAME speaker do not reseed."""
        if (not self._alive or self._seeded or self._degraded
                or sample is None
                or sample.shape[0] < int(_SEED_MIN_S * _SR)):
            return False
        with self._lock:
            self._seeded = True
            self._reseed = sample
        logger.info(
            f"[{self.session.client_id}] voxlock seeded from verified "
            f"enrollment audio ({sample.shape[0] / _SR:.1f}s)"
        )
        return True

    def stop(self) -> None:
        self._alive = False
        logger.info(
            f"[{self.session.client_id}] voxlock isolator stopped "
            f"(isolating={self._isolating}, degraded={self._degraded})"
        )

    # ---- worker thread ----

    def _run(self) -> None:
        try:
            import voxlock                  # lazy: off-switch sessions
        except Exception as exc:            # never import this at module top
            logger.warning(f"[voxlock] import failed: {exc}")
            self._degraded = True
            return
        try:
            vl = voxlock.VoxLock(threads=1)
            if self._initial_sample is not None:
                stream = vl.stream(self._initial_sample)
                logger.info(
                    f"[{self.session.client_id}] voxlock seeded at start "
                    f"({self._initial_sample.shape[0] / _SR:.1f}s sample)"
                )
            else:
                stream = vl.live()
        except Exception as exc:
            logger.warning(f"[voxlock] init failed: {exc}")
            self._degraded = True
            return
        was_isolating = False
        while self._alive:
            try:
                chunk = self._in_q.get(timeout=0.5)
            except queue.Empty:
                continue
            with self._lock:
                seed = self._reseed
                self._reseed = None
            if seed is not None:
                try:
                    stream = vl.stream(seed)
                except Exception as exc:
                    logger.warning(f"[voxlock] reseed failed: {exc}")
            try:
                out = stream.push(chunk)
            except Exception as exc:
                logger.warning(
                    f"[{self.session.client_id}] voxlock push failed: "
                    f"{exc} — bypassing isolation"
                )
                self._degraded = True
                return
            if out is not None and getattr(out, "shape", None) \
                    and out.shape[0]:
                self._out_q.put(out)
            isolating = bool(getattr(stream, "locked", True))
            if isolating != was_isolating:
                was_isolating = isolating
                with self._lock:
                    self._isolating = isolating
                    self._locked_at = time.time() if isolating else None
                logger.info(
                    f"[{self.session.client_id}] voxlock "
                    f"{'ISOLATING' if isolating else 'passthrough'} "
                    f"({'seeded' if self._seeded else 'auto-lock'})"
                )
            # steady-state health: a backlog that persists well past the
            # lock/warm transition means this box cannot keep up —
            # bypass beats an ever-growing lag. (During the auto-lock
            # warm-up the queue legitimately spikes ~18 s; the grace
            # window covers it and the filter's fail-open debit keeps
            # the stream consistent meanwhile.)
            if (isolating and self._locked_at is not None
                    and time.time() - self._locked_at > _STEADY_GRACE_S
                    and self._in_q.qsize() * chunk.shape[0]
                    > _STEADY_BACKLOG_S * _SR):
                self._degraded = True
                logger.warning(
                    f"[{self.session.client_id}] voxlock isolator degraded "
                    f"(steady backlog > {_STEADY_BACKLOG_S:.0f}s) — "
                    f"bypassing isolation for the rest of the session"
                )
                return


# ---- receive-loop hook (app.py) ----

def voxlock_asr_filter(session, wav: np.ndarray) -> np.ndarray:
    """The single integration point in the receive loop. Mirrors the
    stream-gate discipline: called AFTER every original-audio consumer
    (recording, pvad ring/sidecar, barge-in evidence), replaces only
    what session.vad.process() — the ASR feed — hears."""
    iso = getattr(session, "voxlock_isolator", None)
    if iso is None:
        return wav
    if not voxlock_enabled(session):
        # toggled off mid-session: stop feeding (the queue would back up)
        # and tear it down; the next loop iteration recreates if re-on
        iso.stop()
        session.voxlock_isolator = None
        return wav
    return iso.filter(wav)


def ensure_isolator(session):
    """Lazy creation from the receive loop. Seeds from the pvad bank when
    the speaker is already enrolled (verified audio only); otherwise the
    stream live()-locks on its own."""
    iso = getattr(session, "voxlock_isolator", None)
    if iso is not None:
        return iso
    sample = None
    if getattr(session, "pvad", None) is not None \
            and session.pvad.enrolled:
        try:
            from pvad_gate import _pvad_enroll_audio
            wav, total_s = _pvad_enroll_audio(session)
            if wav is not None and total_s >= _SEED_MIN_S:
                sample = wav
        except Exception as exc:      # noqa: BLE001 — seeding is best-effort
            logger.warning(f"[voxlock] bank seeding skipped: {exc}")
    iso = VoxLockIsolator(session, sample=sample)
    session.voxlock_isolator = iso
    return iso


# ---- shadow observability (replaces speaker_sep.sep_shadow_round) ----

def sep_shadow_round(session, wav) -> dict | None:
    """One shadow observation for the stream gate: ECAPA-score what the
    turn server actually heard (the isolated ring tail) against the
    enrolled reference. Returns {"sep_cos"} or None. Never raises,
    never changes any decision — same contract as the retired
    MossFormer shadow (the `wav` argument is the gate's original ring
    tail, kept for signature compatibility; alignment is approximate —
    the isolated ring lags the mic ring by the isolator's ~50-100 ms)."""
    iso = getattr(session, "voxlock_isolator", None)
    if iso is None or not voxlock_enabled(session):
        return None
    try:
        tail = iso.isolated_tail(1.2)     # same window as the gate scores
        if tail is None or tail.shape[0] < _SR:
            return None
        cos = session.pvad.score(tail)
        if cos is None:
            return None
        return {"sep_cos": round(cos, 3)}
    except Exception as exc:              # shadow must never disturb the gate
        logger.warning(f"[voxlock] shadow round error: {exc}")
        return None
