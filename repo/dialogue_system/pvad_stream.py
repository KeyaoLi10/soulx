"""Pre-ASR stream gate (2026-11-05 背景视频调查的落地): an independent
channel that silences non-main-speaker audio BEFORE it reaches the turn
server, so background voices never enter ASR, never surface as live text
in the frontend, and never drive turn commits — instead of the old
post-hoc veto (pvad_turn_gate), which could only stop a committed turn
from reaching the LLM after ASR had already transcribed and displayed it.

Architecture (mirrors pvad_client's sidecar discipline — the receive loop
never waits on the pvad service):
  - The receive loop calls stream_gate_filter(session, chunk) per chunk:
    O(1) flag read; returns the chunk unchanged unless the gate is
    actively suppressing.
  - A per-session daemon scorer thread wakes every _SCORE_INTERVAL_S,
    scores the current ring-tail utterance (same speech-bounded window
    the turn gate uses, 1.2 s — ECAPA does not separate speakers on
    shorter windows) against the enrolled reference, and drives a
    HYSTERESIS state machine:
        suppress after >= _SUPPRESS_LOW_RUN consecutive scores < T_low
        release on any score >= T_high (threshold itself)
    The dead band [T_low, T_high) holds the current state: measured on
    the 2026-11-05 video session, user-under-video and video-only
    ECAPA bands OVERLAP in 0.11-0.40 (report docs/
    PVAD背景视频失效调查报告-20261105.md §5) — a single threshold would
    either leak video or eat the user. Sustained-below-T_low is the
    "clearly not the enrolled speaker" signature; one high score
    releases instantly so a loud user sentence recovers within one
    scoring round.
  - Suppression replaces the chunk with zeros: the turn server keeps
    its chunk cadence (its protocol is continuous) but sees silence —
    no nonidle, no live ASR text, no commits, no barge-in machinery
    churn from the background voice. The ORIGINAL audio still feeds the
    pvad ring, the pvad sidecar (/feed), the forensics sidecar and the
    stream-gate scorer itself (scoring must see the real mic).

Scope guards (the "no collateral damage when unused" invariant):
  - pvad_mode=off  => stream_gate_filter is never called (the app checks
    pvad_enabled first) and the scorer thread does zero HTTP work.
  - unenrolled     => never suppress (nothing to verify against).
  - service down / unscoreable window (silence, <1 s) => fail-open,
    exactly like the turn gate: a falsely muted user is this gate's
    worst failure.
  - shadow mode    => evaluates and reports (pvad_status.gated) but
    never suppresses.
  (The FRC engine that had its own pvad endpointing was removed 2026-11-09;
  SoulX is the only turn engine.)
"""
import logging
import threading
import time

import numpy as np

from settings import Config, DEFAULT_SETTINGS

logger = logging.getLogger(__name__)

_SCORE_INTERVAL_S = 0.4   # scorer cadence; also the worst-case release lag
_STREAM_WINDOW_S = 1.2    # ring-tail utterance scored per round (>=1 s is
                          # where ECAPA separates; 0.5 s does not at all)
_STREAM_MARGIN = 0.20     # T_low = threshold - this (0.50 -> 0.30)
_SUPPRESS_LOW_RUN = 2     # consecutive sub-T_low scores before suppressing
                          # (~0.8 s of sustained evidence; one noisy low
                          # score mid-user-speech must not mute them)


def stream_gate_active(session) -> bool:
    """Cheap enable check for the receive loop: pvad on AND the stream
    gate switch on. (Enrolled/mode checks live in the scorer — they flip
    at runtime.)"""
    return (
        session.settings.get("pvad_mode", DEFAULT_SETTINGS["pvad_mode"]) != "off"
        and session.settings.get(
            "pvad_stream_gate", DEFAULT_SETTINGS["pvad_stream_gate"]
        )
    )


class PvadStreamGate:
    """Per-session pre-ASR suppression state + scorer thread. Created lazily
    by app.py only while pvad is enabled; a live switch to pvad off just
    makes the scorer idle (no HTTP, no suppression)."""

    def __init__(self, session, start_thread: bool = True):
        self.session = session
        self._lock = threading.Lock()
        self._suppressed = False
        self._low_run = 0
        self._last_cos = None
        self._last_sep = None          # shadow sep_cos (speaker_sep switch)
        self._wake = threading.Event()
        self._alive = True
        self._thread = None
        if start_thread:
            self._thread = threading.Thread(
                target=self._scorer, name=f"pvadstream-{session.client_id[:8]}",
                daemon=True,
            )
            self._thread.start()

    # ---- receive-loop side (must never block) ----

    def suppressing(self) -> bool:
        with self._lock:
            return self._suppressed

    def stop(self) -> None:
        self._alive = False
        self._wake.set()

    # ---- scorer thread ----

    def _scorer(self) -> None:
        while self._alive:
            self._wake.wait(_SCORE_INTERVAL_S)
            self._wake.clear()
            if not self._alive:
                break
            self._evaluate()

    def _evaluate(self) -> None:
        """One scoring round (separate method so the regression replay can
        drive it offline with a stubbed pvad client). Reads the session's
        live settings each round: a mid-session switch to pvad off makes
        the gate inert without recreating it."""
        # Imported here: pvad_gate pulls emitter/settings, and app imports
        # this module — keep the import graph acyclic at module load.
        from pvad_gate import _pvad_ring_tail, pvad_enabled

        session = self.session
        try:
            mode = session.settings.get(
                "pvad_mode", DEFAULT_SETTINGS["pvad_mode"]
            )
            if not pvad_enabled(session) or not session.pvad.enrolled:
                # off or nothing to verify against: always let audio
                # through (fail-open, zero HTTP work when off)
                self._set_state(False, None, "inactive")
                return
            threshold = session.settings.get(
                "pvad_threshold", DEFAULT_SETTINGS["pvad_threshold"]
            )
            wav = _pvad_ring_tail(session, _STREAM_WINDOW_S)
            if wav is None or wav.shape[0] < int(1.0 * Config.SAMPLE_RATE):
                # silence / sub-second utterance: nothing to attribute,
                # and ECAPA cannot score it — release (fail-open)
                self._set_state(False, None, "unscoreable")
                return
            cos = session.pvad.score(wav)
            if cos is None:
                self._set_state(False, None, "service-blind")  # fail-open
                return
            # shadow-only isolation observation (speaker_sep switch,
            # default off): ECAPA-scores what the turn server actually
            # heard (the VoxLock-isolated ring tail) against the
            # enrolled reference, reported alongside the raw cos — never
            # changes the suppress/release decision (see voxlock_isolate.py)
            from voxlock_isolate import sep_shadow_round
            self._last_sep = sep_shadow_round(session, wav)
            t_low = threshold - _STREAM_MARGIN
            if cos >= threshold:
                self._low_run = 0
                self._set_state(False, cos, "release")
            elif cos < t_low:
                self._low_run += 1
                if self._low_run >= _SUPPRESS_LOW_RUN and mode == "enforce":
                    self._set_state(True, cos, "suppress")
                else:
                    # shadow mode (or first low score): evaluate, report
                    self._set_state(False, cos, "low-run")
            else:
                # dead band: hold the current state, but the run of
                # clearly-low scores is broken
                self._low_run = 0
                self._set_state(self._suppressed, cos, "dead-band")
        except Exception as e:  # the gate must never die silently
            logger.warning(
                f"[{session.client_id}] pvad stream-gate scorer error: "
                f"{type(e).__name__}: {e}"
            )
            self._set_state(False, None, "error")

    def _set_state(self, suppressed: bool, cos, why: str) -> None:
        with self._lock:
            changed = suppressed != self._suppressed
            self._suppressed = suppressed
            self._last_cos = cos
        if changed:
            sep = self._last_sep
            logger.info(
                f"[{self.session.client_id}] pvad stream gate "
                f"{'SUPPRESSING' if suppressed else 'releasing'} mic->ASR "
                f"(why={why}, cos={None if cos is None else round(cos, 3)}"
                + (f", sep_cos={sep['sep_cos']}" if sep else "") + ")"
            )

    def sep_status(self) -> dict | None:
        """Latest shadow separation observation, for pvad_status."""
        with self._lock:
            return self._last_sep


def stream_gate_filter(session, wav: np.ndarray) -> np.ndarray:
    """Receive-loop hook: return silence while the gate is suppressing,
    the original chunk otherwise. Call AFTER the ring / sidecar / pvad
    feeder have seen the original audio (they must keep scoring the real
    mic, or the gate could never release)."""
    gate = session.pvad_stream_gate
    if gate is None or not gate.suppressing():
        return wav
    return np.zeros_like(wav)
