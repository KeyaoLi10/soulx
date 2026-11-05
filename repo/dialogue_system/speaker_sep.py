"""Speaker-separation client + shadow hook (task 2, 2026-11-06 —
背景有其他人说话下的主说话人音频分离, user requirement: modular,
independent switch, off = zero impact on anything else).

Talks to the stateless separation sidecar (services/tse_server.py,
MossFormer2_SS_16K, :8087, runs from /mnt/disk3/krisl/soulx/venv-tse).
The sidecar separates a wav into two speaker streams; WHICH stream is
the enrolled main speaker is decided here, with the pvad ECAPA /score
call (per-window — stream identity is NOT stable across time, measured
17 flips in a 160 s session).

Integration is SHADOW-ONLY by design. The 2026-11-06 offline validation
(docs/说话人分离验证报告-20261106.md) measured that scoring the
separated user-stream instead of raw audio does NOT improve the stream
gate (interrupt bursts 61%→64% suppression, user speech 0%→5% false
mutes, video-only unchanged): the ECAPA band overlap is intra-speaker
variance, not mixture contamination. So when `speaker_sep` is on, the
stream gate additionally separates its scoring window each round and
reports sep_cos alongside the raw cos (pvad_status.sep_cos, transition
logs) — an experimental observability surface for trying better TSE
models later. It never influences the suppress/release decision.

Scope guards:
  - `speaker_sep` setting (default False) — off means this module is
    never called and costs nothing (no HTTP, no threads).
  - Every failure (service down, timeout, unscoreable) returns None
    silently-after-first-log; the shadow must never disturb the gate.
"""

import base64
import logging
import os

import numpy as np
import requests

from settings import DEFAULT_SETTINGS

logger = logging.getLogger(__name__)

TSE_BASE = os.environ.get("TSE_BASE", "http://127.0.0.1:8087")
_TIMEOUT_S = 2.5          # separation call budget; the gate cadence is 0.4 s
_http = requests.Session()
_logged_fail = False      # one log line per process when the service is down


def sep_enabled(session) -> bool:
    return bool(session.settings.get(
        "speaker_sep", DEFAULT_SETTINGS["speaker_sep"]))


def separate(wav: np.ndarray, sample_rate: int = 16000):
    """POST the window to the sidecar, return (s1, s2) float32 arrays or
    None on any failure. Stateless — safe to call from any worker
    thread; callers must not assume stream identity across calls."""
    global _logged_fail
    try:
        r = _http.post(
            f"{TSE_BASE}/separate",
            json={
                "audio": base64.b64encode(
                    wav.astype("<f4").tobytes()).decode(),
                "sample_rate": sample_rate,
            },
            timeout=_TIMEOUT_S,
        )
        if not r.ok:
            raise requests.RequestException(f"HTTP {r.status_code}")
        out = r.json()
        s1 = np.frombuffer(base64.b64decode(out["s1"]), dtype=np.float32)
        s2 = np.frombuffer(base64.b64decode(out["s2"]), dtype=np.float32)
        return s1, s2
    except (requests.RequestException, ValueError, KeyError) as exc:
        if not _logged_fail:
            _logged_fail = True
            logger.warning(f"[speaker_sep] separation unavailable: {exc}")
        return None


def sep_shadow_round(session, wav: np.ndarray) -> dict | None:
    """One shadow observation for the stream gate: separate the scoring
    window, ECAPA-score both streams against the enrolled reference,
    report the better one. Returns {"sep_cos", "sep_stream"} or None.
    Never raises, never blocks long (separation ≤ _TIMEOUT_S, two pvad
    /score calls ~150 ms each — runs on the gate's scorer thread)."""
    if not sep_enabled(session) or wav is None:
        return None
    try:
        parts = separate(wav)
        if parts is None:
            return None
        best = None
        for i, stream in enumerate(parts, start=1):
            if stream.shape[0] < 16000:
                continue
            cos = session.pvad.score(stream)
            if cos is None:
                continue
            if best is None or cos > best[0]:
                best = (cos, i)
        if best is None:
            return None
        return {"sep_cos": round(best[0], 3), "sep_stream": best[1]}
    except Exception as exc:      # shadow must never disturb the gate
        logger.warning(f"[speaker_sep] shadow round error: {exc}")
        return None
