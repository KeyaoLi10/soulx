"""Configuration module (extracted from app.py 2026-10-30).

Three kinds of configuration live here:
  - Config          — process-wide constants (audio format, pool size, port)
  - DEFAULT_SETTINGS— per-session live settings defaults (web UI "Settings"
                      panel; mirrored by frontend DEFAULT_PERSONA /
                      currentSettings() in script.js — keep in sync)
  - sanitize_updates— pure validation/clamping of partial settings updates
                      arriving over the WebSocket (update_config event)

Applying a sanitized update to a session (and its side effects: turn-server
push, FRC engine reset) stays in app.py — this module holds no session or
service state.
"""
import os

from clients.llm_client import QwenLLM_stream


class Config:
    """Global configuration constants."""

    SAMPLE_RATE = 16000
    VAD_POOL_SIZE = 10
    PORT = 8090


# Command-line TTS engine choice (TTS_ENGINE env) only sets the DEFAULT_SETTINGS
# default below; the live engine is chosen per session from the web UI panel.
TTS_ENGINE = os.environ.get("TTS_ENGINE", "voxcpm")

# Per-session live settings (web UI "Settings" panel, applied via the
# update_config WS event; no restart needed).
DEFAULT_SETTINGS = {
    "system_prompt": QwenLLM_stream.SYSTEM_PROMPT,
    "max_history": 5,          # LLM memory turns
    "min_seg_len": 10,         # chars before a segment is sent to TTS
    "tts_engine": TTS_ENGINE,  # voxcpm | firered
    "tts_cfg": 2.0,            # VoxCPM2 expressiveness
    "tts_timesteps": 10,       # VoxCPM2 quality steps
    # (voice: fixed by the TTS server's anchor reference, not a setting —
    # the voice_desc text-design knob was removed 2026-10-29)
    # Main-speaker gate (pvad sidecar :8086). off = ignore (direct SoulX
    # turn decisions, no pvad work at all); shadow = evaluate and report
    # but never block; enforce = reject non-main-speaker turns.
    # DEFAULT off since 2026-10-30: measured on real browser-WebRTC audio
    # (forensics session 4600b3b4), the same speaker scores 0.15-0.57 vs
    # their own enrollment — the 0.5 threshold (calibrated on clean
    # eot-bench audio: same 0.59-0.80 / different 0.06-0.44) sits inside
    # the real-user band, so enforce chronically rejects the enrolled
    # user (the "30 秒后一直不说话" incident); and the anti-pollution
    # verified-only banking means a thin enrollment can never recover.
    "pvad_mode": "off",
    "pvad_threshold": 0.50,    # ECAPA cosine vs enrolled speaker must be
                               # above this (measured: same 0.59-0.80,
                               # different 0.06-0.44)
    # Pre-ASR stream gate (2026-11-05): in enforce mode, once enrolled,
    # audio that sustainably scores well below the enrolled speaker is
    # silenced BEFORE the turn server — background voices never enter
    # ASR / live text / turn commits. Hysteresis: suppress after 2
    # consecutive scores < threshold-0.20, release on any score >=
    # threshold; shadow mode reports only; unenrolled/service-down/
    # unscoreable always pass (fail-open). Only active with pvad_mode
    # != off, so the default-off pvad path is untouched.
    "pvad_stream_gate": True,
    # Fast/slow brain agent (delegate_task → background slow brain on
    # :6008, results delivered at the next quiet gap)
    "agent_enabled": True,
    # Voice frontend processing mode. "soulx" = SoulX-Duplug turn server
    # (:8000, the default this system was built around). "frc" =
    # FireRedChat architecture (frc_engine.py): pvad speech-activity
    # endpointing + dual EOT (ultraVAD audio + Qwen first-token 是/否) +
    # FireRedASR — same LLMs and TTS, different turn-taking. TTS/LLM
    # services are shared, the FRC-only services (ultraVAD :8082,
    # FireRedASR :8010) must be running for "frc".
    "turn_engine": "soulx",
    # Wait-path silence confirmation after an <|user_incomplete|>: how long
    # the user must stay quiet before the reply fires. Live-tunable; shown
    # as ms in the UI and pushed to the turn server as max_wait_num chunks
    # of 160 ms (800 ms = 5 chunks, the config.yaml default).
    "turn_wait_ms": 800,
    # Complete-trigger hold window: after a (zero-confirmation) "complete"
    # turn fires, LLM output is held invisible for this long; resumed
    # speech inside the window silently discards the generation. 0 disables
    # the window (pre-2026-10-30 behavior — not recommended, that was the
    # barge-in bug).
    "turn_grace_ms": 900,
    # Playback/in-flight barge-in guard (2026-11-02 幻影打断修复; window
    # widened to the generation phase 2026-11-03): while a reply is being
    # generated OR played, a barge-in needs >= 3 nonidle chunks in a 1.6 s
    # sliding window, and a turn that commits on sparser evidence is
    # dropped — speaker echo / AEC residue / breath blips arrive as 1-2-
    # chunk episodic blips and must not chop the TTS or hijack the reply
    # (the 01:01 incident: a blip turn's hallucinated "是。" replaced the
    # user's real sentence in the UI and entered history mid-generation).
    # Off restores the raw zero-hysteresis behavior (instant interrupt on
    # ANY nonidle chunk, phantom turns included). SoulX engine only.
    "playback_guard": True,
    # Two-pass ASR correction (2026-11-05 方案A): at turn commit the turn
    # server re-recognizes the WHOLE turn offline with its paraformer and
    # pushes a turn_final; the pipeline worker holds up to
    # asr_correct_wait_ms for it (live-text fallback on timeout/empty).
    # Live text (3.2s rolling window, chunks glued without pauses) is the
    # known ASR-error source; the whole-turn pass measured markedly
    # better on 37 real turns at 0.08-0.26s latency. SoulX engine only.
    "asr_correct": True,
    "asr_correct_wait_ms": 500,
}

# Validation for update_config: (key, type, lo, hi) — ints/floats clamped to
# [lo, hi]; strings truncated to hi chars. Anything else in the payload is
# silently dropped so the UI can send a superset safely.
SETTING_RULES = {
    "max_history": (int, 1, 20),
    "min_seg_len": (int, 4, 40),
    "tts_timesteps": (int, 4, 40),
    "tts_cfg": (float, 0.5, 5.0),
    "system_prompt": (str, 0, 4000),
    "pvad_threshold": (float, 0.3, 0.9),
    # turn timing: wait confirmation 1-20 chunks; grace window 0-3 s
    # (0 = disabled)
    "turn_wait_ms": (int, 160, 3200),
    "turn_grace_ms": (int, 0, 3000),
    "asr_correct_wait_ms": (int, 0, 2000),
}
TTS_ENGINES = ("voxcpm", "firered")
PVAD_MODES = ("off", "shadow", "enforce")
TURN_ENGINES = ("soulx", "frc")


def sanitize_updates(data) -> dict:
    """Validate + clamp a partial settings payload from the web UI.

    Returns a clean dict holding only recognized keys with valid values —
    applying it to session.settings (and the side effects that switching
    some settings triggers) is the caller's job. Non-dict input yields {}.
    """
    if not isinstance(data, dict):
        return {}
    clean = {}
    for key, (typ, lo, hi) in SETTING_RULES.items():
        if key not in data:
            continue
        value = data[key]
        if typ in (int, float):
            try:
                value = typ(value)
            except (TypeError, ValueError):
                continue
            clean[key] = max(lo, min(hi, value))
        else:  # str
            if isinstance(value, str):
                clean[key] = value[:hi]
    if data.get("tts_engine") in TTS_ENGINES:
        clean["tts_engine"] = data["tts_engine"]
    if data.get("pvad_mode") in PVAD_MODES:
        clean["pvad_mode"] = data["pvad_mode"]
    if data.get("turn_engine") in TURN_ENGINES:
        clean["turn_engine"] = data["turn_engine"]
    if "agent_enabled" in data:
        clean["agent_enabled"] = bool(data["agent_enabled"])
    if "playback_guard" in data:
        clean["playback_guard"] = bool(data["playback_guard"])
    if "asr_correct" in data:
        clean["asr_correct"] = bool(data["asr_correct"])
    if "pvad_stream_gate" in data:
        clean["pvad_stream_gate"] = bool(data["pvad_stream_gate"])
    return clean
