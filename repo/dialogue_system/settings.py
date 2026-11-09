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
import json
import os

from clients.llm_client import QwenLLM_stream


class Config:
    """Global configuration constants."""

    SAMPLE_RATE = 16000
    VAD_POOL_SIZE = 10
    PORT = 8090


# Per-session live settings (web UI "Settings" panel, applied via the
# update_config WS event; no restart needed). 2026-11-09: VoxCPM2 is the
# only TTS and the SoulX turn server the only turn engine — the tts_engine /
# turn_engine selectors were removed along with FireRedTTS and the FRC
# dual-EOT engine; the VoxCPM LoRA checkpoint (tts_lora) is selectable live.
DEFAULT_SETTINGS = {
    # identity-only（2026-11-09 persona 拆分）：UI / settings 只承载身份类
    # 设定（"你叫 Elva"），说话风格纪律在服务端 STYLE_PROMPT，由
    # QwenLLM_stream.compose_system_prompt() 组装——此值不再存全文 prompt
    "system_prompt": QwenLLM_stream.DEFAULT_IDENTITY,
    "max_history": 5,          # LLM memory turns
    "min_seg_len": 10,         # chars before a segment is sent to TTS
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
    # Speaker separation (replaced 2026-11-10 with VoxLock, in-process):
    # when on, the mic audio the TURN SERVER hears goes through a VoxLock
    # target-speaker extractor that removes every voice but the caller's
    # (pass-through until locked — auto-lock, or seeded from the pvad
    # enrollment when the speaker gate is enrolled). Recording, the pvad
    # ring/sidecar and barge-in evidence always see the original mic.
    # The stream gate additionally reports the isolated-stream cos
    # (pvad_status.sep_cos, shadow only — never changes any decision;
    # see voxlock_isolate.py).
    "speaker_sep": False,
    # Fast/slow brain agent (delegate_task → background slow brain on
    # :6008, results delivered at the next quiet gap)
    "agent_enabled": True,
    # Which agent runtime drives control turns (manual §19, 2026-11-06).
    # legacy = the current path (fast brain calls delegate_task itself +
    # auto_cancel/auto_delegate safety nets) — the default, byte-identical
    # to pre-V2 behavior. shadow = the V2 router (route_turn plan_json)
    # runs in the background and only logs its decisions (route_log /
    # web.log "[..][shadow]"); speech is untouched. enforce = V2 takes
    # over control turns (plan commit after the shared grace gate →
    # template receipts → expression round); the legacy AgentRuntime
    # stops receiving new work. Live-switchable from the web UI (a
    # session created under legacy lazily builds V2 on first need); the
    # process-wide default can also be pinned via the AGENT_RUNTIME_MODE
    # env var at startup. While legacy background tasks are still
    # running, enforce defers to the legacy path until they drain.
    "agent_runtime_mode": os.environ.get("AGENT_RUNTIME_MODE", "legacy"),
    # VoxCPM LoRA checkpoint 选择（2026-11-09 双 checkpoint）：服务端
    # (services/voxcpm_tts_server.py) 同时加载 step_0000500 / step_0000400
    # 两份 LoRA，请求按此值路由；UI 下拉可实时切换。500 是长期默认
    # （400 曾实测数数音色崩，见 memory soulx-voxcpm-lora-enum-instability）。
    "tts_lora": "step_0000500",
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
    # 慢脑结果播报的打断保护阈值（晚八引入、晚十一补 commit 路径漏洞
    # 后前端可调）：播报期间 barge-in 须 ≥ delivery_bargein_chunks 个
    # nonidle chunk 且首块距今 ≥ delivery_bargein_span_ms 才真正打断；
    # 话轮 commit 证据不足同样暂缓打断（turn held）。1/0 = 退回普通
    # 3-in-1.6s 规则的近似行为（任何单 chunk 即放行）。
    "delivery_bargein_chunks": 5,
    "delivery_bargein_span_ms": 1000,
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
    # 播报打断保护：chunk 数 1-20；跨度 0-5 s（0 = 不要求跨度）
    "delivery_bargein_chunks": (int, 1, 20),
    "delivery_bargein_span_ms": (int, 0, 5000),
}
PVAD_MODES = ("off", "shadow", "enforce")
TTS_LORAS = ("step_0000500", "step_0000400")
AGENT_RUNTIME_MODES = ("legacy", "shadow", "enforce")


def pin_env_runtime_mode(clean: dict) -> dict:
    """env 钉死的 AGENT_RUNTIME_MODE 不被任何 UI 回推改写（就地修改）。

    2026-11-09 实机事故：该不变量此前只在加载/落盘路径
    （load/save_ui_default_overrides）生效，漏了 _merge_session_settings
    的会话内合并路径——前端 onopen 的全量设置回推携带 Runtime select
    的 DOM 默认值 legacy，一次浏览器连接就把 enforce 会话（及部署
    意图）静默翻回 legacy，话轮全走旧路、V2 路由/任务面板一概不生效。
    env 未钉死时 UI 实时切换照常可用。
    """
    env_mode = os.environ.get("AGENT_RUNTIME_MODE")
    if (env_mode in AGENT_RUNTIME_MODES
            and clean.get("agent_runtime_mode") not in (None, env_mode)):
        clean.pop("agent_runtime_mode", None)
    return clean


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
    if data.get("pvad_mode") in PVAD_MODES:
        clean["pvad_mode"] = data["pvad_mode"]
    if data.get("tts_lora") in TTS_LORAS:
        clean["tts_lora"] = data["tts_lora"]
    if data.get("agent_runtime_mode") in AGENT_RUNTIME_MODES:
        clean["agent_runtime_mode"] = data["agent_runtime_mode"]
    if "agent_enabled" in data:
        clean["agent_enabled"] = bool(data["agent_enabled"])
    if "playback_guard" in data:
        clean["playback_guard"] = bool(data["playback_guard"])
    if "asr_correct" in data:
        clean["asr_correct"] = bool(data["asr_correct"])
    if "pvad_stream_gate" in data:
        clean["pvad_stream_gate"] = bool(data["pvad_stream_gate"])
    if "speaker_sep" in data:
        clean["speaker_sep"] = bool(data["speaker_sep"])
    return clean


# ==== 前端设置跨会话持久化（2026-11-07 晚十一） =========================
# 用户在前端设置过的值自动成为后续会话的默认值：update_config 落地时
# 把清洗后的键值合并进 checkpoints/ui_settings_default.json（gitignored
# 运行产物），会话创建时叠加在 DEFAULT_SETTINGS 之上。环境变量钉死的
# AGENT_RUNTIME_MODE 始终优先（部署拓扑约束不该被 UI 点击改写）。
UI_DEFAULTS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "checkpoints", "ui_settings_default.json")

# 进程内镜像：import 时读一次，之后由 save_ui_default_updates 维护，
# 会话创建读它而不是每次开文件（写少读多）。
UI_DEFAULT_OVERRIDES = {}


def load_ui_default_overrides() -> dict:
    """读取持久化的前端默认值（坏文件/缺文件 = 空覆盖，绝不拦启动）。"""
    try:
        with open(UI_DEFAULTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        data = {}
    clean = sanitize_updates(data)
    env_mode = os.environ.get("AGENT_RUNTIME_MODE")
    if env_mode in AGENT_RUNTIME_MODES:
        clean["agent_runtime_mode"] = env_mode
    return clean


def save_ui_default_updates(clean: dict) -> None:
    """把一次已清洗的设置更新合并进持久化默认值并落盘（原子替换）。"""
    if not clean:
        return
    merged = dict(UI_DEFAULT_OVERRIDES)
    merged.update(clean)
    env_mode = os.environ.get("AGENT_RUNTIME_MODE")
    if env_mode in AGENT_RUNTIME_MODES:
        merged["agent_runtime_mode"] = env_mode
    try:
        os.makedirs(os.path.dirname(UI_DEFAULTS_FILE), exist_ok=True)
        tmp = UI_DEFAULTS_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        os.replace(tmp, UI_DEFAULTS_FILE)
        UI_DEFAULT_OVERRIDES.clear()
        UI_DEFAULT_OVERRIDES.update(merged)
    except OSError:
        # 磁盘问题不拦设置生效——只是丢了跨会话持久化
        pass


UI_DEFAULT_OVERRIDES.update(load_ui_default_overrides())
