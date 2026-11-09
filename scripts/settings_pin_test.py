# -*- coding: utf-8 -*-
"""offline 回归:AGENT_RUNTIME_MODE env 钉死不变量(2026-11-09 实机事故)。

事故:前端 onopen 的全量设置回推携带 Runtime select 的 DOM 默认值
legacy,而 env 钉死只在 load/save_ui_default_overrides 落盘/加载路径
生效,_merge_session_settings 的会话内合并漏设防——一次浏览器连接把
enforce 会话(及部署意图)静默翻回 legacy,话轮全走旧路。

用法:venv/bin/python -I scripts/settings_pin_test.py
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                       / "repo" / "dialogue_system"))
import settings  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("[PASS] " if cond else "[FAIL] ") + name +
          (f" — {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(name)


def merge_like_app(data, env_mode):
    """复刻 _merge_session_settings 的清洗+钉死序列(env 可空=未钉死)。"""
    old = os.environ.pop("AGENT_RUNTIME_MODE", None)
    if env_mode:
        os.environ["AGENT_RUNTIME_MODE"] = env_mode
    try:
        clean = settings.sanitize_updates(data)
        settings.pin_env_runtime_mode(clean)
        return clean
    finally:
        os.environ.pop("AGENT_RUNTIME_MODE", None)
        if old is not None:
            os.environ["AGENT_RUNTIME_MODE"] = old


# ---- 1. env 钉死:全量回推(事故形态)不得翻回 legacy --------------------
push = {"agent_enabled": True, "agent_runtime_mode": "legacy",
        "turn_engine": "soulx", "turn_wait_ms": 800}
clean = merge_like_app(push, "enforce")
check("env=enforce 时回推 legacy 被摘除",
      "agent_runtime_mode" not in clean, str(clean))
check("同推送里其余键照常生效",
      clean.get("agent_enabled") is True and clean.get("turn_wait_ms") == 800)

# ---- 2. env 钉死:与钉死值一致的回推原样通过(幂等) --------------------
clean = merge_like_app({"agent_runtime_mode": "enforce"}, "enforce")
check("env=enforce 时回推 enforce 原样通过",
      clean.get("agent_runtime_mode") == "enforce")

# ---- 3. env 未钉死:UI 实时切换照常可用(不误伤既有功能) ----------------
clean = merge_like_app({"agent_runtime_mode": "legacy"}, None)
check("env 未钉死时 UI 切到 legacy 照常生效",
      clean.get("agent_runtime_mode") == "legacy")
clean = merge_like_app({"agent_runtime_mode": "shadow"}, None)
check("env 未钉死时 UI 切到 shadow 照常生效",
      clean.get("agent_runtime_mode") == "shadow")

# ---- 4. 落盘路径:env 钉死时 persist 回推不把文件写成 legacy ------------
with tempfile.TemporaryDirectory() as td:
    old_file = settings.UI_DEFAULTS_FILE
    old_over = dict(settings.UI_DEFAULT_OVERRIDES)
    old_env = os.environ.pop("AGENT_RUNTIME_MODE", None)
    os.environ["AGENT_RUNTIME_MODE"] = "enforce"
    settings.UI_DEFAULT_OVERRIDES.clear()
    settings.UI_DEFAULT_OVERRIDES["agent_runtime_mode"] = "enforce"
    settings.UI_DEFAULTS_FILE = os.path.join(td, "ui.json")
    try:
        settings.save_ui_default_updates(
            settings.sanitize_updates({"agent_runtime_mode": "legacy"}))
        import json
        with open(settings.UI_DEFAULTS_FILE, encoding="utf-8") as f:
            on_disk = json.load(f)
        check("env=enforce 时 persist 回推 legacy 落盘仍是 enforce",
              on_disk.get("agent_runtime_mode") == "enforce", str(on_disk))
    finally:
        settings.UI_DEFAULTS_FILE = old_file
        settings.UI_DEFAULT_OVERRIDES.clear()
        settings.UI_DEFAULT_OVERRIDES.update(old_over)
        os.environ.pop("AGENT_RUNTIME_MODE", None)
        if old_env is not None:
            os.environ["AGENT_RUNTIME_MODE"] = old_env

# ---- 5. 非法值仍被 sanitize 拦下(不因钉死逻辑放行垃圾) ----------------
clean = merge_like_app({"agent_runtime_mode": "yolo"}, "enforce")
check("非法 runtime 值照旧被 sanitize 拦下",
      "agent_runtime_mode" not in clean)

print()
if failures:
    print(f"FAILED: {len(failures)} — {failures}")
    sys.exit(1)
print("all pass")
