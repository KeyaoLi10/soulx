# -*- coding: utf-8 -*-
"""Offline tests for agent/bdl.py（balldontlie 接入，2026-11-09）。

无外网依赖：只测参数校验与工具注册（不发网络请求的路径）。
sports_games 的真实 API 调用留待实机验证（见 docs/修改日志.md）。
（webhook 验签/事件环/一次性提示已于同日随 webhook 移除一并删除。）

Run: /home/research/krisl/soulx/venv/bin/python -I scripts/agent_bdl_test.py
"""
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "repo" / "dialogue_system"))

from agent import bdl  # noqa: E402
from agent import tools as T  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok  {name}")
    else:
        FAILS.append(name)
        print(f"FAIL  {name}  {detail}")


# ---- sports_games 参数校验（不发网络请求的路径） -----------------------------

tr = T.execute_tool_v2("sports_games", {"sport": "nhl"})
check("games bad sport", tr.status == "error" and tr.error_code == "bad_sport",
      f"{tr.status} {tr.error_code}")
tr = T.execute_tool_v2("sports_games", {"sport": "laliga"})
check("games soccer also bad sport (paid tier)",
      tr.status == "error" and tr.error_code == "bad_sport",
      f"{tr.status} {tr.error_code}")
tr = T.execute_tool_v2("sports_games", {"date": "2026/11/09"})
check("games bad date", tr.status == "error" and tr.error_code == "bad_date",
      f"{tr.status} {tr.error_code}")

# ---- 工具注册 ----------------------------------------------------------------

names = [s["function"]["name"] for s in T.NEW_TOOL_SCHEMAS]
check("sports_games registered", "sports_games" in names, str(names))
check("sports_events removed", "sports_events" not in names, str(names))

# ---- webhook 痕迹清理 --------------------------------------------------------

check("no webhook attr left in bdl",
      not any(hasattr(bdl, a) for a in (
          "verify_signature", "ingest_webhook", "record_event",
          "sports_events", "take_game_notes", "recent_events")))

print()
if FAILS:
    print(f"FAILED: {len(FAILS)} -> {FAILS}")
    sys.exit(1)
print("ALL OK")
