# -*- coding: utf-8 -*-
"""Offline tests for agent/bdl.py（balldontlie 接入，2026-11-09）。

无外网依赖：验签用本地 HMAC 计算，事件用 record_event 直接注入。
sports_games 的真实 API 调用留待实机验证（见 docs/修改日志.md）。

Run: /home/research/krisl/soulx/venv/bin/python -I scripts/agent_bdl_test.py
"""
import hashlib
import hmac
import json
import os
import sys
import tempfile
import time
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


# ---- 构造假事件 -----------------------------------------------------------

SECRET = "test-secret-0123456789"

# ingest_webhook 用工作区的正式密钥文件；测试把它指到临时文件
with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
    f.write(SECRET)
os.environ["BDL_WEBHOOK_SECRET_FILE"] = f.name


def _evt(etype, home, away, hs, vs, gid, eid):
    return {
        "id": eid, "type": etype, "sport": "nba", "game_id": gid,
        "payload": {"game": {
            "id": gid, "status": "Final",
            "home_team_score": hs, "visitor_team_score": vs,
            "home_team": {"id": 14, "name": "Heat", "full_name":
                          "Miami Heat", "abbreviation": "MIA"},
            "visitor_team": {"id": 23, "name": "Kings", "full_name":
                             "Sacramento Kings", "abbreviation": "SAC"},
        }},
    }


def _sign(raw: bytes, ts: str, secret=SECRET):
    mac = hmac.new(secret.encode(), f"{ts}.".encode() + raw,
                   hashlib.sha256).hexdigest()
    return f"v1={mac}"


BODY = json.dumps(_evt("nba.game.ended", "Heat", "Kings", 115, 121,
                       21718099, "evt-test-1")).encode()

# ---- 验签 -------------------------------------------------------------------

ts = str(int(time.time()))
check("sig ok", bdl.verify_signature(BODY, ts, _sign(BODY, ts), secret=SECRET))
check("sig bad secret",
      not bdl.verify_signature(BODY, ts, _sign(BODY, ts, "wrong"), secret=SECRET))
check("sig tampered body",
      not bdl.verify_signature(BODY + b"x", ts, _sign(BODY, ts), secret=SECRET))
check("sig no v1 prefix",
      not bdl.verify_signature(BODY, ts, _sign(BODY, ts)[3:], secret=SECRET))
check("sig empty", not bdl.verify_signature(b"", ts, "v1=abc", secret=SECRET))

# ---- ingest：验签入口（含新鲜度/JSON 校验） --------------------------------

st, msg = bdl.ingest_webhook(BODY, ts, _sign(BODY, ts))
check("ingest ok", st == 200 and msg == "ok", f"{st} {msg}")
st, msg = bdl.ingest_webhook(BODY, ts, _sign(BODY, ts))
check("ingest duplicate", st == 200 and msg == "duplicate", f"{st} {msg}")
st, msg = bdl.ingest_webhook(BODY, ts, "v1=" + "0" * 64)
check("ingest bad sig -> 401", st == 401, f"{st} {msg}")
st, msg = bdl.ingest_webhook(b"not json", ts, _sign(b"not json", ts))
check("ingest bad json -> 400", st == 400, f"{st} {msg}")
old_ts = str(int(time.time()) - 7200)
st, msg = bdl.ingest_webhook(BODY, old_ts, _sign(BODY, old_ts))
check("ingest stale -> 401", st == 401, f"{st} {msg}")

# ---- sports_events 工具读取 --------------------------------------------------

tr = T.execute_tool_v2("sports_events", {"count": 10})
check("events tool ok", tr.status == "ok" and "Heat" in tr.data
      and "Kings" in tr.data and "121" in tr.data,
      f"status={tr.status} data={tr.data[:80]}")
tr = T.execute_tool_v2("sports_events", {"count": 10, "sport": "nfl"})
check("events tool sport filter empty", tr.status == "empty", tr.status)

# ---- take_game_notes：一次性提示 + 中文匹配 + 去重 ---------------------------

announced = set()
note = bdl.take_game_notes(
    ["查询今天热火跟国王队的比赛情况"], announced)
check("note mentions game", "热火" in note or "Heat" in note, note[:100])
check("note zh score", "121" in note and "115" in note, note[:100])
note2 = bdl.take_game_notes(
    ["查询今天热火跟国王队的比赛情况"], announced)
check("note one-shot (no repeat)", note2 == "", note2[:100])
announced2 = set()
note3 = bdl.take_game_notes(["查询北京今天的天气"], announced2)
check("note no match -> empty", note3 == "", note3[:100])

# 无关事件的中文队名不误伤：fake 一场无关比赛后仍只匹配关注过的
ok_other = bdl.record_event("evt-test-2", "nba.game.started", "nba", 1,
                            {"game": {"home_team": {"name": "Lakers",
                                      "full_name": "Los Angeles Lakers",
                                      "abbreviation": "LAL"},
                                      "visitor_team": {"name": "Celtics",
                                       "full_name": "Boston Celtics",
                                       "abbreviation": "BOS"}}})
check("record started event", ok_other)
announced3 = set()
note4 = bdl.take_game_notes(["湖人凯尔特人比赛怎么样了"], announced3)
check("note not fired for started event", note4 == "", note4[:100])

# ---- sports_games 参数校验（不发网络请求的路径） -----------------------------

tr = T.execute_tool_v2("sports_games", {"sport": "nhl"})
check("games bad sport", tr.status == "error" and tr.error_code == "bad_sport",
      f"{tr.status} {tr.error_code}")
tr = T.execute_tool_v2("sports_games", {"date": "2026/11/09"})
check("games bad date", tr.status == "error" and tr.error_code == "bad_date",
      f"{tr.status} {tr.error_code}")

# ---- 工具注册 ----------------------------------------------------------------

names = [s["function"]["name"] for s in T.NEW_TOOL_SCHEMAS]
check("schemas registered", "sports_games" in names and "sports_events" in names,
      str(names))

print()
if FAILS:
    print(f"FAILED: {len(FAILS)} -> {FAILS}")
    sys.exit(1)
print("ALL OK")
