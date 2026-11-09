# -*- coding: utf-8 -*-
"""balldontlie 体育数据接入（2026-11-09）。

``sports_games`` 工具（慢脑 v2，经 agent/tools.py 注册）：按日期/球队查
NBA/NFL/MLB 的赛程与比分。动机：当天比分在 web_search 里经常查不到
（2026-11-09 热火国王任务搜了 203s 无果），官方数据 API 一次调用即得。

webhook 推送（原 ``sports_events`` / ``take_game_notes`` / 验签）已于
同日按用户要求移除，本模块只做 API 拉取。

密钥（工作区约定：明文文件，绝不入库、绝不在日志里回显）：
API key：``balldontlie_apikey.txt``（env ``BDL_API_KEY`` /
``BDL_KEY_FILE`` 覆盖）。
"""
import datetime
import logging
import os
import re
import time

import requests

logger = logging.getLogger(__name__)

# 出站走本地 mihomo 代理（与 agent/tools.py 同一约定）
PROXY = os.environ.get("AGENT_PROXY", "http://127.0.0.1:17892")
PROXIES = {"http": PROXY, "https": PROXY} if PROXY else None

BDL_BASE = "https://api.balldontlie.io"
BDL_TIMEOUT_S = 12
BDL_KEY_FILE = "/home/research/krisl/soulx/balldontlie_apikey.txt"

# 实测（2026-11-09，本 key 的订阅档）：nba/nfl/mlb 可用，nhl 与足球联赛
# （laliga/epl/seriea/bundesliga/ucl 等 /v1/matches 接口）均需付费档
SUPPORTED_SPORTS = ("nba", "nfl", "mlb")

# NBA 球队中文常用名（中文任务文本 ↔ 英文队名匹配用；NFL/MLB 任务
# 标签多为英文或音译不一，先只做英文匹配，后续按需扩充）
TEAM_ZH = {
    "celtics": "凯尔特人", "knicks": "尼克斯", "76ers": "76人",
    "nets": "篮网", "raptors": "猛龙", "bulls": "公牛",
    "cavaliers": "骑士", "cavs": "骑士", "pacers": "步行者",
    "pistons": "活塞", "bucks": "雄鹿", "hawks": "老鹰",
    "hornets": "黄蜂", "heat": "热火", "magic": "魔术",
    "wizards": "奇才", "mavericks": "独行侠", "rockets": "火箭",
    "grizzlies": "灰熊", "pelicans": "鹈鹕", "spurs": "马刺",
    "nuggets": "掘金", "timberwolves": "森林狼", "jazz": "爵士",
    "trail blazers": "开拓者", "blazers": "开拓者", "warriors": "勇士",
    "suns": "太阳", "clippers": "快船", "lakers": "湖人",
    "kings": "国王", "thunder": "雷霆",
}


def _api_key():
    key = os.environ.get("BDL_API_KEY", "").strip()
    if key:
        return key
    try:
        with open(BDL_KEY_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""


def _clip(text, limit=4000):
    text = text or ""
    if len(text) > limit:
        return text[:limit] + f"\n...[truncated, {len(text)} chars total]"
    return text


def _toolresult(**kw):
    """延迟导入避免与 agent.tools 的循环依赖（tools 顶部 import 本模块）。"""
    from agent.tools import ToolResult
    return ToolResult(**kw)


# ---------------------------------------------------------------------------
# sports_games：赛程/比分查询
# ---------------------------------------------------------------------------

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SESSION = requests.Session()
if PROXIES:
    _SESSION.proxies.update(PROXIES)


def _bdl_get(path, params):
    """GET {BDL_BASE}{path}。返回 (data, error_str)；error_str 空即成功。"""
    key = _api_key()
    if not key:
        return None, "no api key (balldontlie_apikey.txt missing?)"
    try:
        r = _SESSION.get(
            BDL_BASE + path, params=params,
            headers={"Authorization": key}, timeout=BDL_TIMEOUT_S)
    except requests.exceptions.Timeout:
        return None, "timeout"
    except Exception as e:  # noqa: BLE001
        return None, f"network: {e}"
    if r.status_code == 401:
        return None, "unauthorized (check balldontlie api key)"
    if r.status_code == 403:
        try:
            return None, f"forbidden: {r.json().get('error', '')}".strip()
        except ValueError:
            return None, "http 403"
    if r.status_code != 200:
        return None, f"http {r.status_code}"
    try:
        return r.json(), ""
    except ValueError:
        return None, "invalid json response"


def _team_zh(team):
    for k in ("name", "full_name"):
        v = (team.get(k) or "").strip().lower()
        if v in TEAM_ZH:
            return TEAM_ZH[v]
    return ""


def _fmt_team(team):
    if not team:
        return "?"
    ab = team.get("abbreviation") or ""
    full = team.get("full_name") or " ".join(
        x for x in (team.get("location") or team.get("city"),
                    team.get("name")) if x)
    zh = _team_zh(team)
    return f"{full} ({ab}{';' + zh if zh else ''})" if ab else full


def _fmt_game(g):
    dt = (g.get("datetime") or "").replace("T", " ").replace("Z", "UTC")
    status = g.get("status") or g.get("status_state") or "?"
    return (f"- {g.get('date', '?')} {dt} | {status} | "
            f"{_fmt_team(g.get('visitor_team'))} "
            f"{g.get('visitor_team_score', 0)} : "
            f"{g.get('home_team_score', 0)} "
            f"{_fmt_team(g.get('home_team'))} | game_id={g.get('id', '?')}")


def _team_matches(team, q):
    """球队过滤：q（中文或英文，模型传什么查什么）与任一别名互为子串。"""
    q = (q or "").strip().lower()
    if not q:
        return True
    fields = [str(team.get(k) or "") for k in
              ("full_name", "name", "abbreviation", "city", "location")]
    zh = _team_zh(team)
    if zh:
        fields.append(zh)
    for f in fields:
        f = f.strip().lower()
        if f and (q in f or f in q):
            return True
    return False


def sports_games(sport="nba", date="", team="", call_id=""):
    """按日期（UTC）/球队查赛程与比分。date 为空时查「今天+昨天」UTC。

    返回 ToolResult（agent.tools）；失败不抛异常，慢脑据此回退 web_search。
    """
    t0 = time.monotonic()

    def _res(status, data="", error_code="", sources=None, truncated=False):
        return _toolresult(
            call_id=call_id, status=status, data=data, error_code=error_code,
            sources=sources or [], elapsed_ms=int((time.monotonic() - t0) * 1000),
            truncated=truncated)

    sport = (sport or "nba").strip().lower()
    if sport not in SUPPORTED_SPORTS:
        return _res("error", f"unsupported sport: {sport} "
                    f"(supported: {', '.join(SUPPORTED_SPORTS)})",
                    "bad_sport")
    dates = []
    if date:
        d = str(date).strip()
        if not _DATE_RE.match(d):
            return _res("error", f"invalid date '{d}' (want YYYY-MM-DD)",
                        "bad_date")
        dates = [d]
    else:
        today = datetime.datetime.now(datetime.timezone.utc).date()
        dates = [today.isoformat(), (today - datetime.timedelta(days=1)
                                     ).isoformat()]

    games, src_url = [], ""
    for d in dates:
        data, err = _bdl_get(
            f"/{sport}/v1/games",
            {"dates[]": d, "per_page": 100})
        if err:
            return _res("error", f"balldontlie: {err}", "api_error")
        src_url = (BDL_BASE + f"/{sport}/v1/games?dates[]={d}")
        games.extend(data.get("data") or [])

    if team:
        games = [g for g in games
                 if _team_matches(g.get("home_team") or {}, team)
                 or _team_matches(g.get("visitor_team") or {}, team)]

    if not games:
        return _res("empty", f"（{', '.join(dates)} 没有查到"
                    f"{' ' + team if team else ''}的比赛）", "no_games")

    text = (f"balldontlie 官方数据（sport={sport}, 日期为 UTC）：\n"
            + "\n".join(_fmt_game(g) for g in games))
    truncated = len(text) > 4000
    return _res(
        "ok", _clip(text), sources=[{
            "url": src_url,
            "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%S")}],
        truncated=truncated)
