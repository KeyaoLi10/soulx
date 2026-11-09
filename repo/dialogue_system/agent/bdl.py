# -*- coding: utf-8 -*-
"""balldontlie 体育数据接入（2026-11-09：API 拉取 + webhook 推送）。

两个入口：

- ``sports_games`` / ``sports_events`` 工具（慢脑 v2，经 agent/tools.py
  注册）：按日期/球队查 NBA/NFL/MLB 的赛程与比分。动机：当天比分在
  web_search 里经常查不到（2026-11-09 热火国王任务搜了 203s 无果），
  官方数据 API 一次调用即得。
- webhook 接收（app.py ``/webhook/balldontlie``，兼容 ``/webhook``）：
  balldontlie 推送 ``nba.game.started`` / ``nba.game.ended`` 等事件，
  HMAC-SHA256 验签后存进进程内事件环（去重、限界）。闲聊路经
  :func:`take_game_notes` 把「用户关注过的比赛刚结束」作为一次性提示
  注入下一轮闲聊上下文；慢脑经 ``sports_events`` 工具读全部近期事件。

密钥（工作区约定：明文文件，绝不入库、绝不在日志里回显）：
- API key：``balldontlie_apikey.txt``（env ``BDL_API_KEY`` /
  ``BDL_KEY_FILE`` 覆盖）
- webhook 签名密钥：``balldontile_webhook.txt``（文件名沿用注册时的
  实际拼写；env ``BDL_WEBHOOK_SECRET_FILE`` 覆盖）

验签规格（balldontlie webhooks 文档）：消息 = ``{timestamp}.{raw_body}``，
HMAC-SHA256，签名头 ``X-BDL-Webhook-Signature: v1=<hex>``，常量时间比较。
"""
import datetime
import hashlib
import hmac
import json
import logging
import os
import re
import threading
import time
from collections import deque

import requests

logger = logging.getLogger(__name__)

# 出站走本地 mihomo 代理（与 agent/tools.py 同一约定）
PROXY = os.environ.get("AGENT_PROXY", "http://127.0.0.1:17892")
PROXIES = {"http": PROXY, "https": PROXY} if PROXY else None

BDL_BASE = "https://api.balldontlie.io"
BDL_TIMEOUT_S = 12
BDL_KEY_FILE = "/home/research/krisl/soulx/balldontlie_apikey.txt"
BDL_WEBHOOK_SECRET_FILE = (
    "/home/research/krisl/soulx/balldontile_webhook.txt")

# 实测（2026-11-09，本 key 的订阅档）：nba/nfl/mlb 可用，nhl 需要付费档
SUPPORTED_SPORTS = ("nba", "nfl", "mlb")

# webhook 验签的时间窗（秒）：拒绝过期重放
WEBHOOK_FRESHNESS_S = 600
# 事件环与去重环的限界
_EVENT_RING = 500
_SEEN_RING = 2000
# take_game_notes 只看最近这么多秒内收到的 game.ended
NOTE_MAX_AGE_S = 1800

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


def _webhook_secret():
    path = os.environ.get(
        "BDL_WEBHOOK_SECRET_FILE", "").strip() or BDL_WEBHOOK_SECRET_FILE
    try:
        with open(path, "r", encoding="utf-8") as f:
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


# ---------------------------------------------------------------------------
# webhook：验签 + 事件环 + 读取
# ---------------------------------------------------------------------------

_EVENTS = deque(maxlen=_EVENT_RING)
_SEEN = deque(maxlen=_SEEN_RING)
_LOCK = threading.Lock()


def verify_signature(raw_body, timestamp, signature, secret=None):
    """HMAC-SHA256(f"{timestamp}.{raw_body}") 与 ``v1=<hex>`` 常量时间比对。"""
    if not raw_body or not timestamp or not signature:
        return False
    sig = str(signature)
    if not sig.startswith("v1="):
        return False
    secret = secret if secret is not None else _webhook_secret()
    if not secret:
        return False
    expected = hmac.new(
        secret.encode("utf-8"),
        f"{timestamp}.".encode("utf-8") + raw_body,
        hashlib.sha256).hexdigest()
    return hmac.compare_digest(sig[3:].strip().lower(), expected.lower())


def record_event(event_id, etype, sport, game_id, payload):
    """入环（去重）。返回 True=新事件，False=重复。线程安全。"""
    with _LOCK:
        if event_id and event_id in _SEEN:
            return False
        if event_id:
            _SEEN.append(event_id)
        _EVENTS.append({
            "id": event_id, "type": etype, "sport": sport,
            "game_id": game_id, "payload": payload or {},
            "received_ts": time.time(),
        })
        return True


def ingest_webhook(raw, timestamp, signature, event_id_header=""):
    """app.py 的接收端调用：验签 → 新鲜度 → 解析 → 入环。

    返回 (http_status, message)。重复事件返回 200（webhook 重试机制
    靠 2xx 停止重试）。
    """
    if not _webhook_secret():
        return 503, "webhook secret not configured"
    if not verify_signature(raw, timestamp, signature):
        return 401, "bad signature"
    try:
        ts = float(timestamp)
    except (TypeError, ValueError):
        return 401, "bad timestamp"
    if abs(time.time() - ts) > WEBHOOK_FRESHNESS_S:
        return 401, "stale timestamp"
    try:
        evt = json.loads(raw)
    except (ValueError, TypeError):
        return 400, "invalid json"
    if not isinstance(evt, dict):
        return 400, "invalid event"
    etype = str(evt.get("type") or "")
    if not etype:
        return 400, "missing event type"
    event_id = str(event_id_header or evt.get("id") or "")
    fresh = record_event(
        event_id, etype, str(evt.get("sport") or ""),
        evt.get("game_id"), evt.get("payload"))
    logger.info("[bdl-webhook] %s %s game_id=%s%s",
                etype, evt.get("sport"), evt.get("game_id"),
                "" if fresh else " (duplicate)")
    return 200, "ok" if fresh else "duplicate"


def _event_line(ev):
    game = (ev.get("payload") or {}).get("game") or {}
    home, away = game.get("home_team") or {}, game.get("visitor_team") or {}
    when = datetime.datetime.fromtimestamp(
        ev["received_ts"]).strftime("%m-%d %H:%M")
    head = f"- [{ev['type']}] 收到于 {when}"
    if home or away:
        head += (f" | {_fmt_team(away)} {game.get('visitor_team_score', '?')}"
                 f" : {game.get('home_team_score', '?')} {_fmt_team(home)}"
                 f" | {game.get('status', '')}")
    if ev.get("game_id"):
        head += f" | game_id={ev['game_id']}"
    return head


def recent_events(count=20, sport=""):
    """格式化近期事件（新→旧）。"""
    with _LOCK:
        evs = list(_EVENTS)
    if sport:
        evs = [e for e in evs if not e["sport"] or e["sport"] == sport]
    return [_event_line(e) for e in reversed(evs[-int(count):])]


def sports_events(count=20, sport="", call_id=""):
    """慢脑工具：读 webhook 推送的近期比赛事件（本地即时，无网络调用）。"""
    t0 = time.monotonic()
    lines = recent_events(count, (sport or "").strip().lower())
    if not lines:
        return _toolresult(
            call_id=call_id, status="empty",
            data="（webhook 尚未收到任何推送事件）",
            error_code="no_events",
            elapsed_ms=int((time.monotonic() - t0) * 1000))
    return _toolresult(
        call_id=call_id, status="ok",
        data=_clip("\n".join(lines)),
        elapsed_ms=int((time.monotonic() - t0) * 1000))


# ---------------------------------------------------------------------------
# take_game_notes：闲聊路一次性提示（防复读：每事件每会话只注入一次）
# ---------------------------------------------------------------------------

def _event_team_aliases(team):
    """事件里一支球队的全部别名（小写）+ 中文常用名。"""
    if not team:
        return []
    aliases = [str(team.get(k) or "").strip().lower() for k in
               ("full_name", "name", "abbreviation", "city", "location")]
    zh = _team_zh(team)
    if zh:
        aliases.append(zh)
    return [a for a in aliases if a]


def _team_in_texts(aliases, blob_lower):
    # aliases 已小写；中文别名无大小写问题
    return any(a in blob_lower for a in aliases if a)


def take_game_notes(match_texts, announced_ids, max_age_s=NOTE_MAX_AGE_S):
    """把「用户关注过 & 刚结束」的比赛整理成一次性闲聊提示。

    - match_texts：任务面板各任务的 goal/label 文本（中文亦可，经
      TEAM_ZH 匹配）；announced_ids：调用方的会话级已告知集合（本函数
      直接往里加，调用方闭包持有即可）。
    - 只处理 max_age_s 内收到的 ``*.game.ended``；每个事件只输出一次
      （复读教训：2026-11-09「结果马上就到」死循环——一次性注入 +
      提示词里明确「说完不再提」）。
    返回 "" 表示没有需要告知的。
    """
    blob = " ".join(str(t) for t in (match_texts or []) if t).lower()
    if not blob:
        return ""
    now = time.time()
    notes = []
    with _LOCK:
        evs = list(_EVENTS)
    for ev in reversed(evs):  # 新→旧
        if not ev["type"].endswith(".game.ended"):
            continue
        if now - ev["received_ts"] > max_age_s:
            continue
        if ev["id"] and ev["id"] in announced_ids:
            continue
        game = (ev.get("payload") or {}).get("game") or {}
        home_al = _event_team_aliases(game.get("home_team"))
        away_al = _event_team_aliases(game.get("visitor_team"))
        if not (_team_in_texts(home_al, blob)
                or _team_in_texts(away_al, blob)):
            continue
        if ev["id"]:
            announced_ids.add(ev["id"])
        notes.append(_event_line(ev))
    if not notes:
        return ""
    return (
        "【体育赛事实时推送（balldontlie webhook 刚收到）】\n"
        "以下比赛刚刚结束，而且用户之前关注过。在合适的时机（不打断用户"
        "当前话题）自然地告知一次比分即可；说过之后就不再提，用户没问就"
        "不要反复说：\n" + "\n".join(notes))
