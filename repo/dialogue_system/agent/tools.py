"""Tool layer for the slow brain (background agent).

Each tool = JSON-schema (sent to the LLM via /agent_chat `tools`) + a plain
Python executor returning a string observation. Executors are deliberately
stdlib-only and defensive: the LLM chooses arguments, so every input is
bounded (timeouts, truncation, isolated workdir for code execution).

Research-demo sandboxing note: python_repl runs model-generated code in a
subprocess with a timeout and a throwaway working directory, but WITHOUT
privilege isolation — do not expose this system to untrusted users.
"""
import html.parser
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
import datetime
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

# all outbound web traffic goes through the local mihomo proxy (same one the
# workspace uses for HF); empty string disables proxying
PROXY = os.environ.get("AGENT_PROXY", "http://127.0.0.1:17892")
PROXIES = {"http": PROXY, "https": PROXY} if PROXY else None

SEARCH_TIMEOUT_S = 15
# fetch_page: a slow/JS-heavy page that returns garbage still burns the full
# timeout and typically triggers retry rounds — 10s is plenty for pages worth
# reading (demo-latency tuning 2026-10-30)
FETCH_TIMEOUT_S = 10
REPL_TIMEOUT_S = 20
MAX_TOOL_OUTPUT_CHARS = 4000

# Primary search: Tavily API (agent-oriented, no bot-gating, ~1-2s). The old
# DDG-html + baidu stack stays as _fallback_search for when Tavily is
# unavailable (no key / quota / network). Key resolution order: TAVILY_API_KEY
# env, then the plain key file (workspace convention — never commit it).
TAVILY_KEY_FILE = os.environ.get(
    "TAVILY_KEY_FILE", "/home/research/krisl/soulx/apikey.txt"
)
TAVILY_TIMEOUT_S = 10
# keep-alive session: saves the per-call TLS handshake through the proxy
# (measured ~1s of the per-call latency). include_answer is deliberately
# OFF — it adds ~3s per query and answers in English even for Chinese
# queries; the result snippets are enough for the slow brain.
_TAVILY_SESSION = requests.Session()
if PROXIES:
    _TAVILY_SESSION.proxies.update(PROXIES)


def _tavily_key():
    key = os.environ.get("TAVILY_API_KEY", "").strip()
    if key:
        return key
    try:
        return Path(TAVILY_KEY_FILE).read_text().strip()
    except OSError:
        return ""


def _clip(text, limit=MAX_TOOL_OUTPUT_CHARS):
    text = text or ""
    if len(text) > limit:
        return text[:limit] + f"\n...[truncated, {len(text)} chars total]"
    return text


# ---------------------------------------------------------------------------
# web_search — DuckDuckGo html endpoint (POST; GET gets bot-challenged)
# ---------------------------------------------------------------------------

class _DDGParser(html.parser.HTMLParser):
    """Collect (title, url, snippet) rows from html.duckduckgo.com results.

    Result rows look like:
      <a rel="nofollow" class="result__a" href="URL">title</a>
      <a class="result__snippet" href="URL">…<b>…</b>…</a>
    (nested <b> inside the snippet anchor is fine — we only keep text data)
    """

    def __init__(self):
        super().__init__()
        self.results = []
        self._cur = {}
        self._in_link = False
        self._in_snippet = False

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        cls = dict(attrs).get("class") or ""
        if "result__a" in cls:
            self._in_link = True
            self._cur = {
                "title": "", "url": dict(attrs).get("href", ""), "snippet": "",
            }
        elif "result__snippet" in cls:
            self._in_snippet = True

    def handle_endtag(self, tag):
        if tag == "a" and self._in_link:
            self._in_link = False
            if self._cur.get("title"):
                self.results.append(self._cur)
                self._cur = {}
        elif tag == "a" and self._in_snippet:
            self._in_snippet = False

    def handle_data(self, data):
        if self._in_link:
            # title of the row being built (a link already flushed has no _cur)
            if self._cur:
                self._cur["title"] += data
        elif self._in_snippet and self.results:
            self.results[-1]["snippet"] += data


_SEARCH_UA = "Mozilla/5.0 (X11; Linux x86_64) soulx-agent/1.0"
# baidu's bot gate keys on a real browser UA + Accept headers; the
# self-identifying agent UA gets an immediate redirect to 安全验证
_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
_BD_COOKIE_FILE = os.path.join(tempfile.gettempdir(), "soulx_baidu_cookies")

# DDG challenges the shared proxy exit IP in bursts (measured: ~4 rapid
# queries pass, then everything is challenged for a couple of minutes). Once
# a challenge is confirmed we stop hitting DDG for a while and search through
# the baidu fallback instead (2026-10-30: the pre-fallback behavior — a 90s
# dead-end message — let a weather task wander for 203s and end in an
# apology, which is what "工具调用总是很慢" was in demo sessions).
_SEARCH_BLOCK_COOLDOWN_S = 90
_search_blocked_until = 0.0  # time.monotonic() deadline


# ---------------------------------------------------------------------------
# baidu fallback — separate bot-detection stack from DDG, so a DDG burst
# challenge never dead-ends a task. Measured 2026-10-30: 3 rapid queries via
# curl through the proxy all returned 6-9 results in ~1s each. Result blocks
# look like
#   <div class="result c-container ..." mu="REAL_URL" ...>
#     <h3 ...><a href="http://www.baidu.com/link?url=...">title</a></h3>
#     ...snippet/aladdin-widget html...
# The mu= attribute carries the real destination URL (the anchor href is a
# baidu redirect); snippets have no stable class, so the block text is
# tag-stripped instead — which also surfaces aladdin direct-answer widgets
# (e.g. weather cards) the model can read without fetching anything.
# ---------------------------------------------------------------------------

def _baidu_search(query, max_results):
    """Returns [(title, url, snippet)] or None on failure/challenge."""
    from urllib.parse import quote
    try:
        proc = subprocess.run(
            [
                "curl", "-s", "-m", str(SEARCH_TIMEOUT_S), "-L",
                *(["-x", PROXY] if PROXY else []),
                "-A", _BROWSER_UA,
                "-H", "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "-H", "Accept-Language: zh-CN,zh;q=0.9",
                "--compressed",
                "-b", _BD_COOKIE_FILE, "-c", _BD_COOKIE_FILE,
                f"https://www.baidu.com/s?wd={quote(str(query))}",
            ],
            capture_output=True, text=True, timeout=SEARCH_TIMEOUT_S + 5,
        )
    except Exception:
        return None
    if proc.returncode != 0 or not proc.stdout:
        return None
    text = proc.stdout
    if "百度安全验证" in text or "result c-container" not in text:
        return None
    rows = []
    for block in text.split("result c-container")[1:]:
        m = re.search(
            r'<h3[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', block, re.S
        )
        if not m:
            continue
        title = re.sub(r"<[^>]+>", "", m.group(2))
        title = " ".join(html.parser.unescape(title).split())
        mu = re.search(r'\smu="(https?://[^"]+)"', block)
        url = mu.group(1) if mu else m.group(1)
        snippet_html = re.sub(
            r"<(script|style)[^>]*>.*?</\1>", " ", block[m.end():], flags=re.S
        )
        snippet = " ".join(
            html.parser.unescape(re.sub(r"<[^>]+>", " ", snippet_html)).split()
        )
        rows.append((title, url, snippet))
        if len(rows) >= max_results:
            break
    return rows or None


def _format_rows(rows):
    lines = []
    for i, (title, url, snippet) in enumerate(rows, 1):
        lines.append(f"{i}. {title}\n   {url}\n   {snippet[:300]}")
    return _clip("\n".join(lines))


def web_search(query, max_results=6):
    """Tavily search API, with the legacy DDG/baidu stack as fallback.

    Tavily became the primary path 2026-10-30 ("工具调用总是很慢"): the
    DDG-html endpoint intermittently bot-challenges the shared proxy exit
    IP, and every challenged query burned a 5s retry plus the baidu detour
    before any result came back. Tavily is an agent-oriented API (no
    bot-gating) whose content snippets are usually enough for the slow
    brain to answer without a follow-up fetch_page round.

    Any Tavily failure — missing key, quota exhausted, network — falls
    back to _fallback_search rather than dead-ending the task.
    """
    q = " ".join(str(query).split())
    key = _tavily_key()
    if key:
        try:
            r = _TAVILY_SESSION.post(
                "https://api.tavily.com/search",
                headers={"Authorization": f"Bearer {key}"},
                json={"query": q, "max_results": max_results},
                timeout=TAVILY_TIMEOUT_S,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            logger.info("tavily search failed (%s); falling back to ddg/baidu", e)
        else:
            rows = [
                (x.get("title", ""), x.get("url", ""), x.get("content", ""))
                for x in data.get("results", [])
                if x.get("url")
            ]
            if rows:
                return _format_rows(rows)
            # Tavily came back empty — try the fallback before giving up
            logger.info("tavily search returned no results; trying fallback")
    return _fallback_search(q, max_results)


def _fallback_search(query, max_results=6):
    """Legacy search stack: DDG html endpoint via POST (curl), baidu fallback.

    Kept as the fallback for when Tavily is unavailable. Measured on this
    box: requests/urllib3 gets a bot-challenge page (HTTP 202,
    TLS-fingerprint level) while curl through the same proxy and UA
    succeeds at the very same moment — so this path uses curl. The DDG
    challenge still appears intermittently on the shared proxy exit IP
    (typically after a burst of queries); on challenge (and during the DDG
    cooldown, and when DDG returns nothing) the query is retried on baidu
    before giving up."""
    global _search_blocked_until
    q = " ".join(str(query).split())
    now = time.monotonic()
    ddg_challenged = now < _search_blocked_until
    text = ""
    if not ddg_challenged:
        for attempt, delay in enumerate((0, 5), start=1):
            if delay:
                time.sleep(delay)
            try:
                proc = subprocess.run(
                    [
                        "curl", "-s", "--max-time", str(SEARCH_TIMEOUT_S),
                        *(["-x", PROXY] if PROXY else []),
                        "-A", _SEARCH_UA,
                        "--data-urlencode", f"q={q}",
                        "https://html.duckduckgo.com/html/",
                    ],
                    capture_output=True, text=True, timeout=SEARCH_TIMEOUT_S + 5,
                )
            except Exception as e:
                return f"search failed: {e}"
            if proc.returncode != 0 or not proc.stdout:
                return f"search failed: curl exit {proc.returncode} {proc.stderr[:120]}"
            text = proc.stdout
            if "anomaly" in text[:4000].lower():
                if attempt < 2:
                    continue
                _search_blocked_until = time.monotonic() + _SEARCH_BLOCK_COOLDOWN_S
                ddg_challenged = True
                break
            break
    if ddg_challenged:
        # DDG is cooling down — baidu is the search path now, not a dead end
        rows = _baidu_search(q, max_results)
        if rows:
            return "(duckduckgo 被限流，以下结果来自百度)\n" + _format_rows(rows)
        return (
            "search unavailable: duckduckgo is rate-limiting this IP and the "
            "baidu fallback failed too. 请改用 fetch_page 打开已知网址，"
            "或基于已有搜索结果作答，不要继续重试搜索。"
        )
    parser = _DDGParser()
    try:
        parser.feed(text)
    except Exception as e:
        return f"search parse failed: {e}"
    rows = [
        (x["title"].strip(), x["url"], " ".join(x["snippet"].split()))
        for x in parser.results
        if x.get("url", "").startswith("http")
    ][: max_results]
    if rows:
        return _format_rows(rows)
    # DDG came back empty — often weak on Chinese queries; try baidu
    bd_rows = _baidu_search(q, max_results)
    if bd_rows:
        return _format_rows(bd_rows)
    return "no results found"


# ---------------------------------------------------------------------------
# fetch_page — retrieve one URL as readable plain text
# ---------------------------------------------------------------------------

class _TextExtractor(html.parser.HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "head"}

    def __init__(self):
        super().__init__()
        self.parts = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip_depth += 1
        if tag in ("p", "div", "br", "li", "h1", "h2", "h3", "tr"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data):
        if not self._skip_depth and data.strip():
            self.parts.append(data)


def fetch_page(url, max_chars=MAX_TOOL_OUTPUT_CHARS):
    url = str(url).strip()
    if not url.startswith(("http://", "https://")):
        return "invalid url"
    try:
        r = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) soulx-agent/1.0"},
            proxies=PROXIES,
            timeout=FETCH_TIMEOUT_S,
        )
        r.raise_for_status()
    except Exception as e:
        return f"fetch failed: {e}"
    ctype = r.headers.get("content-type", "")
    if "html" in ctype:
        ext = _TextExtractor()
        try:
            ext.feed(r.text)
        except Exception:
            pass
        text = re.sub(r"\n{3,}", "\n\n", "".join(ext.parts))
    else:
        text = r.text
    return _clip(text.strip(), max_chars)


# ---------------------------------------------------------------------------
# python_repl — model-written code, subprocess + timeout + throwaway cwd
# ---------------------------------------------------------------------------

def python_repl(code, workdir=None):
    code = str(code)
    if not code.strip():
        return "no code"
    wd = workdir or tempfile.mkdtemp(prefix="soulx_agent_")
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", code],
            capture_output=True,
            text=True,
            timeout=REPL_TIMEOUT_S,
            cwd=wd,
        )
    except subprocess.TimeoutExpired:
        return f"timeout after {REPL_TIMEOUT_S}s"
    out = (proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
    if not out.strip():
        out = f"(no output, exit code {proc.returncode})"
    return _clip(out)


# ---------------------------------------------------------------------------
# get_datetime
# ---------------------------------------------------------------------------

def get_datetime():
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S %A (local server time)")


# ---------------------------------------------------------------------------
# registry: schema for the LLM + executor
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web (Tavily API, automatic fallback). Returns an "
                "optional summary plus numbered results with title, url and a "
                "short snippet. Use for facts you are not sure about or that "
                "change over time."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "search query"},
                    "max_results": {"type": "integer", "description": "max results, default 6"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_page",
            "description": (
                "Fetch one web page and return its readable plain text. "
                "Use on a promising url from web_search when the snippet is "
                "not enough."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "http(s) url to fetch"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "python_repl",
            "description": (
                "Execute Python code (stdlib only, no interactive input) and "
                "return stdout/stderr. Use for calculation, data processing "
                "or anything deterministic. Keep runtime under a few seconds."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "python code to run"},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_datetime",
            "description": "Get the current local date and time.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

TOOL_EXECUTORS = {
    "web_search": lambda args: web_search(
        args.get("query", ""), int(args.get("max_results") or 6)
    ),
    "fetch_page": lambda args: fetch_page(args.get("url", "")),
    "python_repl": lambda args: python_repl(args.get("code", "")),
    "get_datetime": lambda args: get_datetime(),
}


def execute_tool(name, arguments):
    """Run one tool; never raises — errors become observations for the LLM."""
    fn = TOOL_EXECUTORS.get(name)
    if fn is None:
        return f"unknown tool: {name}"
    try:
        return _clip(str(fn(arguments or {})))
    except Exception as e:
        return f"tool error: {e}"


# ===========================================================================
# v2 工具层（agent-runtime P1c，手册 §11）。上方的 legacy TOOL_SCHEMAS /
# TOOL_EXECUTORS / execute_tool 原样保留 —— legacy 慢脑行为不变。新 runtime
# 默认工具集 = web_search / fetch_page / get_datetime / calculate；
# python_repl 不进默认集，AGENT_TOOLS_PYTHON_REPL=1 显式恢复。
# ===========================================================================

import ast as _ast
import ipaddress as _ip
import socket as _socket
import urllib.parse as _urlparse
from dataclasses import dataclass, field as _dcfield
from zoneinfo import ZoneInfo as _ZoneInfo

# balldontlie 体育数据（agent/bdl.py，2026-11-09）：拉取 + webhook 推送。
# bdl.py 只在函数内延迟 import agent.tools（ToolResult），无循环依赖。
from agent import bdl as _bdl

# ---- 统一工具返回（手册 §11）---------------------------------------------


@dataclass
class ToolResult:
    call_id: str = ""
    status: str = "ok"          # ok|empty|error|canceled|timeout|unavailable
    data: str = ""
    sources: list = _dcfield(default_factory=list)
    error_code: str = ""
    elapsed_ms: int = 0
    truncated: bool = False
    side_effect: str = "none"   # none|committed|unknown

    def to_observation(self):
        """给慢脑看的字符串观测；失败不得伪装成成功内容。"""
        if self.status == "ok":
            return self.data if not self.truncated else (
                self.data + "\n[truncated]")
        if self.status == "empty":
            return "(no results)"
        return f"[{self.status}] {self.error_code or self.status}: {self.data}"


# ---- fetch_page v2：按连接目的地址核验 + 每跳重定向核验（SSRF） ----------

FETCH_V2_MAX_REDIRECTS = 4
FETCH_V2_MAX_BYTES = 2 * 1024 * 1024   # 响应体上限
FETCH_V2_TIMEOUT_S = 10

# 连接目的地址必须全部落在公网单播；私网/回环/链路本地(含云元数据
# 169.254.169.254)/保留/组播/CGNAT/未指定 全部拒绝。
def _ip_is_public(addr: str) -> bool:
    try:
        ip = _ip.ip_address(addr)
    except ValueError:
        return False
    if isinstance(ip, _ip.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return not (
        ip.is_private or ip.is_loopback or ip.is_link_local
        or ip.is_reserved or ip.is_multicast or ip.is_unspecified
        or ip in _ip.ip_network("100.64.0.0/10")
    )


def _validate_public_host(hostname: str) -> list:
    """解析并核验 hostname 的全部连接目的地址。返回地址列表；
    任一地址不合规即抛 ValueError（不只做字符串黑名单）。"""
    if not hostname:
        raise ValueError("empty host")
    # 主机名本身是 IP 的情况（含十进制/十六进制整数形式）
    try:
        ip = _ip.ip_address(hostname)
    except ValueError:
        pass
    else:
        if not _ip_is_public(str(ip)):
            raise ValueError(f"non-public destination {hostname}")
        return [str(ip)]
    if (hostname == "localhost" or hostname.endswith(".local")
            or hostname.endswith(".internal") or hostname.endswith(".lan")):
        raise ValueError(f"non-public host {hostname}")
    try:
        infos = _socket.getaddrinfo(
            hostname, None, _socket.AF_UNSPEC, _socket.SOCK_STREAM)
    except _socket.gaierror as e:
        raise ValueError(f"dns resolution failed: {e}") from None
    addrs = {info[4][0] for info in infos}
    for a in addrs:
        if not _ip_is_public(a):
            raise ValueError(f"non-public destination {hostname} -> {a}")
    return sorted(addrs)


def fetch_page_v2(url, max_chars=MAX_TOOL_OUTPUT_CHARS,
                  call_id="", deadline_s=FETCH_V2_TIMEOUT_S):
    """HTTP(S) 公网读取：URL 限制、逐跳重定向核验、响应大小/输出长度/
    deadline 限制。返回 ToolResult，不抛异常。"""
    t0 = time.monotonic()
    url = str(url or "").strip()

    def _res(status, data="", error_code=""):
        return ToolResult(
            call_id=call_id, status=status, data=data, error_code=error_code,
            elapsed_ms=int((time.monotonic() - t0) * 1000))

    cur = url
    for _hop in range(FETCH_V2_MAX_REDIRECTS + 1):
        try:
            parsed = _urlparse.urlsplit(cur)
        except Exception:
            return _res("error", "invalid url", "bad_url")
        if parsed.scheme not in ("http", "https"):
            return _res("error", "only http(s) allowed", "bad_scheme")
        try:
            _validate_public_host(parsed.hostname or "")
        except ValueError as e:
            # 本机/内网/元数据地址：明确拒绝，绝不返回内容
            return _res("error", str(e), "ssrf_blocked")
        remaining = deadline_s - (time.monotonic() - t0)
        if remaining <= 0:
            return _res("timeout", "deadline exceeded", "deadline")
        try:
            r = requests.get(
                cur,
                headers={"User-Agent":
                         "Mozilla/5.0 (X11; Linux x86_64) soulx-agent/1.0"},
                proxies=PROXIES, timeout=remaining, stream=True,
                allow_redirects=False)
        except requests.exceptions.Timeout:
            return _res("timeout", "fetch timed out", "timeout")
        except Exception as e:
            return _res("error", f"fetch failed: {e}", "network")
        if r.status_code in (301, 302, 303, 307, 308):
            loc = r.headers.get("location", "")
            r.close()
            if not loc:
                return _res("error", "redirect without location", "bad_redirect")
            cur = _urlparse.urljoin(cur, loc)
            continue
        try:
            if r.status_code >= 400:
                r.close()
                return _res("error", f"http {r.status_code}", f"http_{r.status_code}")
            # 流式截断：超过上限即停，不把超大响应整段读进内存
            chunks, total = [], 0
            for chunk in r.iter_content(chunk_size=65536):
                total += len(chunk)
                if total > FETCH_V2_MAX_BYTES:
                    r.close()
                    return _res("error", "response too large", "too_large")
                chunks.append(chunk)
                if b"</html" in b"".join(chunks[-2:]):
                    break
            body = b"".join(chunks)
        finally:
            r.close()
        ctype = r.headers.get("content-type", "")
        try:
            text = body.decode(r.encoding or "utf-8", errors="replace")
        except Exception:
            text = body.decode("utf-8", errors="replace")
        if "html" in ctype:
            ext = _TextExtractor()
            try:
                ext.feed(text)
            except Exception:
                pass
            text = re.sub(r"\n{3,}", "\n\n", "".join(ext.parts))
        text = text.strip()
        if not text:
            return _res("empty", "", "empty_body")
        truncated = len(text) > max_chars
        return ToolResult(
            call_id=call_id, status="ok",
            data=_clip(text, max_chars),
            sources=[{"url": cur, "retrieved_at":
                      time.strftime("%Y-%m-%dT%H:%M:%S")}],
            elapsed_ms=int((time.monotonic() - t0) * 1000),
            truncated=truncated)
    return _res("error", "too many redirects", "redirect_loop")


# ---- calculate：AST 白名单纯计算（替代 python_repl 的默认能力） ----------

_CALC_MAX_EXPR_CHARS = 200
_CALC_MAX_NODES = 60
_CALC_MAX_INT = 10 ** 15
_CALC_MAX_EXPONENT = 10 ** 6

_CALC_ALLOWED_NODES = (
    _ast.Expression, _ast.BinOp, _ast.UnaryOp, _ast.Constant,
    _ast.Add, _ast.Sub, _ast.Mult, _ast.Div, _ast.FloorDiv, _ast.Mod,
    _ast.Pow, _ast.USub, _ast.UAdd, _ast.Call, _ast.Load, _ast.Name,
)
_CALC_FUNCS = {
    "abs": abs, "round": round, "min": min, "max": max,
}


def _days_between(d1, d2):
    a = datetime.date.fromisoformat(str(d1))
    b = datetime.date.fromisoformat(str(d2))
    return (b - a).days


_CALC_FUNCS["days_between"] = _days_between


def calculate(expression, call_id=""):
    """算术表达式 / 日期差白名单计算。AST 解析 + 节点与数值规模限制，
    绝不 eval 任意 Python。返回 ToolResult。"""
    t0 = time.monotonic()
    expr = str(expression or "").strip()
    if not expr:
        return ToolResult(call_id=call_id, status="error",
                          error_code="empty_expr", data="no expression")
    if len(expr) > _CALC_MAX_EXPR_CHARS:
        return ToolResult(call_id=call_id, status="error",
                          error_code="too_long",
                          data="expression too long")
    try:
        tree = _ast.parse(expr, mode="eval")
    except SyntaxError as e:
        return ToolResult(call_id=call_id, status="error",
                          error_code="syntax", data=f"syntax error: {e}")

    nodes = list(_ast.walk(tree))
    if len(nodes) > _CALC_MAX_NODES:
        return ToolResult(call_id=call_id, status="error",
                          error_code="too_complex",
                          data="expression too complex")

    def _eval(node):
        if isinstance(node, _ast.Expression):
            return _eval(node.body)
        if isinstance(node, _ast.Constant):
            if isinstance(node.value, str):
                return node.value   # 仅 days_between 接受；别处运算会 TypeError
            if isinstance(node.value, bool) or not isinstance(
                    node.value, (int, float)):
                raise ValueError("non-numeric literal")
            if isinstance(node.value, int) and abs(node.value) > _CALC_MAX_INT:
                raise ValueError("literal too large")
            if isinstance(node.value, float) and abs(node.value) > _CALC_MAX_INT:
                raise ValueError("literal too large")
            return node.value
        if isinstance(node, _ast.BinOp):
            # ** 的指数规模单限：防 10**10**8 之类指数爆炸
            if isinstance(node.op, _ast.Pow):
                e = _eval(node.right)
                if not isinstance(e, (int, float)) or abs(e) > 1000:
                    raise ValueError("exponent too large")
            if isinstance(node.op, (_ast.Add, _ast.Sub, _ast.Mult, _ast.Div,
                                    _ast.FloorDiv, _ast.Mod, _ast.Pow)):
                a, b = _eval(node.left), _eval(node.right)
                try:
                    if isinstance(node.op, _ast.Add):
                        v = a + b
                    elif isinstance(node.op, _ast.Sub):
                        v = a - b
                    elif isinstance(node.op, _ast.Mult):
                        v = a * b
                    elif isinstance(node.op, _ast.Div):
                        v = a / b
                    elif isinstance(node.op, _ast.FloorDiv):
                        v = a // b
                    elif isinstance(node.op, _ast.Mod):
                        v = a % b
                    else:
                        v = a ** b
                except ZeroDivisionError:
                    raise ValueError("division by zero") from None
                except OverflowError:
                    raise ValueError("overflow") from None
                if isinstance(v, float) and (v != v or abs(v) == float("inf")):
                    raise ValueError("non-finite result") from None
                if abs(v) > _CALC_MAX_INT * 1000:
                    raise ValueError("result too large") from None
                return v
            raise ValueError("operator not allowed")
        if isinstance(node, _ast.UnaryOp):
            v = _eval(node.operand)
            if isinstance(node.op, _ast.USub):
                return -v
            if isinstance(node.op, _ast.UAdd):
                return +v
            raise ValueError("operator not allowed")
        if isinstance(node, _ast.Call):
            if not isinstance(node.func, _ast.Name):
                raise ValueError("call target not allowed")
            if node.func.id not in _CALC_FUNCS:
                raise ValueError(f"function not allowed: {node.func.id}")
            if node.keywords:
                raise ValueError("keyword args not allowed")
            args = [_eval(a) for a in node.args]
            if node.func.id == "days_between":
                if len(args) != 2:
                    raise ValueError("days_between takes 2 dates")
                try:
                    return _days_between(args[0], args[1])
                except (ValueError, TypeError):
                    raise ValueError("invalid date (YYYY-MM-DD)") from None
            return _CALC_FUNCS[node.func.id](*args)
        raise ValueError(f"node not allowed: {type(node).__name__}")

    # 白名单之外的节点类型直接拒绝（属性/下标/lambda/推导式…）。
    # Name 只允许作为白名单函数的调用目标——裸名/变量引用拒绝。
    call_funcs = {n.func.id for n in nodes
                  if isinstance(n, _ast.Call) and isinstance(n.func, _ast.Name)}

    def _precheck():
        for n in nodes:
            if isinstance(n, _ast.Name):
                if n.id not in _CALC_FUNCS or n.id not in call_funcs:
                    raise ValueError(f"name not allowed: {n.id}")
                continue
            if not isinstance(n, _CALC_ALLOWED_NODES):
                raise ValueError(
                    f"disallowed syntax: {type(n).__name__}")

    try:
        _precheck()
        value = _eval(tree)
    except (ValueError, TypeError) as e:
        return ToolResult(call_id=call_id, status="error",
                          error_code="rejected",
                          data=f"calculation rejected: {e}",
                          elapsed_ms=int((time.monotonic() - t0) * 1000))
    if isinstance(value, float) and value == int(value) and abs(value) < 1e15:
        value = int(value)
    return ToolResult(call_id=call_id, status="ok", data=str(value),
                      elapsed_ms=int((time.monotonic() - t0) * 1000))


# ---- get_datetime v2：IANA 时区（默认 Asia/Shanghai）----------------------


def get_datetime_v2(timezone="Asia/Shanghai", call_id=""):
    t0 = time.monotonic()
    tz_name = str(timezone or "Asia/Shanghai").strip()
    try:
        tz = _ZoneInfo(tz_name)
    except Exception:
        return ToolResult(call_id=call_id, status="error",
                          error_code="bad_timezone",
                          data=f"unknown IANA timezone: {tz_name}")
    now = datetime.datetime.now(tz)
    return ToolResult(
        call_id=call_id, status="ok",
        data=f"{now:%Y-%m-%d %H:%M:%S %A} ({tz_name})",
        elapsed_ms=int((time.monotonic() - t0) * 1000))


# ---- v2 注册表 -------------------------------------------------------------


def _web_search_v2(args, call_id=""):
    t0 = time.monotonic()
    out = web_search(args.get("query", ""),
                     int(args.get("max_results") or 6))
    out = str(out)
    status = "ok"
    if out.startswith(("search failed", "search unavailable",
                       "search parse failed")):
        status, error_code = "unavailable", "search_unavailable"
    elif out.strip() in ("no results found", ""):
        status, error_code = "empty", "no_results"
    else:
        error_code = ""
    return ToolResult(call_id=call_id, status=status, data=out,
                      error_code=error_code,
                      elapsed_ms=int((time.monotonic() - t0) * 1000))


NEW_TOOL_EXECUTORS = {
    "web_search": _web_search_v2,
    "fetch_page": lambda args, call_id="": fetch_page_v2(
        args.get("url", ""), call_id=call_id),
    "get_datetime": lambda args, call_id="": get_datetime_v2(
        args.get("timezone") or "Asia/Shanghai", call_id=call_id),
    "calculate": lambda args, call_id="": calculate(
        args.get("expression", ""), call_id=call_id),
    # balldontlie（agent/bdl.py）：体育比赛专用数据源
    "sports_games": lambda args, call_id="": _bdl.sports_games(
        args.get("sport") or "nba", args.get("date") or "",
        args.get("team") or "", call_id=call_id),
    "sports_events": lambda args, call_id="": _bdl.sports_events(
        int(args.get("count") or 20), args.get("sport") or "",
        call_id=call_id),
}

NEW_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web. Returns numbered results with title, url "
                "and a short snippet. Use for facts you are not sure about "
                "or that change over time."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string",
                              "description": "search query"},
                    "max_results": {"type": "integer",
                                    "description": "max results, default 6"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_page",
            "description": (
                "Fetch one public web page and return its readable plain "
                "text. Only http(s) public addresses; private/localhost "
                "addresses are rejected. Use when a search snippet is not "
                "enough."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string",
                            "description": "http(s) url to fetch"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_datetime",
            "description": (
                "Get the current date and time in an IANA timezone "
                "(default Asia/Shanghai)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {"type": "string",
                                 "description":
                                 "IANA timezone, e.g. Asia/Shanghai"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Evaluate an arithmetic expression or date difference. "
                "Allowed: + - * / // % **, abs/round/min/max, "
                "days_between('2026-11-06','2026-12-25'). No variables, "
                "no function definitions, no arbitrary code."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string",
                                   "description": "expression to evaluate"},
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sports_games",
            "description": (
                "Query official schedules, scores and results for "
                "basketball/american-football/baseball games (NBA, NFL, "
                "MLB) from the balldontlie API. For ANY question about "
                "game scores, results, or what games are on, use this "
                "FIRST instead of web_search — web search is unreliable "
                "for same-day scores. date is a UTC date (YYYY-MM-DD); "
                "games played in the Chinese evening are usually on the "
                "PREVIOUS UTC date, so query both today and yesterday "
                "when unsure. team accepts Chinese or English "
                "(e.g. '热火', 'heat', 'lakers')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sport": {
                        "type": "string", "enum": ["nba", "nfl", "mlb"],
                        "description": "default nba",
                    },
                    "date": {
                        "type": "string",
                        "description": "UTC date YYYY-MM-DD; empty = "
                                       "today + yesterday (UTC)",
                    },
                    "team": {
                        "type": "string",
                        "description": "filter by team name/abbr "
                                       "(Chinese ok); empty = all teams",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sports_events",
            "description": (
                "Read recent game events pushed in real time by the "
                "balldontlie webhook (game started / game ended with "
                "final scores). Local and instant — when the user asks "
                "about a game that may have been played in the last few "
                "hours ('那场比赛打完了吗/比分多少'), check this first, "
                "then sports_games."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer",
                        "description": "max events to return, default 20",
                    },
                    "sport": {
                        "type": "string",
                        "description": "optional filter, e.g. nba",
                    },
                },
                "required": [],
            },
        },
    },
]

if os.environ.get("AGENT_TOOLS_PYTHON_REPL", "").strip() == "1":
    # 显式开关恢复 python_repl（任意代码子进程，非安全沙箱——手册 §11）
    NEW_TOOL_SCHEMAS.append(TOOL_SCHEMAS[2])
    NEW_TOOL_EXECUTORS["python_repl"] = lambda args, call_id="": ToolResult(
        call_id=call_id, status="ok",
        data=python_repl(args.get("code", "")))


def execute_tool_v2(name, arguments, call_id=""):
    """新 runtime 的工具执行入口：返回 ToolResult，不抛异常。
    参数在此验证（不只依赖 JSON Schema 提示）。"""
    fn = NEW_TOOL_EXECUTORS.get(name)
    if fn is None:
        return ToolResult(call_id=call_id, status="error",
                          error_code="unknown_tool",
                          data=f"unknown tool: {name}")
    try:
        return fn(arguments or {}, call_id=call_id)
    except Exception as e:
        return ToolResult(call_id=call_id, status="error",
                          error_code="exception", data=f"tool error: {e}")
