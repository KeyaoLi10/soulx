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
import os
import re
import subprocess
import sys
import tempfile
import time
import datetime

import requests

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
    """DDG html endpoint via POST (curl), baidu fallback.

    Measured on this box: requests/urllib3 gets a bot-challenge page (HTTP
    202, TLS-fingerprint level) while curl through the same proxy and UA
    succeeds at the very same moment — so the search tool uses curl. The
    DDG challenge still appears intermittently on the shared proxy exit IP
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
                "Search the web (DuckDuckGo, automatic baidu fallback). "
                "Returns numbered results with title, url and a short "
                "snippet. Use for facts you are not sure about or that "
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
