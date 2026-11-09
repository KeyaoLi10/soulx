# -*- coding: utf-8 -*-
"""Offline tests for agent/tools.py v2 layer (P1c, 手册 §11)。

无外网依赖：SSRF 用 127.0.0.1 本机 HTTP 服务验证拒绝路径（这正是要挡的
地址）；DNS 解析核验用 monkeypatch 的 getaddrinfo；公网放行路径留待实机
验证（未验证声明见 docs/agent-runtime实施前审查-20261106.md）。

Run: /home/research/krisl/soulx/venv/bin/python -I scripts/agent_tools_test.py
"""
import http.server
import socket
import sys
import threading
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "repo" / "dialogue_system"))

from agent import tools as T  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok  {name}")
    else:
        FAILS.append(name)
        print(f"FAIL  {name}  {detail}")


# ---- calculate：AST 白名单 ------------------------------------------------
r = T.calculate("1 + 2 * 3")
check("arith precedence", r.status == "ok" and r.data == "7", r.data)
r = T.calculate("(120 - 30) / 4")
check("parens division", r.status == "ok" and r.data == "22.5", r.data)
r = T.calculate("2 ** 10")
check("pow allowed", r.status == "ok" and r.data == "1024", r.data)
r = T.calculate("abs(-5) + round(2.7) + min(3, 9) + max(1, 4)")
check("whitelist funcs", r.status == "ok" and r.data == "15", r.data)
r = T.calculate("days_between('2026-11-06', '2026-12-25')")
check("date diff", r.status == "ok" and r.data == "49", r.data)

for expr, why in [
    ("__import__('os').system('id')", "dunder import"),
    ("open('/etc/passwd')", "builtin call"),
    ("x = 5", "assignment"),
    ("[i for i in range(10)]", "comprehension"),
    ("'abc'.upper()", "method call"),
    ("lambda: 1", "lambda"),
    ("1 if 2 else 3", "ternary"),
    ("foo(1)", "unknown name call"),
    ("1/0", "division by zero"),
    ("9" + "**9" * 6, "exponent bomb"),
    ("1" + "+1" * 150, "too long"),
    ("10**100000000", "huge exponent literal"),
    ("1" + "+1" * 35, "too many nodes"),
]:
    r = T.calculate(expr)
    check(f"reject: {why}", r.status == "error", f"status={r.status} data={r.data[:60]}")

# 十六进制整数字面量是合法 Constant —— 属白名单内（规模限制仍然生效）
r = T.calculate("0x10 + 1")
check("hex int allowed", r.status == "ok" and r.data == "17", r.data)
# 巨大数值字面量拒绝
r = T.calculate(str(10 ** 20))
check("huge literal rejected", r.status == "error")

# ---- get_datetime v2：IANA 时区 -------------------------------------------
r = T.get_datetime_v2("Asia/Shanghai")
check("default tz works", r.status == "ok" and "Asia/Shanghai" in r.data,
      r.data)
r = T.get_datetime_v2("UTC")
check("utc works", r.status == "ok" and "(UTC)" in r.data, r.data)
r = T.get_datetime_v2("America/New_York")
check("iana tz works", r.status == "ok" and "America/New_York" in r.data)
r = T.get_datetime_v2("Beijing/Time")
check("bad tz rejected", r.status == "error"
      and r.error_code == "bad_timezone")
r = T.get_datetime_v2("")
check("empty tz falls back to default", r.status == "ok"
      and "Asia/Shanghai" in r.data)

# ---- SSRF：目的地址核验 ----------------------------------------------------
for bad in ["127.0.0.1", "10.1.2.3", "192.168.1.1", "172.16.0.5",
            "169.254.169.254", "0.0.0.0", "::1", "fd00::1", "100.64.0.1",
            "224.0.0.1", "2130706433", "0x7f000001"]:
    check(f"blocked ip {bad}", not T._ip_is_public(bad))
for good in ["8.8.8.8", "1.1.1.1", "93.184.216.34", "2606:4700:4700::1111"]:
    check(f"public ip {good}", T._ip_is_public(good))
try:
    T._validate_public_host("localhost")
    ok = False
except ValueError:
    ok = True
check("localhost blocked", ok)
try:
    T._validate_public_host("printer.lan")
    ok = False
except ValueError:
    ok = True
check("lan suffix blocked", ok)

# DNS 解析核验（monkeypatch getaddrinfo，无网络）
_orig_getaddrinfo = socket.getaddrinfo


def _fake_getaddrinfo(host, *a, **kw):
    table = {
        "public.example": [(socket.AF_INET, socket.SOCK_STREAM, 6, "",
                            ("93.184.216.34", 0))],
        "internal.example": [(socket.AF_INET, socket.SOCK_STREAM, 6, "",
                              ("10.0.0.9", 0))],
        "meta.example": [(socket.AF_INET, socket.SOCK_STREAM, 6, "",
                          ("169.254.169.254", 0))],
        "mixed.example": [(socket.AF_INET, socket.SOCK_STREAM, 6, "",
                           ("93.184.216.34", 0)),
                          (socket.AF_INET, socket.SOCK_STREAM, 6, "",
                           ("192.168.0.2", 0))],
    }
    return table[host]


socket.getaddrinfo = _fake_getaddrinfo
try:
    T._validate_public_host("public.example")
    ok = True
except ValueError:
    ok = False
check("public hostname passes", ok)
for h in ["internal.example", "meta.example", "mixed.example"]:
    try:
        T._validate_public_host(h)
        ok = False
    except ValueError:
        ok = True
    check(f"hostname resolves private blocked: {h}", ok)
socket.getaddrinfo = _orig_getaddrinfo

# ---- fetch_page_v2：本机服务必须被拒（离线可测的 SSRF 路径） ---------------
srv = http.server.HTTPServer(
    ("127.0.0.1", 0), lambda *a, **kw: None)  # placeholder, replaced below


class _Handler(http.server.BaseHTTPRequestHandler):
    hits = 0

    def do_GET(self):
        _Handler.hits += 1
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "/secret")
            self.end_headers()
        else:
            body = b"internal secret data"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, *a):
        pass


srv = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
port = srv.server_address[1]
th = threading.Thread(target=srv.serve_forever, daemon=True)
th.start()

r = T.fetch_page_v2(f"http://127.0.0.1:{port}/secret", call_id="c1")
check("loopback fetch blocked", r.status == "error"
      and r.error_code == "ssrf_blocked", f"{r.status}/{r.error_code}")
check("blocked fetch never reaches server", _Handler.hits == 0)
r = T.fetch_page_v2(f"http://localhost:{port}/secret")
check("localhost hostname blocked", r.status == "error"
      and r.error_code == "ssrf_blocked")
r = T.fetch_page_v2(f"http://127.0.0.1:{port}/redirect")
check("redirect to internal still blocked", r.status == "error"
      and r.error_code == "ssrf_blocked")
r = T.fetch_page_v2("ftp://example.com/x")
check("non-http scheme rejected", r.status == "error"
      and r.error_code == "bad_scheme")
r = T.fetch_page_v2("not a url")
check("garbage url rejected", r.status == "error")
check("server never touched by blocked fetches", _Handler.hits == 0)
srv.shutdown()

# ---- ToolResult 语义 --------------------------------------------------------
r = T.ToolResult(status="ok", data="abc")
check("ok observation is data", r.to_observation() == "abc")
r = T.ToolResult(status="empty")
check("empty observation marked", r.to_observation() == "(no results)")
r = T.ToolResult(status="unavailable", error_code="search_unavailable")
check("failure never fake success", "search_unavailable" in
      r.to_observation() and not r.to_observation().startswith("1."))

# ---- 注册表与 python_repl 开关 ----------------------------------------------
names = {s["function"]["name"] for s in T.NEW_TOOL_SCHEMAS}
check("default v2 toolset", names == {"web_search", "fetch_page",
                                      "get_datetime", "calculate",
                                      "sports_games", "sports_events"}, names)
check("python_repl off by default", "python_repl" not in names)
r = T.execute_tool_v2("calculate", {"expression": "6*7"})
check("execute_tool_v2 works", r.status == "ok" and r.data == "42")
r = T.execute_tool_v2("nope", {})
check("unknown tool error result", r.status == "error"
      and r.error_code == "unknown_tool")
r = T.execute_tool_v2("fetch_page", {"url": f"http://10.9.9.9:{port}/x"})
check("executor validates args (private ip)", r.status == "error"
      and r.error_code == "ssrf_blocked")

# legacy 层原样（P1c 保护清单）
check("legacy TOOL_SCHEMAS untouched",
      [s["function"]["name"] for s in T.TOOL_SCHEMAS] ==
      ["web_search", "fetch_page", "python_repl", "get_datetime"])
check("legacy execute_tool still string-based",
      isinstance(T.execute_tool("get_datetime", {}), str))

print()
if FAILS:
    print(f"agent_tools_test: {len(FAILS)} FAILED: {FAILS}")
    sys.exit(1)
print("agent_tools_test: all passed")
