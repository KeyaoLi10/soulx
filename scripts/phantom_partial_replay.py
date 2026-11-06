# -*- coding: utf-8 -*-
"""浏览器端回放：2026-11-06 14:35 会话 65172a9e 的前端事件序列，复现
"幻影『你帮我』" 是否能在当前 script.js 下出现。

用 playwright 起真页面（frontend/ 由本地静态服务器提供），add_init_script
把 window.WebSocket 换成假实现，然后按 web.log 的真实时间序推
user_transcription / turn_start / text_response 事件，最后读 chatLog DOM。

Run: /home/research/krisl/soulx/venv/bin/python -I scripts/phantom_partial_replay.py
"""
import json
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from playwright.sync_api import sync_playwright

FRONTEND = Path(__file__).resolve().parents[1] / "repo" / "dialogue_system" / "frontend"
CHROME = ("/home/research/.cache/ms-playwright/chromium_headless_shell-1228/"
          "chrome-headless-shell-linux64/chrome-headless-shell")
CHROME_LIBS = "/tmp/chromium-libs/ext/usr/lib/x86_64-linux-gnu"

FAKE_WS = """
window.__pushed = [];
window.__sockets = [];
class FakeWebSocket {
  constructor(url) {
    this.url = url;
    this.readyState = 1; // OPEN
    this.binaryType = 'arraybuffer';
    this.onopen = null; this.onclose = null; this.onmessage = null; this.onerror = null;
    window.__sockets.push(this);
    const self = this;
    setTimeout(() => { if (self.onopen) self.onopen(); }, 0);
  }
  send(data) { window.__pushed.push(String(data)); }
  close() {
    this.readyState = 3;
    if (this.onclose) this.onclose();
  }
}
FakeWebSocket.OPEN = 1; FakeWebSocket.CONNECTING = 0; FakeWebSocket.CLOSING = 2; FakeWebSocket.CLOSED = 3;
window.WebSocket = FakeWebSocket;
window.__feed = (event, data) => {
  const s = window.__sockets[window.__sockets.length - 1];
  if (s && s.onmessage) s.onmessage({ data: JSON.stringify({ event, data }) });
};
"""


def serve():
    import os

    class H(SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass

    os.chdir(FRONTEND)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def up(text):
    return ("user_transcription", {"text": text})


def uf(text, append):
    return ("user_transcription", {"text": text, "final": True, "append": append})


def ts():
    return ("turn_start", {})


def tr(text):
    return ("text_response", {"text": text})


# 14:34:47 - 14:35:23 的真实序列（web.log 逐行对照；turn_start 时点按
# 管线启动推定）。append 旗标按 _is_continuation_turn 语义计算。
SEQ = [
    up("喂你好"),
    uf("喂，你好你好。", False),            # 48,922 会话首轮
    up("可以"),
    up("可以听到"),
    up("可以听到吗"),
    uf("我可以听到吗？", True),             # 49,633 0.7s 前轮 + 无助手声 → fold
    ts(),                                   # ~50 表达轮启动
    tr("喂喂，听得见呢！我在呢，你好呀。"),  # 55,598
    uf("嗯，好的好的。", False),             # 00,421 距上轮 10.8s
    ts(),                                   # ~02,2 表达轮（后被掐 rc=None）
    up("你帮我"),                           # 02,833
    up("你帮我查"),                         # 02,995
    up("你帮我查一"),                       # 03,168
    up("你帮我查一下"),                     # 03,331
    up("不对"),                             # 04,201 游离 partial
    up("我最近"),                           # 04,382 游离 partial
    uf("嗯，你帮我查一下。", True),          # 04,383 3.96s < 4s → fold
    up("北京今天"),                         # 04,565
    up("北京今天的天气"),                    # 04,878
    up("北京今天的天气怎么样"),               # 05,195
    up("北京今天的天气怎么样了"),             # 05,355
    up("北京今天的天气怎么样吧"),             # 05,629
    uf("北京今天的天气怎么样吧？", True),     # 05,899 1.5s → fold
    ts(),                                   # 07,7 澄清上口
    tr("您想让我帮您查什么内容呢？"),
    ts(),                                   # 12,7 回执
    tr("北京天气已经安排好了，稍后告诉你结果。"),
    up("到了"),                             # 17,127
    up("挺好的"),                           # 17,360
    up("然后"),                             # 17,801
    uf("呃，嗯嗯嗯，好的，好的。", False),    # 17,959 距上轮 12s
    up("然后呃"),                           # 18,606
    up("然后呃你再查一"),                    # 19,009
    up("然后呃你再查一下"),                  # 19,179
    up("然后呃你再查一下那"),                 # 19,347
    up("然后呃你再查一下那个"),               # 19,592
    up("备注"),                             # 20,480 缓冲重置
    up("对左下"),                           # 20,672
    uf("然后呃，你再查一下那个湖人队。", True),  # 20,672 2.7s → fold
    up("对昨天的"),                         # 20,867
    up("对昨天的比赛"),                      # 21,033
    up("昨天的比赛结果"),                     # 21,435
    up("昨天的比赛结果怎么"),                 # 21,601
    up("昨天的比赛结果怎么样"),               # 21,775
    up("昨天的比赛结果怎么样吧"),             # 22,403
    uf("你昨天的比赛结果怎么样吧？", True),   # 22,942 → fold
]


def dump_bubbles(page):
    return page.evaluate(
        "() => Array.from(document.querySelectorAll('#chatLog .msg')).map(m => "
        "(m.className.includes('msg-user') ? 'YOU: ' : 'AI:  ') "
        "+ m.querySelector('.msg-body').textContent)")


def run_seq(playwright, url, seq, label):
    browser = playwright.chromium.launch(
        executable_path=CHROME,
        env={**__import__("os").environ,
             "LD_LIBRARY_PATH": CHROME_LIBS})
    page = browser.new_page()
    page.add_init_script(FAKE_WS)
    page.goto(url + "/index.html")
    page.wait_for_function("window.__sockets.length > 0")
    page.wait_for_function(
        "document.querySelector('#loadingOverlay') === null || true")
    for event, data in seq:
        page.evaluate("([e, d]) => window.__feed(e, d)", [event, data])
        page.wait_for_timeout(5)
    bubbles = dump_bubbles(page)
    print(f"===== {label} =====")
    for b in bubbles:
        print("  " + b)
    browser.close()
    return bubbles


def main():
    srv, url = serve()
    with sync_playwright() as pw:
        base = run_seq(pw, url, SEQ, "A: 按日志原序（fragment1 append=True）")

        # 变体 B：fragment 1 的 final append=False（3.96s 临界值另一侧）
        seq_b = list(SEQ)
        for i, (e, d) in enumerate(seq_b):
            if d.get("text") == "嗯，你帮我查一下。":
                seq_b[i] = (e, dict(d, append=False))
        run_seq(pw, url, seq_b, "B: fragment1 append=False")

        # 变体 C：表达轮 turn_start 晚于 partials（晚启动竞态）
        seq_c = list(SEQ)
        # 简化：在 fragment1 final 前插入一个 turn_start
        idx = next(i for i, (e, d) in enumerate(seq_c)
                   if d.get("text") == "嗯，你帮我查一下。")
        seq_c.insert(idx, ts())
        run_seq(pw, url, seq_c, "C: fragment1 final 前插入 turn_start")

        # 变体 D：turn_start 落在「你帮我查」和「你帮我查一」两个 partial
        # 之间（14:35:03 实况，幻影『你帮我』的精确复现路径）——修复后
        # fold 扫掉全部未冻结 partial 气泡，不得再留下孤行
        seq_d = list(SEQ)
        idx = next(i for i, (e, d) in enumerate(seq_d)
                   if d.get("text") == "你帮我查一")
        seq_d.insert(idx, ts())
        run_seq(pw, url, seq_d, "D: turn_start 插在 partial 之间（原始事故）")
        phantom = [b for b in run_seq(pw, url, seq_d,
                                       "D2: 复跑确认稳定") if "你帮我" in b]
        assert not any(b.strip() == "YOU: 你帮我" for b in phantom), \
            f"幻影『你帮我』复现：{phantom}"
    srv.shutdown()
    print("\n(replay done)")


if __name__ == "__main__":
    main()
