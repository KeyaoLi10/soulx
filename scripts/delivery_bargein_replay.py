# -*- coding: utf-8 -*-
"""Offline regression for the slow-brain delivery barge-in protection
(2026-11-07, sessions 9122660a / db169080).

事故（18:09:41，会话 9122660a）：交付泵刚开口播张家口天气，用户一声
"嗯，好"（~3 个 nonidle chunk / 0.3s，其中 2 个还是播报开口**前**说的）
0.15s 即把播报掐死在 ratio 0.22 —— interrupted_partial 终态，结果永久
丢失，用户随后只能问"怎么不说话了"。根修 = 慢脑结果播报（交付泵
result 语音）挂独立打断保护：播报期间 barge-in 须持续判定——窗口内
≥5 个 nonidle chunk 且首块距今 ≥1s 才放行；播报开口前积累的证据在
开口时清零。

方法：AST 抽取 app.py 的 _begin_generation / _reply_in_flight /
_playback_guard_active / _playback_bargein_deferred 与相关模块常量，
对着假 session 重放收包循环的 nonidle 计数逻辑（append + 窗口裁剪）。
函数签名变化会大声失败。

Run: /home/research/krisl/soulx/venv/bin/python -I scripts/delivery_bargein_replay.py
"""
import ast
import collections
import sys
import threading
import time
import types
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "repo" / "dialogue_system" / "app.py"
src = APP.read_text(encoding="utf-8")
tree = ast.parse(src)

WANT_FNS = ("_begin_generation", "_reply_in_flight",
            "_playback_guard_active", "_playback_bargein_deferred",
            "_delivery_hold_turn", "_delivery_bargein_thresholds")
WANT_CONSTS = ("_PLAYBACK_BARGEIN_WINDOW_S", "_PLAYBACK_BARGEIN_NONIDLE",
               "_RESULT_DELIVERY_BARGEIN_NONIDLE",
               "_RESULT_DELIVERY_BARGEIN_SPAN_S")

segments = {}
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in WANT_FNS:
        segments[node.name] = ast.get_source_segment(src, node)
    elif isinstance(node, ast.Assign) and len(node.targets) == 1 \
            and isinstance(node.targets[0], ast.Name) \
            and node.targets[0].id in WANT_CONSTS:
        segments[node.targets[0].id] = ast.get_source_segment(src, node)
missing = [n for n in WANT_FNS + WANT_CONSTS if n not in segments]
assert not missing, f"app.py missing: {missing}"

ns = {
    "time": time,
    "pvad_enabled": lambda session: False,   # 普通直连路径
    # app.py 从 pvad_gate 导入（值见 pvad_gate.py:107）
    "_PVAD_PLAYBACK_GRACE_S": 1.5,
    "DEFAULT_SETTINGS": {"pvad_mode": "off", "playback_guard": True},
}
for name in WANT_CONSTS:
    exec(compile(segments[name], str(APP), "exec"), ns)
for name in WANT_FNS:
    exec(compile(segments[name], str(APP), "exec"), ns)

begin_generation = ns["_begin_generation"]
deferred = ns["_playback_bargein_deferred"]
hold_turn = ns["_delivery_hold_turn"]
thresholds = ns["_delivery_bargein_thresholds"]

CHUNK = 0.165          # 一个上游 chunk ≈ 165ms（收包循环实测节拍）


class _Log:
    def info(self, *a, **k):
        pass
    warning = error = debug = info


ns["logger"] = _Log()  # 抽取函数体内未用 logger，留空以防未来引用


def mk_session(playing=True):
    s = types.SimpleNamespace()
    s.lock = threading.Lock()
    s.gen_in_progress = False
    s.gen_serial = 0
    s.delivery_speech_active = False
    s.bargein_times = collections.deque()
    s.stop_event = threading.Event()
    s.settings = {"playback_guard": True, "pvad_mode": "off"}
    # playing=True：回放进行中（playback_end 在未来）→ _reply_in_flight 真
    s.playback_end = time.time() + 30.0 if playing else time.time() - 30.0
    return s


def feed(s, n, t0, gap=CHUNK):
    """镜像收包循环：nonidle chunk 计数 + 窗口裁剪，返回最后一个时刻。"""
    now = t0
    for _ in range(n):
        s.bargein_times.append(now)
        while s.bargein_times and s.bargein_times[0] < now - 1.6:
            s.bargein_times.popleft()
        now += gap
    return now - gap


results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(("PASS" if ok else "FAIL") + f"  {name}" + (f"  [{detail}]" if detail else ""))


# ---------------------------------------------------------------------------
# 1. 事故 A 精确重放（18:09:41）：播报开口前的 2 chunk "好" + 开口后 1 chunk
#    = 旧规则 3-in-1.6s → 0.15s 掐断。新规则：开口清零 + 持续判定 → 全程
#    deferred，播报活下来。
s = mk_session()
t0 = time.time()
feed(s, 2, t0)                       # 用户开口前的那声"好"（2 chunk）
begin_generation(s)                  # 交付播报 begin（清保护标志）
with s.lock:
    s.delivery_speech_active = True
    s.bargein_times.clear()          # 开口前证据清零（_speak 同款动作）
now = feed(s, 3, t0 + 0.2)           # "嗯，好"在播报开口后的 3 个 chunk
check("1 事故A重放：播报开口后 3 chunk 短应答不再掐断播报",
      deferred(s, now) is True,
      f"times={len(s.bargein_times)}")

# 2. 同样的 3 chunk 在普通（非慢脑）播报下仍按旧阈值放行——普通语音
#    打断语义不受影响
s2 = mk_session()
t0 = time.time()
now = feed(s2, 3, t0)
check("2 普通播报：3-in-1.6s 阈值原样（真打断不迟滞）",
      deferred(s2, now) is False)

# 3. 真人持续插话 ~1.2s（6 chunk，首块距今 ≥1s）→ 保护放行
s3 = mk_session()
begin_generation(s3)
with s3.lock:
    s3.delivery_speech_active = True
    s3.bargein_times.clear()
now = feed(s3, 6, time.time(), gap=0.22)
check("3 慢脑播报：持续 1.1s 的 6 chunk 真打断放行",
      deferred(s3, now) is False,
      f"span={now - s3.bargein_times[0]:.2f}s")

# 4. 5 chunk 但 span 只有 0.66s（急促连发）→ 仍 deferred（时长不够）
s4 = mk_session()
begin_generation(s4)
with s4.lock:
    s4.delivery_speech_active = True
    s4.bargein_times.clear()
now = feed(s4, 5, time.time(), gap=CHUNK)
check("4 慢脑播报：5 chunk 但 span<1s 仍护住",
      deferred(s4, now) is True,
      f"span={now - s4.bargein_times[0]:.2f}s")

# 5. 4 chunk 拉满 1.05s span → chunk 数不够仍 deferred；补第 5 个放行
s5 = mk_session()
begin_generation(s5)
with s5.lock:
    s5.delivery_speech_active = True
    s5.bargein_times.clear()
now = feed(s5, 4, time.time(), gap=0.35)
check("5 慢脑播报：span 够但 chunk<5 仍护住",
      deferred(s5, now) is True)
now = feed(s5, 1, now + 0.35)
check("5b 补满第 5 个 chunk 且 span≥1s → 放行",
      deferred(s5, now) is False)

# 6. 保护标志生命周期：话权空闲（播报早结束）时 deferred 恒 False 且
#    迟到标志被懒清
s6 = mk_session(playing=False)
s6.delivery_speech_active = True
check("6 播报结束后：不再 deferred 且保护标志被清",
      deferred(s6) is False and s6.delivery_speech_active is False)

# 7. 新一代语音上口清保护：上一段播报的保护不得罩到新话轮的语音上；
#    但吸收轮（ack commit 后无人说话，_begin_generation 不清保护）不得
#    中断仍在播放的播报保护
s7 = mk_session()
begin_generation(s7)
s7.delivery_speech_active = True     # 交付播报进行中
begin_generation(s7)                 # 吸收轮 commit（无语音跟上）
check("7 absorbed-ack commit does NOT strip delivery protection",
      s7.delivery_speech_active is True)
with s7.lock:                        # 新话轮真正上口（_speak 非 delivery
    s7.delivery_speech_active = False  # 入口的清位动作）
now = feed(s7, 3, time.time())
check("7b 新话轮语音走普通 3-chunk 阈值（无残留保护）",
      deferred(s7, now) is False)

# 8. 保护独立于 playback_guard 开关（关掉回声防御不关结果播报保护）
s8 = mk_session()
s8.settings["playback_guard"] = False
begin_generation(s8)
with s8.lock:
    s8.delivery_speech_active = True
    s8.bargein_times.clear()
now = feed(s8, 3, time.time())
check("8 playback_guard 关闭：慢脑播报保护仍然生效",
      deferred(s8, now) is True)

# ---------------------------------------------------------------------------
# 晚十一：话轮 commit 路径补漏（fd106958 会话 22:05:01/22:07:41 两起
# 同形事故——chunk 级迟滞全 suppress 了"嗯，好。"，但 turn commit 的
# pipeline_worker 无条件 interrupt 把播报掐到 Ratio 0.38/0.36）。
# _delivery_hold_turn 在 receive loop 清 bargein_times 之前判定。

# 9. 事故重放：播报中 3 chunk "嗯，好的" 短应答 turn commit → hold=True
#    （commit 不掐播报，等 _speak 真上口再替换）
s9 = mk_session()
begin_generation(s9)
with s9.lock:
    s9.delivery_speech_active = True
    s9.bargein_times.clear()
now9 = feed(s9, 3, time.time())
check("9 晚十一事故重放：3 chunk 短应答的 turn commit 不再掐播报",
      hold_turn(s9, now9) is True,
      f"times={len(s9.bargein_times)}")

# 10. 播报中持续插话 ~1.1s（7 chunk，≥1s span）的 turn commit → hold=False
#     （"你先别说了"级真打断，commit 走正常 interrupt 路径）
s10 = mk_session()
begin_generation(s10)
with s10.lock:
    s10.delivery_speech_active = True
    s10.bargein_times.clear()
now10 = feed(s10, 7, time.time(), gap=0.17)
check("10 播报中 7 chunk/1.0s 真插话的 turn commit 照常打断",
      hold_turn(s10, now10) is False)

# 11. barge-in 已 confirm（stop_event 置位）后的尾随 turn → hold=False：
#     播报已死，不能把已置位的 stop_event 护在原地吞掉新话轮
s11 = mk_session()
begin_generation(s11)
with s11.lock:
    s11.delivery_speech_active = True
    s11.bargein_times.clear()
now11 = feed(s11, 3, time.time())
s11.stop_event.set()
check("11 已 confirm 的真打断后：尾随 turn 不再 hold",
      hold_turn(s11, now11) is False)

# 12. 非播报期（flag off）→ hold=False（普通话轮 commit 行为不变）
s12 = mk_session()
feed(s12, 3, time.time())
check("12 非播报期 turn commit 行为不变",
      hold_turn(s12) is False)

# 13. 阈值前端可调：调松（2 chunk / 0 span）后 3 chunk 短应答也放行
s13 = mk_session()
s13.settings["delivery_bargein_chunks"] = 2
s13.settings["delivery_bargein_span_ms"] = 0
begin_generation(s13)
with s13.lock:
    s13.delivery_speech_active = True
    s13.bargein_times.clear()
now13 = feed(s13, 3, time.time())
check("13 前端调松阈值（2/0ms）：3 chunk 短应答可打断",
      hold_turn(s13, now13) is False and deferred(s13, now13) is False)

# 14. 阈值兜底：无设置的会话回落模块常量（5 chunk / 1.0s）
s14 = types.SimpleNamespace(settings={})
check("14 无设置会话回落常量阈值 5/1000ms",
      thresholds(s14) == (5, 1.0),
      f"got={thresholds(s14)}")

print()
if all(results):
    print(f"ALL {len(results)} PASS")
else:
    print(f"{results.count(False)}/{len(results)} FAIL")
    raise SystemExit(1)
