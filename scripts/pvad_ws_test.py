"""WS-level test of the pvad main-speaker gate in the SoulX web app.

Phases (one WS connection, mimicking the browser):
  0. hostile settings: pvad_threshold=99 / pvad_mode="bogus" must be
     clamped/ignored in the config_applied echo.
  1. first turn (speaker A, unenrolled) => fail-open: turn_start fires and
     auto-enrollment kicks in (pvad_status enrolled=true).
  2. same speaker again => gate passes: turn_start, no pvad_reject(turn).
  3. different speaker (enforce) => pvad_reject {kind:"turn"} and NO
     turn_start in that window.
  4. different speaker (shadow) => pvad_reject {kind:"turn", shadow:true}
     AND turn_start (pipeline proceeds, nothing blocked).

Barge-in rejects may occur incidentally while TTS is playing; they are
counted but not asserted on.

Usage: venv/bin/python -I scripts/pvad_ws_test.py [--url ws://127.0.0.1:8090/ws]
"""
import argparse
import json
import threading
import time
import wave
from pathlib import Path

import numpy as np
import websocket

CHUNK = 2560  # 160 ms @ 16k
SR = 16000
DATA = Path("/home/research/krisl/soulx/data/eval_inputs")
SAME = DATA / "eotbench_zh/zh__2284/input.wav"          # speaker A
OTHER = DATA / "xyz/paoti_miaowu-t0069/input.wav"       # speaker B

failures = []
events = []           # (ts, event, data-dict)
events_lock = threading.Lock()
stop_flag = False


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}", flush=True)
    if not cond:
        failures.append(name)


def log(tag, text):
    print(f"{time.strftime('%H:%M:%S')} [{tag}] {text}", flush=True)


def reader(ws):
    while not stop_flag:
        try:
            raw = ws.recv()
        except Exception:
            return
        if isinstance(raw, bytes):
            continue
        try:
            msg = json.loads(raw)
        except Exception:
            continue
        ev = msg.get("event")
        data = msg.get("data", {})
        with events_lock:
            events.append((time.time(), ev, data))
        if ev == "pvad_reject":
            log("REJECT", str(data))
        elif ev == "pvad_status" and data.get("enrolled"):
            log("PVAD", "enrolled")


def load(path):
    with wave.open(str(path), "rb") as w:
        return (
            np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
            .astype(np.float32)
            / 32768.0
        )


def send_audio(ws, samples):
    payload = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    ws.send(payload, opcode=websocket.ABNF.OPCODE_BINARY)


def stream(ws, wav, label, tail_silence=4.0):
    log("STREAM", f"{label}: {len(wav)/SR:.1f}s speech + {tail_silence}s silence")
    for i in range(0, len(wav), CHUNK):
        send_audio(ws, wav[i : i + CHUNK])
        target = 0.16 * (i // CHUNK + 1)
        sent = 0.16 * (i // CHUNK)
        time.sleep(max(0.0, min(0.16, target - sent)))
    for _ in range(int(tail_silence * SR / CHUNK)):
        send_audio(ws, np.zeros(CHUNK, dtype=np.float32))
        time.sleep(0.16)


def snapshot():
    with events_lock:
        return list(events)


def wait_for(pred, timeout, what):
    t0 = time.time()
    while time.time() - t0 < timeout:
        for _, ev, data in snapshot():
            if pred(ev, data):
                return True
        time.sleep(0.3)
    log("TIMEOUT", f"waiting for {what}")
    return False


def phase_window(t_from):
    """Events strictly after t_from."""
    return [(ts, ev, d) for ts, ev, d in snapshot() if ts > t_from]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8090/ws")
    args = ap.parse_args()

    same, other = load(SAME), load(OTHER)

    ws = websocket.create_connection(args.url, timeout=5)
    ws.settimeout(300)
    ws.send(json.dumps({"event": "config_audio", "data": {"sample_rate": SR}}))
    threading.Thread(target=reader, args=(ws,), daemon=True).start()

    # ---- phase 0: hostile settings are clamped/ignored ----
    ws.send(json.dumps({
        "event": "update_config",
        "data": {"pvad_threshold": 99, "pvad_mode": "bogus"},
    }))
    got = wait_for(
        lambda ev, d: ev == "config_applied", 5, "config_applied"
    )
    applied = next(
        (d for _, ev, d in snapshot() if ev == "config_applied"), {}
    ).get("settings", {})
    check("threshold clamped to 0.9", applied.get("pvad_threshold") == 0.9,
          str(applied.get("pvad_threshold")))
    check("bogus mode ignored (stays enforce)",
          applied.get("pvad_mode") == "enforce", str(applied.get("pvad_mode")))

    # Restore the sane threshold — the clamp test must not poison the
    # remaining phases (first run of this test left it at 0.9 and every
    # later gate evaluation rejected *everyone*).
    t_restore = time.time()
    ws.send(json.dumps({
        "event": "update_config", "data": {"pvad_threshold": 0.5},
    }))
    wait_for(
        lambda ev, d: ev == "config_applied"
        and any(ts > t_restore for ts, e, _ in snapshot() if e == "config_applied"),
        5, "config restore",
    )
    applied = next(
        (d for ts, ev, d in snapshot()
         if ev == "config_applied" and ts > t_restore), {}
    ).get("settings", {})
    check("threshold restored to 0.5", applied.get("pvad_threshold") == 0.5,
          str(applied.get("pvad_threshold")))

    # ---- phase 1: first turn, unenrolled => fail-open + auto-enroll ----
    t1 = time.time()
    stream(ws, same[: 12 * SR], "phase1/speaker-A (enroll)")
    ok_turn = wait_for(
        lambda ev, d: ev == "turn_start" and True, 25, "turn_start (fail-open)"
    )
    check("turn fires while unenrolled (fail-open)", ok_turn)
    ok_enroll = wait_for(
        lambda ev, d: ev == "pvad_status" and d.get("enrolled") is True,
        25, "pvad enrolled",
    )
    check("auto-enroll after first turn", ok_enroll)

    # ---- phase 2: same speaker passes ----
    t2 = time.time()
    stream(ws, same[20 * SR : 30 * SR], "phase2/speaker-A (same)")
    ok_turn = wait_for(
        lambda ev, d: ev == "turn_start", 25, "turn_start (same speaker)"
    )
    check("same speaker passes the gate", ok_turn)
    rejects = [d for _, ev, d in phase_window(t2)
               if ev == "pvad_reject" and d.get("kind") == "turn"
               and not d.get("shadow")]
    check("no turn reject for the main speaker", not rejects, str(rejects[:1]))

    # ---- phase 3: different speaker, enforce ----
    t3 = time.time()
    stream(ws, other[5 * SR : 15 * SR], "phase3/speaker-B (enforce)")
    ok_reject = wait_for(
        lambda ev, d: ev == "pvad_reject" and d.get("kind") == "turn"
        and not d.get("shadow"),
        30, "pvad_reject(turn)",
    )
    check("different speaker rejected (enforce)", ok_reject)
    # give the reject a moment to (not) spawn a pipeline
    time.sleep(4.0)
    turns = [1 for _, ev, _ in phase_window(t3) if ev == "turn_start"]
    check("no pipeline for the rejected turn", not turns, f"{len(turns)} turn_start")

    # ---- phase 4: different speaker, shadow ----
    ws.send(json.dumps({"event": "update_config", "data": {"pvad_mode": "shadow"}}))
    wait_for(
        lambda ev, d: ev == "config_applied"
        and d.get("settings", {}).get("pvad_mode") == "shadow",
        5, "shadow mode applied",
    )
    t4 = time.time()
    stream(ws, other[20 * SR : 30 * SR], "phase4/speaker-B (shadow)")
    ok_reject = wait_for(
        lambda ev, d: ev == "pvad_reject" and d.get("kind") == "turn"
        and d.get("shadow") is True,
        30, "pvad_reject(turn, shadow)",
    )
    check("shadow reports would-reject", ok_reject)
    ok_turn = wait_for(
        lambda ev, d: ev == "turn_start", 25, "turn_start (shadow proceeds)"
    )
    check("shadow still lets the turn through", ok_turn)

    global stop_flag
    stop_flag = True
    n_bargein = sum(
        1 for _, ev, d in snapshot()
        if ev == "pvad_reject" and d.get("kind") == "bargein"
    )
    log("INFO", f"incidental barge-in rejects: {n_bargein}")
    try:
        ws.close()
    except Exception:
        pass

    print(f"\n[RESULT] {'FAIL: ' + str(failures) if failures else 'PASS — pvad gate behaves as designed'}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
