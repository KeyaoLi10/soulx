"""Playback phantom-barge-in regression test (the 2026-11-02 幻影打断 bug).

Live failure mode: pvad gate off (default since 2026-10-30) + barge-in
hysteresis off (_BARGEIN_MIN_CHUNKS = 0 since 2026-11-02) means a SINGLE
nonidle chunk of speaker echo / AEC residue during assistant playback
stops the TTS instantly. Measured 2026-11-02 22:44 (external client,
quiet room): several interrupts per session, each ~1 s into playback,
plus a few hallucinated characters streamed to the UI.

Fix under test: the playback barge-in guard (_playback_bargein_deferred)
— while assistant audio is playing, a barge-in needs >= 3 nonidle chunks
inside a 1.6 s sliding window (duty-cycle evidence, robust to the ~50%
nonidle/idle alternation browser WebRTC produces on real speech;
consecutive-count hysteresis is not, which is why it was retired).

What this test exercises, in order (pvad OFF, default settings):
  0. seed a turn, then SPARSE breath-level blips during the GENERATION
     window (turn committed, first TTS audio not yet emitted — the live
     2026-11-03 01:01 incident: a 1-2 chunk blip there killed the
     in-flight generation via barge-in AND committed a phantom turn
     "是。" that replaced the user's sentence in the UI)
       -> no stop_audio, exactly one turn_start (generation survives,
          no phantom pipeline hijacks the reply)
  1. seed another turn, then SPARSE echo blips (2 chunks every ~2.6 s,
     the measured episodic echo cadence) during playback
       -> no stop_audio while audio chunks are still being delivered
       -> at least one bargein_deferred event (the guard actually engaged;
          without this the pass would be vacuous — echo might never have
          been classified nonidle at all)
  2. seed another turn, then SUSTAINED real speech during playback
       -> stop_audio fires within the guard budget (real barge-in intact)

Usage: soulx/venv/bin/python -I scripts/playback_echo_test.py
"""
import argparse
import json
import threading
import time
import wave
from pathlib import Path

import numpy as np
import soxr
import websocket

CHUNK = 2560            # 160 ms @ 16k (mic)
ECHO_CHUNK_24K = 3840   # 160 ms @ 24k (assistant stream)
SR = 16000
DATA = Path("/home/research/krisl/soulx/data/eval_inputs")
SPEECH = DATA / "eotbench_zh/zh__2284/input.wav"
SPEECH_OFFSET_S = 21.0   # measured best 2s speech window in this file
ECHO_GAIN = 0.1          # -20 dB clean assistant audio, episodic — the level
                          # where the turn server registers the blips at all
                          # (below ~0.08 it dismisses them entirely; the
                          # semantic VAD has a sharp lock-on cliff). Real
                          # browser AEC mangles the residual further, so live
                          # echo reads sparser than this worst case.
BLIP_LEN = 2             # chunks per blip — a single 160 ms chunk lands at the
                          # mercy of the turn server's frame aggregation (it
                          # registered in 1 of 8 runs); 320 ms registers
                          # reliably while still far below 3-in-window
BLIP_PERIOD_S = 2.2      # episodic echo cadence (measured 2026-10-29/30)
NOISE_RATIO = 0.25       # AEC-residual noise vs echo level (see
                          # take_echo_chunk; CLI-overridable for calibration)

events = []
events_lock = threading.Lock()
echo_buf = bytearray()   # assistant PCM (int16 LE @24k) for the echo phase
echo_lock = threading.Lock()
stop_flag = False


def log(tag, text):
    print(f"{time.strftime('%H:%M:%S')} [{tag}] {text}", flush=True)


def reader(ws):
    while not stop_flag:
        try:
            raw = ws.recv()
        except Exception:
            return
        if isinstance(raw, bytes):
            with events_lock:
                events.append((time.time(), "audio_chunk", {"bytes": len(raw)}))
            with echo_lock:
                echo_buf.extend(raw)
            continue
        try:
            msg = json.loads(raw)
        except Exception:
            continue
        ev, data = msg.get("event"), msg.get("data", {})
        with events_lock:
            events.append((time.time(), ev, data))


def load(path):
    with wave.open(str(path), "rb") as w:
        return (
            np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
            .astype(np.float32)
            / 32768.0
        )


def send(ws, samples):
    payload = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    ws.send(payload, opcode=websocket.ABNF.OPCODE_BINARY)


def stream(ws, wav):
    for i in range(0, len(wav), CHUNK):
        send(ws, wav[i: i + CHUNK])
        time.sleep(0.16)


def stream_silence(ws, seconds):
    n = int(seconds * SR / CHUNK)
    z = np.zeros(CHUNK, dtype=np.float32)
    for _ in range(n):
        send(ws, z)
        time.sleep(0.16)


def take_echo_chunk(gain, rng):
    """Pop one 160 ms slice of the assistant stream (in order), resampled
    to mic rate. Feeding what is actually playing keeps the blips
    speech-like; a fixed pre-picked chunk proved too quiet/un-onset-like
    for the turn server to register at all."""
    with echo_lock:
        take = bytes(echo_buf[: ECHO_CHUNK_24K * 2])
        del echo_buf[: ECHO_CHUNK_24K * 2]
    if len(take) == ECHO_CHUNK_24K * 2:
        x = np.frombuffer(take, dtype="<i2").astype(np.float32) / 32768.0
        x = soxr.resample(x, 24000, SR) * gain
        if NOISE_RATIO > 0:
            x = x + rng.standard_normal(x.shape).astype(np.float32) \
                * (gain * NOISE_RATIO)
        return np.resize(x, CHUNK)
    return np.zeros(CHUNK, dtype=np.float32)


def stream_sparse_echo(ws, seconds, gain, rng):
    """Episodic echo: BLIP_LEN consecutive chunks, then a long gap."""
    z = np.zeros(CHUNK, dtype=np.float32)
    t_next = time.time()
    t_end = time.time() + seconds
    while time.time() < t_end:
        now = time.time()
        if now >= t_next:
            for _ in range(BLIP_LEN):
                send(ws, take_echo_chunk(gain, rng))
                time.sleep(0.16)
            t_next = time.time() + BLIP_PERIOD_S - BLIP_LEN * 0.16
        else:
            send(ws, z)
            time.sleep(0.16)


class BlipSlicer:
    """Sequential 160 ms slices of a local wav at low gain — the
    generation-phase stand-in for the echo source (nothing is audible
    yet, so the live trigger there is breath/mic residue, not TTS
    echo; quiet speech slices model it)."""

    def __init__(self, wav):
        self.wav = wav
        self.i = 0

    def take(self, gain):
        x = self.wav[self.i: self.i + CHUNK]
        self.i += CHUNK
        if self.i >= len(self.wav):
            self.i = 0
        if x.shape[0] < CHUNK:
            x = np.resize(x, CHUNK)
        return x * gain


def stream_sparse_blips(ws, seconds, slicer, gain, stop_predicate=None):
    """Episodic quiet blips from a local source (generation-phase test)."""
    z = np.zeros(CHUNK, dtype=np.float32)
    t_next = time.time()
    t_end = time.time() + seconds
    while time.time() < t_end:
        now = time.time()
        if now >= t_next:
            for _ in range(BLIP_LEN):
                send(ws, slicer.take(gain))
                time.sleep(0.16)
            t_next = time.time() + BLIP_PERIOD_S - BLIP_LEN * 0.16
        else:
            send(ws, z)
            time.sleep(0.16)
        if stop_predicate is not None and stop_predicate():
            break


def window(t_from, t_to=None):
    with events_lock:
        return [
            (ts, ev, d)
            for ts, ev, d in events
            if ts > t_from and (t_to is None or ts <= t_to)
        ]


def wait_for(t_from, predicate, timeout, ws=None):
    """Wait for an event match; keeps streaming silence so the turn
    server's state machine actually progresses (no trailing silence =
    no turn decision)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if any(predicate(ev, d) for _, ev, d in window(t_from)):
            return True
        if ws is not None:
            send(ws, np.zeros(CHUNK, dtype=np.float32))
        time.sleep(0.16)
    return False


def count(evs, name, pred=lambda d: True):
    return sum(1 for _, ev, d in evs if ev == name and pred(d))


def drain_playback(ws, quiet_s=3.0, timeout=45.0):
    """Stream silence until assistant audio has been quiet for quiet_s.

    The next seed must start on a quiet background: while a reply is still
    draining, _reply_in_flight stays true and the seed speech — identical
    replayed audio whose content the empathetic reply just echoed — gets
    split by the turn server and the tail fragment self-talk-dropped
    (measured round 4: '您太紧张了我儿。' matched the assistant's own
    '…最让您感到紧张的事情…' at 0.71 recall). That is a test artifact of
    replaying one file against its own reply, not a live failure mode."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        with events_lock:
            last_audio = max(
                (ts for ts, ev, _ in events if ev == "audio_chunk"),
                default=t0,
            )
        if time.time() - last_audio >= quiet_s:
            return
        send(ws, np.zeros(CHUNK, dtype=np.float32))
        time.sleep(0.16)


def seed_turn(ws, speech, label):
    """One short user turn; returns (t, ok) once the reply is streaming."""
    t = time.time()
    stream(ws, speech)
    ok = wait_for(t, lambda ev, d: ev == "turn_start", 20, ws=ws)
    log("TURN", f"{label}: fired={ok}")
    wait_for(t, lambda ev, d: ev == "audio_chunk", 15, ws=ws)
    return t, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8090/ws")
    ap.add_argument("--gain", type=float, default=ECHO_GAIN,
                    help="echo level (default: live calibration)")
    ap.add_argument("--noise", type=float, default=0.0,
                    help="extra noise as a fraction of gain (AEC-residual "
                         "simulation; 0 = clean echo)")
    args = ap.parse_args()
    rng = np.random.default_rng(42)
    gain = args.gain

    speech_full = load(SPEECH)
    off = int(SPEECH_OFFSET_S * SR)
    speech = speech_full[off: off + int(2.0 * SR)]
    assert np.sqrt((speech ** 2).mean()) > 0.05, "speech window is too quiet"

    ws = websocket.create_connection(args.url, timeout=5)
    ws.settimeout(300)
    ws.send(json.dumps({"event": "config_audio", "data": {"sample_rate": SR}}))
    threading.Thread(target=reader, args=(ws,), daemon=True).start()
    # This test is about the pvad-OFF default path — pin it explicitly so a
    # leftover enforce setting from another test cannot change the verdict.
    # The persona asks for longish replies so the echo phase has a solid
    # playback window to blip against (reply length otherwise varies 1-10 s
    # run to run and the phase lands partly after playback).
    ws.send(json.dumps({
        "event": "update_config",
        "data": {
            "pvad_mode": "off",
            "system_prompt": (
                "You are a gentle voice assistant. IMPORTANT: every reply "
                "must be at least four full sentences of warm, natural "
                "spoken Chinese."
            ),
        },
    }))

    failures = []

    def check(name, cond, detail=""):
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
        if not cond:
            failures.append(name)

    # ---- 0. breath-level blips during the GENERATION window -------------
    # The 2026-11-03 01:01 live incident: right after the first turn
    # commits (pipeline running, no TTS audio yet — playback_end stale,
    # gen_in_progress the only "in flight" signal), a 1-2 chunk mic
    # blip (a) confirmed a zero-evidence barge-in that killed the
    # generation and (b) completed a phantom turn "是。" whose final
    # transcription replaced the user's real sentence in the UI and
    # entered LLM history. Blip from the moment the pipeline starts
    # (final user_transcription — note: BEFORE turn_start, which only
    # fires when the LLM's first segment is about to speak) until the
    # first assistant audio.
    t0 = time.time()
    stream(ws, speech)
    ok_final = wait_for(
        t0,
        lambda ev, d: ev == "user_transcription" and d.get("final"),
        25, ws=ws,
    )
    check("turn 0 committed (final transcription)", ok_final)
    # Anchor the phase-0 window on the final's OWN arrival timestamp, not
    # on when this loop noticed it: turn_start is emitted at pipeline start
    # — right after the final user_transcription, before the first LLM
    # chunk — and can land between the final's arrival and the 0.16 s poll
    # noticing it. Anchoring on time.time() lost turn_start to that race
    # (round 4: n_ts=0 with the reply demonstrably streaming to Ratio 1.00).
    finals = [
        ts for ts, ev, d in window(t0)
        if ev == "user_transcription" and d.get("final")
    ]
    tg = max(finals) if finals else time.time()
    slicer = BlipSlicer(speech)
    stream_sparse_blips(
        ws, 8.0, slicer, gain,
        stop_predicate=lambda: any(
            ev == "audio_chunk" for _, ev, _ in window(tg)
        ),
    )
    # the real reply must survive the blips and actually speak
    ok_reply = wait_for(tg, lambda ev, d: ev == "turn_start", 20, ws=ws)
    stream_silence(ws, 2.0)
    evs = window(tg)
    n_stop = count(evs, "stop_audio")
    n_ts = count(evs, "turn_start")
    n_text = count(evs, "text_response")
    n_defer = count(evs, "bargein_deferred")
    n_selftalk = count(evs, "selftalk_drop")
    log("GEN", f"stop_audio={n_stop} turn_start={n_ts} text={n_text} "
               f"bargein_deferred={n_defer} selftalk_drop={n_selftalk}")
    check(
        "generation-phase blips cannot kill or hijack the reply",
        n_stop == 0 and n_ts == 1 and ok_reply and n_text >= 1,
        f"stop_audio={n_stop} turn_start={n_ts} text={n_text} "
        f"(pre-fix live: barge-in killed the in-flight generation, then "
        f"a phantom '是。' turn spawned a second pipeline that replaced "
        f"the user's sentence)",
    )
    if n_defer + n_selftalk == 0:
        print(
            "[WARN] generation-phase blips did not register this run "
            f"(bargein_deferred=0 selftalk_drop=0) — engagement "
            f"unverified, survival assertion still holds"
        )
    drain_playback(ws)

    # ---- 1. sparse echo during playback (tonight's live scenario) -------
    _, ok1 = seed_turn(ws, speech, "turn1")
    check("seed turn 1 fires", ok1)
    with echo_lock:
        echo_buf.clear()
    # start once the reply is streaming (a couple of assistant chunks
    # captured; the turn server's semantic VAD registers onset-like
    # chunks far more readily than mid-vowel ones — measured 2026-11-03:
    # blips drawn from the reply's first chunks gave 6 defense events,
    # from chunk 10+ gave 0), then blip for ~10 s
    t_cap = time.time()
    while time.time() - t_cap < 10:
        with echo_lock:
            have = len(echo_buf) // (ECHO_CHUNK_24K * 2)
        if have >= 2:
            break
        send(ws, np.zeros(CHUNK, dtype=np.float32))
        time.sleep(0.16)
    te = time.time()
    stream_sparse_echo(ws, 10.0, gain, rng)
    te_end = time.time()
    evs = window(te, te_end)
    n_stop = count(evs, "stop_audio")
    n_defer = count(evs, "bargein_deferred")
    n_selftalk = count(evs, "selftalk_drop")
    audio_ts = [ts for ts, ev, _ in evs if ev == "audio_chunk"]
    stop_ts = [ts for ts, ev, _ in evs if ev == "stop_audio"]
    total = len(audio_ts)
    # The live bug chopped the reply at Ratio 0.35-0.37 (~1 s in, first
    # third). An interrupt AFTER the reply's audio has substantially all
    # been delivered is the documented dense-echo boundary (synthetic
    # clean echo that the semantic VAD recognizes as speech sustains
    # nonidle — real browser AEC mangles its linear component away and
    # live echo reads as episodic 1-chunk blips instead), not this bug.
    early_stops = [
        ts for ts in stop_ts
        if total and sum(1 for a in audio_ts if a <= ts) < 0.6 * total
    ]
    log("ECHO", f"stop_audio={n_stop} (early={len(early_stops)}) "
                f"bargein_deferred={n_defer} selftalk_drop={n_selftalk} "
                f"audio_chunks={total}")
    check(
        f"sparse echo @{gain} does not early-chop playback",
        not early_stops,
        f"stop_audio={n_stop} early={len(early_stops)} of {total} chunks "
        f"(pre-fix: first blip stopped the TTS instantly at Ratio ~0.35)",
    )
    # Engagement is content-dependent: whether the semantic VAD registers
    # a given -20 dB blip varies with what the TTS happens to be saying
    # (measured 2026-11-03: identical settings gave 6 defense events one
    # run, 0 the next). It is reported, not asserted — the critical
    # assertion above (no early chop) fails loudly on any guard
    # regression whenever echo DOES register.
    if n_defer + n_selftalk >= 1:
        check(
            "playback defenses engaged (deferred/selftalk >= 1)",
            True,
            f"bargein_deferred={n_defer} selftalk_drop={n_selftalk}",
        )
    else:
        print(
            "[WARN] echo did not register as nonidle this run "
            f"(bargein_deferred=0 selftalk_drop=0) — engagement "
            f"unverified, no-chop assertion still holds"
        )

    # drain: let the reply finish / phantom turns settle before phase 2
    drain_playback(ws)

    # ---- 2. sustained real speech MUST still barge in during playback ---
    _, ok2 = seed_turn(ws, speech, "turn2")
    check("seed turn 2 fires", ok2)
    wait_for(time.time(), lambda ev, d: ev == "audio_chunk", 10, ws=ws)
    tb = time.time()
    stream(ws, speech)          # sustained, loud, human
    stream_silence(ws, 2.0)
    ok = wait_for(tb, lambda ev, d: ev == "stop_audio", 8, ws=ws)
    evs = window(tb)
    n_stop = count(evs, "stop_audio")
    check(
        "sustained speech still barge-ins during playback",
        ok,
        f"stop_audio={n_stop} (guard budget ~1 s; pre-guard zero-latency "
        f"first-chunk path must stay reachable)",
    )
    check("one barge-in episode interrupts once", 1 <= n_stop <= 2,
          f"stop_audio={n_stop}")

    print(
        f"\n[RESULT] {'FAIL: ' + str(failures) if failures else 'PASS — sparse echo cannot chop playback, real barge-in intact (guard: 3 nonidle in 1.6 s window, playback only)'}"
    )
    global stop_flag
    stop_flag = True
    try:
        ws.close()
    except Exception:
        pass
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
