"""Blip-gate calibration + regression harness for the SoulX turn server.

Streams audio through ws://127.0.0.1:8000/turn (bypassing the web app) and
records, per 160ms chunk: the predicted state, any blip-gate rewrite
("gate": prob/rms/mistake_no_text/commit_no_text) and the pre-latch nonidle
diagnostics ("diag": {p_nonidle, rms} — the Stage 0 calibration signal).

Stages (see docs/修改日志.md 2026-11-04 and config/config.yaml):
  Stage 0 (gates off):   run --synth, read the p_nonidle/rms distributions
                         of blip vs real-speech chunks from the JSON, pick
                         nonidle_prob_gate / nonidle_rms_gate, write them
                         into config.yaml, restart the turn server.
  Stage 1 (gates on):    run --synth --check — blip bursts must produce
                         zero ungated nonidle and zero speak; the quiet
                         second utterance in the control file must still
                         commit with text.

Usage (from the soulx root, real-time paced like a live mic):
  venv/bin/python -I scripts/blip_gate_replay.py --synth [--check] [--fast]
  venv/bin/python -I scripts/blip_gate_replay.py --input some.wav [--label t0,t1,name ...]

Results land in results/blip_gate_replay/<ts>_<name>.json (one file per
input) plus a human-readable summary on stdout.
"""
import argparse
import base64
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np
import websocket

SR = 16000
CHUNK = 2560  # samples per turn-server chunk (160ms)
WS_URL = "ws://127.0.0.1:8000/turn"
SPEECH_WAV = Path(
    "/home/research/krisl/soulx/data/eval_inputs/eotbench_zh/zh__2284/input.wav"
)
SPEECH_OFFSET_S = 21.0  # same anchor as scripts/playback_echo_test.py
SPEECH_LEN_S = 2.0
OUT_DIR = Path("/home/research/krisl/soulx/results/blip_gate_replay")


def load_wav(path, offset_s=0.0, dur_s=None):
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == SR and w.getnchannels() == 1, (
            f"{path}: need 16kHz mono (got {w.getframerate()}Hz "
            f"{w.getnchannels()}ch)"
        )
        wav = (
            np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
            .astype(np.float32)
            / 32768.0
        )
    start = int(offset_s * SR)
    end = len(wav) if dur_s is None else min(len(wav), start + int(dur_s * SR))
    return wav[start:end]


def band_limited_noise(dur_s, lo_hz, hi_hz, rms):
    """White noise band-limited to [lo_hz, hi_hz], scaled to target RMS."""
    n = int(dur_s * SR)
    x = np.random.randn(n).astype(np.float32)
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    spec[(freqs < lo_hz) | (freqs > hi_hz)] = 0
    y = np.fft.irfft(spec, n).astype(np.float32)
    cur = np.sqrt(np.mean(y**2))
    return y * (rms / cur) if cur > 0 else y


def synth_corpus():
    """Build the calibration corpus in memory.

    Blip files mimic the incident shape: a real committed utterance, then a
    short energetic-but-nonlinguistic burst (breath/laugh residue), then
    silence. The control file checks the no-false-kill floor: a real second
    utterance at half gain must still latch and commit.

    Returns [(name, audio, regions)] where regions = [(t0, t1, label), ...]
    mark the burst / quiet-speech spans used for assertions and for labeling
    Stage-0 distributions.
    """
    speech = load_wav(SPEECH_WAV, SPEECH_OFFSET_S, SPEECH_LEN_S)
    silence = lambda s: np.zeros(int(s * SR), dtype=np.float32)

    corpus = []
    # 12 burst variants: RMS x duration. 300-1000Hz covers the band where
    # breath/laugh energy concentrates while staying clear of the murmur
    # fundamental (~100-200Hz) so a too-aggressive rms gate shows up as
    # burst leakage rather than hiding behind speech-band overlap.
    # The 3.0s gap leaves room for the lead utterance's own commit: with
    # mistake_no_text_reset on, quiet-lead commits move from the old
    # mistake-forced "complete" (mid-sentence split, ~0.4s after speech
    # end) to a whole-utterance "wait" commit (~1-2.2s after speech end).
    for rms in (0.005, 0.01, 0.02):
        for dur_ms in (160, 320, 480, 640):
            burst = band_limited_noise(dur_ms / 1000.0, 300.0, 1000.0, rms)
            lead, gap, tail = 2.0, 3.0, 2.0
            audio = np.concatenate([speech, silence(gap), burst, silence(tail)])
            t0 = lead + gap
            regions = [(t0, t0 + dur_ms / 1000.0, "blip")]
            corpus.append((f"blip_rms{rms}_dur{dur_ms}", audio, regions))

    # control: speech -> silence -> quiet real speech (0.5x) -> silence
    quiet = (speech * 0.5).astype(np.float32)
    audio = np.concatenate([speech, silence(1.0), quiet, silence(2.0)])
    regions = [(0.0, 2.0, "speech"), (3.0, 5.0, "quiet_speech")]
    corpus.append(("control_quiet_second_utterance", audio, regions))

    # full-utterance baseline file + 2s trailing silence: a live mic stream
    # keeps flowing silence after speech ends, so the end-of-speech wait
    # commit can still land — the raw 10s cut would truncate it.
    corpus.append(("full_utterance", np.concatenate([load_wav(SPEECH_WAV, 0.0, 10.0), silence(2.0)]), []))
    return corpus


def region_label(regions, t):
    for t0, t1, label in regions:
        if t0 <= t < t1:
            return label
    return None


def stream(sid, audio, fast=False):
    """Stream audio through the turn server at 1x pace; return chunk records."""
    ws = websocket.create_connection(WS_URL, timeout=10.0)
    records = []
    t0 = time.time()
    for i in range(0, len(audio), CHUNK):
        payload = {
            "type": "audio",
            "session_id": sid,
            "audio": base64.b64encode(
                audio[i : i + CHUNK].astype(np.float32).tobytes()
            ).decode(),
        }
        ws.send(json.dumps(payload))
        resp = json.loads(ws.recv())["state"]
        rec = {
            "i": i // CHUNK,
            "t": round((i // CHUNK) * CHUNK / SR, 3),
            "state": resp["state"],
        }
        if resp.get("gate"):
            rec["gate"] = resp["gate"]
        if resp.get("diag"):
            rec["diag"] = resp["diag"]
        if resp.get("asr_segment"):
            rec["asr"] = resp["asr_segment"]
        if resp["state"] == "speak":
            rec["text"] = resp.get("text", "")
            rec["trigger"] = resp.get("trigger", "")
        records.append(rec)
        if not fast:
            target = t0 + 0.16 * (i // CHUNK + 1)
            time.sleep(max(0.0, target - time.time()))
    ws.close()
    return records


def summarize(records, regions):
    counts = {}
    for r in records:
        counts[r["state"]] = counts.get(r["state"], 0) + 1
    gated = [r for r in records if r.get("gate")]
    speaks = [r for r in records if r["state"] == "speak"]
    # Stage-0 calibration table: pre-latch nonidle diagnostics by region
    diag = []
    for r in records:
        if r.get("diag"):
            diag.append(
                {
                    "t": r["t"],
                    "label": region_label(regions, r["t"]),
                    **r["diag"],
                }
            )
    return {"counts": counts, "gated": len(gated), "speaks": speaks, "diag": diag}


def check(name, records, regions):
    """Stage-1 assertions. Returns a list of failure strings."""
    failures = []
    if name.startswith("blip_"):
        # the blip region + 0.8s of trailing context: no ungated nonidle,
        # no speak. A gated chunk arrives as idle with r["gate"] set; a
        # burst the model naturally calls idle needs no gate at all — both
        # are passes, the outcome (no phantom turn) is what we assert.
        span = next((t1 - t0 for t0, t1, _ in regions), 0.0)
        t0 = regions[0][0]
        window = [r for r in records if t0 <= r["t"] <= t0 + span + 0.8]
        bad = [
            r
            for r in window
            if r["state"] == "nonidle" or (r["state"] == "speak" and r["t"] > t0)
        ]
        if bad:
            failures.append(
                f"{name}: {len(bad)} ungated nonidle/speak in blip window "
                f"(first at t={bad[0]['t']}s state={bad[0]['state']})"
            )
    elif name.startswith("control_"):
        # the quiet second utterance (3.0-5.0s) must still commit with text
        speaks = [r for r in records if r["state"] == "speak" and r["t"] >= 3.0]
        if not any((r.get("text") or "").strip() for r in speaks):
            failures.append(
                f"{name}: quiet second utterance produced no text-bearing "
                f"commit (speaks: {[(r['t'], r.get('text')) for r in speaks]})"
            )
    elif name.startswith("full_"):
        # requirement-level assertion, not mechanism: the exact commit count
        # and split points wander with any narrative perturbation (the gates
        # feed different text/state tokens to the LLM). What must hold:
        # every sentence's content lands in some commit, and no commit is
        # empty. The three sentences of this file, by keyword:
        keywords = ["金诗涵", "预定", "抽血"]
        speaks = [
            r for r in records if r["state"] == "speak" and (r.get("text") or "").strip()
        ]
        joined = "".join(r.get("text", "") for r in speaks)
        missing = [k for k in keywords if k not in joined]
        if missing or not speaks:
            failures.append(
                f"{name}: speech content lost — missing {missing}, got "
                f"{[(r['t'], r.get('text')) for r in speaks]}"
            )
    return failures


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--synth", action="store_true", help="built-in corpus")
    ap.add_argument("--input", action="append", default=[], help="wav to stream")
    ap.add_argument(
        "--label",
        action="append",
        default=[],
        metavar="T0,T1,NAME",
        help="mark [T0,T1)s as a named region (assertion/calibration label)",
    )
    ap.add_argument("--check", action="store_true", help="run Stage-1 assertions")
    ap.add_argument("--fast", action="store_true", help="skip real-time pacing")
    args = ap.parse_args()

    if not args.synth and not args.input:
        ap.error("nothing to do: pass --synth and/or --input")

    jobs = []
    if args.synth:
        jobs.extend((name, audio, regions) for name, audio, regions in synth_corpus())
    for p in args.input:
        regions = [
            (float(a), float(b), c)
            for a, b, c in (s.split(",", 2) for s in args.label)
        ]
        jobs.append((Path(p).stem, load_wav(Path(p)), regions))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    all_failures = []
    for name, audio, regions in jobs:
        sid = f"replay-{name}-{int(time.time()*1000)%10**9}"
        t0 = time.time()
        records = stream(sid, audio, fast=args.fast)
        summary = summarize(records, regions)
        out = OUT_DIR / f"{stamp}_{name}.json"
        out.write_text(
            json.dumps(
                {"name": name, "sid": sid, "regions": regions, "records": records,
                 "summary": summary},
                ensure_ascii=False,
                indent=1,
            )
        )
        speak_txt = "; ".join(
            f"t={r['t']}[{r.get('trigger')}]:{r.get('text', '')!r}"
            for r in summary["speaks"]
        )
        print(
            f"[{name}] {len(audio)/SR:.1f}s in {time.time()-t0:.1f}s "
            f"states={summary['counts']} gated={summary['gated']} "
            f"speaks: {speak_txt or '-'}"
        )
        if summary["diag"]:
            for label in ("blip", "quiet_speech", "speech", None):
                pts = [d for d in summary["diag"] if d["label"] == label]
                if pts:
                    ps = sorted(d["p_nonidle"] for d in pts)
                    rs = sorted(d["rms"] for d in pts)
                    show = lambda v: f"min={v[0]:.3f} med={v[len(v)//2]:.3f} max={v[-1]:.3f}"
                    print(
                        f"    diag[{label or 'other'}] n={len(pts)} "
                        f"p_nonidle {show(ps)} | rms {show(rs)}"
                    )
        if args.check:
            all_failures.extend(check(name, records, regions))

    if args.check:
        if all_failures:
            print("\nCHECK FAILED:")
            for f in all_failures:
                print(f"  - {f}")
            sys.exit(1)
        print("\nCHECK PASSED: blips gated, quiet speech intact")
    print(f"results -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
