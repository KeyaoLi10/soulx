"""Discrimination + latency test for the pvad microservice (:8086).

The app gates turns on the /score ECAPA cosine (identity), with the pvad
ONNX prob (/feed) as live activity telemetry. This test asserts both:

  - /score: enroll with eotbench_zh/zh__2284 (mid 8 s), then same-speaker
    audio scores high (~0.6-0.8) and a different speaker (xyz podcast
    host) scores low (~0.1-0.3). Measured across clips: same 0.59-0.80,
    different 0.06-0.44 — threshold 0.5 separates.
  - /feed: prob stream flows, enrolled flag tracks the session, latency
    stays well under the 160 ms chunk cadence.

Usage: venv/bin/python -I scripts/pvad_service_test.py
"""
import base64
import time
import wave
from pathlib import Path

import numpy as np
import requests

BASE = "http://127.0.0.1:8086"
SR = 16000
BATCH = 2560  # 160 ms, the app's feed cadence
DATA = Path("/home/research/krisl/soulx/data/eval_inputs")

failures = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        failures.append(name)


def load(path):
    with wave.open(str(path), "rb") as w:
        return (
            np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
            .astype(np.float32)
            / 32768.0
        )


def b64(wav):
    return base64.b64encode(wav.astype("<f4").tobytes()).decode()


def post(path, payload, timeout=30):
    t0 = time.time()
    r = requests.post(f"{BASE}{path}", json=payload, timeout=timeout)
    dt = (time.time() - t0) * 1000
    r.raise_for_status()
    return r.json(), dt


def feed_clip(session, wav, max_batches=None):
    """Stream audio through /feed in 160 ms batches; returns (probs, latencies_ms)."""
    probs, lats = [], []
    n = len(wav) // BATCH
    if max_batches:
        n = min(n, max_batches)
    for i in range(n):
        chunk = wav[i * BATCH:(i + 1) * BATCH]
        out, dt = post(
            "/feed",
            {"session_id": session, "audio": b64(chunk), "sample_rate": SR},
            timeout=10,
        )
        if out["prob"] is not None:
            probs.append(out["prob"])
        lats.append(dt)
    return probs, lats


h = requests.get(f"{BASE}/health", timeout=5).json()
check("service healthy", h.get("loaded") is True, str(h))

# --- enroll: speaker A = zh__2284, 8-16 s ---
A = load(DATA / "eotbench_zh/zh__2284/input.wav")
out, _ = post(
    "/enroll",
    {"session_id": "test-main", "audio": b64(A[8 * SR:16 * SR]), "sample_rate": SR},
    timeout=60,
)
check("enroll ok", out.get("ok") is True, str(out))

# --- identity discrimination via /score (the app's gate signal) ---
same_out, same_ms = post(
    "/score",
    {"session_id": "test-main", "audio": b64(A[20 * SR:30 * SR]), "sample_rate": SR},
)
other = load(DATA / "xyz/paoti_miaowu-t0069/input.wav")
diff_out, diff_ms = post(
    "/score",
    {"session_id": "test-main", "audio": b64(other[5 * SR:15 * SR]), "sample_rate": SR},
)
cos_same, cos_diff = same_out["cos"], diff_out["cos"]
print(f"[INFO] same-speaker cos={cos_same:.3f} ({same_ms:.0f}ms), "
      f"other-speaker cos={cos_diff:.3f} ({diff_ms:.0f}ms)")

check(f"same-speaker cos > 0.5 ({cos_same:.3f})", cos_same > 0.5)
check(f"other-speaker cos < 0.5 ({cos_diff:.3f})", cos_diff < 0.5)
margin = cos_same - cos_diff
check(f"margin > 0.15 (got {margin:.3f})", margin > 0.15)
check(f"/score latency < 300ms (got {same_ms:.0f}ms)", same_ms < 300)

# unknown session => cos None (fail-open signal)
out, _ = post(
    "/score",
    {"session_id": "unknown-sid", "audio": b64(A[:3 * SR]), "sample_rate": SR},
)
check("unknown session scores None", out.get("cos") is None, str(out))

# --- /feed telemetry path: probs flow, enrolled flag, latency ---
probs, lats = feed_clip("test-main", A[20 * SR:24 * SR])
check("feed returns probs", len(probs) > 0)
check("feed reports enrolled", all(
    requests.post(
        f"{BASE}/feed",
        json={"session_id": "test-main", "audio": b64(A[24 * SR:24 * SR + 2560]),
              "sample_rate": SR},
        timeout=10,
    ).json()["enrolled"] for _ in range(1)
))
lat_p95 = float(np.percentile(lats, 95)) if lats else 999
check(f"/feed latency p95 < 50ms (got {lat_p95:.1f}ms)", lat_p95 < 50)

# --- short audio rejected ---
out, _ = post(
    "/enroll",
    {"session_id": "test-short", "audio": b64(A[:SR]), "sample_rate": SR},
)
check("enroll rejects <2s audio", out.get("ok") is False)

# --- reset drops session state ---
requests.post(f"{BASE}/reset", json={"session_id": "test-main"}, timeout=5)
out, _ = post(
    "/score",
    {"session_id": "test-main", "audio": b64(A[20 * SR:30 * SR]), "sample_rate": SR},
)
check("score None after reset (unenrolled again)", out.get("cos") is None)
probs_after, _ = feed_clip("test-main", A[20 * SR:23 * SR])
check("session usable after reset", len(probs_after) > 0)

print(f"\n[RESULT] {'FAIL: ' + str(failures) if failures else 'PASS — pvad discriminates speakers'}")
raise SystemExit(1 if failures else 0)
