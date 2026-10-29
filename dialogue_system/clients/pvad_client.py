"""Async client for the pvad microservice (:8086) — the "main speaker?" sidecar.

Design constraint (hard): nothing here may sit on the turn-controller path.
`feed()` is a non-blocking queue put; a daemon thread batches audio to
>= 160 ms and POSTs it to the service, recording (timestamp, prob) pairs
for the live UI readout. The only blocking call is `score` (~150 ms, one
ECAPA forward), invoked exclusively from worker threads at decision
points — never from the audio receive loop. If the service is down, slow,
or the session is not enrolled, `score` returns None and the caller fails
open: the turn controller's decisions are then exactly what they were
before pvad existed.

Wire contract (services/pvad_server.py):
  POST /feed   {session_id, audio: b64 float32 LE, sample_rate} -> {prob, enrolled}
  POST /enroll {session_id, audio, sample_rate} -> {ok, enrolled, duration_s}
  POST /score  {session_id, audio, sample_rate} -> {cos, enrolled}
               cosine(enrolled speaker, audio tail) — the identity signal
               the app gates on. (The pvad ONNX prob is kept as live
               activity telemetry; measured on hard negatives it cannot
               separate speakers: diff-speaker speech-mean 0.54 vs main
               0.60, while ECAPA cosine separates 0.19-0.44 vs 0.59-0.80.)
"""

import base64
import os
import queue
import threading
import time
from collections import deque

import numpy as np
import requests

PVAD_BASE = os.environ.get("PVAD_BASE", "http://127.0.0.1:8086")
FEED_MIN_SAMPLES = 2560   # 160 ms @ 16 kHz — one POST per turn-controller chunk
PROB_STALE_S = 1.5        # older than this => data is stale (display only)


class PVADSession:
    """One browser session's pvad sidecar. All public methods are cheap and
    non-blocking except `enroll` (deliberately blocking, called from worker
    threads only)."""

    def __init__(self, session_id: str, base_url: str = PVAD_BASE):
        self.session_id = session_id
        self._base = base_url
        self._http = requests.Session()
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._probs = deque(maxlen=64)   # ~10 s at one prob per 160 ms
        self._lock = threading.Lock()
        self._enrolled = False
        self._alive = True
        self._backoff = 0.5              # grows on feed errors, caps at 5 s
        self._thread = threading.Thread(
            target=self._feeder, name=f"pvad-{session_id[:8]}", daemon=True
        )
        self._thread.start()

    # ---- producer side (turn-controller path; must never block) ----

    def feed(self, wav: np.ndarray, sample_rate: int = 16000) -> None:
        """Queue a copy of a mic chunk for the sidecar. Non-blocking."""
        if not self._alive:
            return
        self._queue.put((wav.copy(), sample_rate))

    # ---- decision-time reads ----

    def score(self, wav: np.ndarray, sample_rate: int = 16000) -> float | None:
        """Blocking identity check (worker threads only): cosine similarity
        between the enrolled speaker and the tail of `wav`. None when
        unusable (not enrolled / service down / too little audio) — callers
        treat None as fail-open. ~120-170 ms on localhost."""
        try:
            r = self._http.post(
                f"{self._base}/score",
                json={
                    "session_id": self.session_id,
                    "audio": base64.b64encode(
                        wav.astype("<f4").tobytes()
                    ).decode(),
                    "sample_rate": sample_rate,
                },
                timeout=10,
            )
            out = r.json() if r.ok else {}
            with self._lock:
                self._enrolled = bool(out.get("enrolled"))
            cos = out.get("cos")
            return None if cos is None else float(cos)
        except (requests.RequestException, ValueError) as exc:
            print(f"[pvad] score failed: {exc}")
            return None

    @property
    def enrolled(self) -> bool:
        with self._lock:
            return self._enrolled

    def latest_prob(self) -> float | None:
        with self._lock:
            return self._probs[-1][1] if self._probs else None

    # ---- enrollment (worker threads only; blocking HTTP) ----

    def enroll(self, wav: np.ndarray, sample_rate: int = 16000) -> bool:
        """Personalize this session with >= 2 s of the main speaker's audio
        (the service keeps only the last 5 s for the ECAPA embedding)."""
        try:
            r = self._http.post(
                f"{self._base}/enroll",
                json={
                    "session_id": self.session_id,
                    "audio": base64.b64encode(
                        wav.astype("<f4").tobytes()
                    ).decode(),
                    "sample_rate": sample_rate,
                },
                timeout=30,
            )
            ok = r.ok and r.json().get("ok") is True
        except requests.RequestException as exc:
            print(f"[pvad] enroll failed: {exc}")
            return False
        if ok:
            with self._lock:
                self._enrolled = True
        return ok

    def stop(self) -> None:
        self._alive = False
        self._queue.put((np.zeros(0, dtype=np.float32), 16000))  # wake feeder

    # ---- feeder thread ----

    def _feeder(self) -> None:
        pending = np.zeros(0, dtype=np.float32)
        pending_sr = 16000
        while self._alive:
            try:
                wav, sr = self._queue.get(timeout=5.0)
            except queue.Empty:
                continue
            if not self._alive:
                break
            if wav.size:
                if pending.size and sr != pending_sr:
                    pending = np.zeros(0, dtype=np.float32)  # rare; drop mix
                pending = np.concatenate((pending, wav))
                pending_sr = sr
            if pending.size < FEED_MIN_SAMPLES:
                continue

            payload = pending
            pending = np.zeros(0, dtype=np.float32)
            try:
                r = self._http.post(
                    f"{self._base}/feed",
                    json={
                        "session_id": self.session_id,
                        "audio": base64.b64encode(
                            payload.astype("<f4").tobytes()
                        ).decode(),
                        "sample_rate": pending_sr,
                    },
                    timeout=10,
                )
                out = r.json() if r.ok else {}
                prob = out.get("prob")
                with self._lock:
                    self._enrolled = bool(out.get("enrolled"))
                    if prob is not None:
                        self._probs.append((time.time(), float(prob)))
                self._backoff = 0.5
            except (requests.RequestException, ValueError) as exc:
                # Service down / bad payload: back off, keep draining the
                # queue so it cannot grow unboundedly, stay fail-open.
                self._backoff = min(self._backoff * 2, 5.0)
                if self._backoff >= 2.0:
                    print(f"[pvad] feed error ({exc}); backing off")
                time.sleep(self._backoff)
                # Drop what accumulated during the sleep to avoid a burst.
                while True:
                    try:
                        wav, sr = self._queue.get_nowait()
                        if wav.size:
                            pending = wav
                            pending_sr = sr
                    except queue.Empty:
                        break
