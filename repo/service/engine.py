import math
from collections import deque

import numpy as np
import torch


class TurnTakingEngine:
    # Whole-turn correction audio (two-pass ASR, 2026-11-05 方案A): the
    # engine is the per-session home for turn audio — the TurnModel is
    # SHARED across sessions and gets reset() mid-turn from several call
    # sites (far-field / backchannel / history_len guards), so its buffers
    # cannot be trusted to hold a full turn. Accumulation starts at the
    # first energetic chunk (a ~1.6s pre-roll ring keeps the word onset
    # that triggered it) so inter-turn idle silence never bloats the
    # correction input, and is capped from the tail to bound memory/latency.
    TURN_ONSET_RMS = 0.01   # below this a chunk is not speech onset
    TURN_MAX_SEC = 35.0     # tail cap on accumulated turn audio

    def __init__(
        self,
        model,
    ):
        """
        model: TurnModel (model.py)
        """
        self.model = model
        self.device = model.device
        self.contexts = None
        # per-session wait-path override (chunks); None -> model config
        # default. The model instance is SHARED across sessions, so this
        # knob lives on the (per-session) engine and travels with each
        # process() call.
        self.max_wait_num = None
        # turn-audio accumulation (see class comment)
        self._sr = model.sampling_rate
        self._preroll = deque(maxlen=10)  # ~1.6s at 160ms chunks
        self.turn_audio = []

    def _accumulate_turn_audio(self, audio: np.ndarray):
        if self.turn_audio:
            self.turn_audio.append(audio)
            # tail cap: drop the oldest chunks beyond TURN_MAX_SEC
            total = sum(len(a) for a in self.turn_audio)
            while total > self.TURN_MAX_SEC * self._sr and len(self.turn_audio) > 1:
                total -= len(self.turn_audio[0])
                self.turn_audio.pop(0)
        else:
            self._preroll.append(audio)
            if float(np.sqrt(np.mean(np.square(audio)))) >= self.TURN_ONSET_RMS:
                # speech onset: stitch the pre-roll (holds the word onset
                # itself — this chunk triggered, the ring holds what came
                # just before) and start accumulating
                self.turn_audio = list(self._preroll)

    def _take_turn_audio(self):
        """Hand out and clear the accumulated turn audio (None if the turn
        never had an energetic chunk — caller falls back to live text)."""
        chunks, self.turn_audio = self.turn_audio, []
        if not chunks:
            return None
        return np.concatenate(chunks)

    def process(self, audio: np.ndarray):
        """
        audio: np.float32, shape [N]
        return:
            None | state_token(str)
        """
        if self.contexts is None:
            self.model.reset()
        else:
            self.model.restore_runtime(self.contexts)

        self._accumulate_turn_audio(audio)
        result = self.model.process(audio, max_wait_num=self.max_wait_num)
        self.contexts = self.model.snapshot_runtime()

        # two-pass ASR (方案A): hand the whole-turn audio to the server
        # with the commit — it pops the key before json.dumps and runs
        # correct_turn_text on a worker thread, pushing a turn_final.
        if result.get("commit"):
            result["corr_audio"] = self._take_turn_audio()

        return result
