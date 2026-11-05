import os, sys
import threading
import torch
import random
import numpy as np
import math
import time
import re
from omegaconf import OmegaConf
import pytorch_lightning as pl

from utils.MyTn.textnorm import zh_norm, zh_remove_punc
from utils.text_utils import split_cn_en, check_en, get_lcs_substrings
from utils.backchannel_utils import check_backchannel, remove_leading_backchannel

from config.config import RunConfig
from model.model import State_Prediction_Model
from transformers import WhisperFeatureExtractor


def _short_state(name):
    """Token name for the always-on STATE log lines:
    <|user_incomplete|> -> incomplete, <|assistant_interrupt|> ->
    assistant_interrupt (prefix kept: user_backchannel and
    assistant_backchannel must not collide). None/empty -> start (first
    chunk after a commit/reset — reset() wipes past_state)."""
    if not name:
        return "start"
    return name.replace("<|user_", "").replace("<|", "").replace("|>", "")


class TurnModel:
    def __init__(
        self,
        config_path=os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../config/config.yaml")
        ),
    ):
        self._init_load_model(config_path)

        self._init_hyperparameters()

        self._init_prompt_embeds()

        self.reset()

    def _init_load_model(self, config_path):
        cfg = OmegaConf.load(config_path)
        default_config = RunConfig()
        config = OmegaConf.merge(default_config, cfg)

        pl.seed_everything(config.infer_config.seed)
        torch.manual_seed(config.infer_config.seed)
        np.random.seed(config.infer_config.seed)
        random.seed(config.infer_config.seed)

        model = State_Prediction_Model(config)

        model.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            config.model_config.glm_tokenizer_path
        )

        model.eval().to(config.infer_config.device)
        self.model = model
        self.config = config
        self.device = config.infer_config.device

        if hasattr(self.model.llm.model, "embed_tokens"):
            self.embed_tokens_func = self.model.llm.model.embed_tokens
        elif hasattr(self.model.llm.model.model, "embed_tokens"):
            self.embed_tokens_func = self.model.llm.model.model.embed_tokens
        else:
            self.embed_tokens_func = self.model.llm.model.model.model.embed_tokens

        # if self.config.model_config.enable_cascade_asr:
        from model.asr import ParaformerASR, SensevoiceASR

        if config.infer_config.asr.model_name == "paraformer":
            self.cascade_asr = ParaformerASR()
        else:
            self.cascade_asr = SensevoiceASR(
                language=config.infer_config.asr.get("language", "auto")
            )
        # Serializes every cascade-ASR call: the modelscope pipeline is not
        # thread-safe, and commit-time whole-turn corrections
        # (correct_turn_text, called from a server worker thread) now run
        # beside the websocket loop's own recognize calls.
        self.asr_lock = threading.Lock()

        print("[DuplexServer] Model loaded successfully.")

    def _init_hyperparameters(self):
        self.sampling_rate = self.config.infer_config.input.sample_rate
        self.device = self.config.infer_config.device
        self.chunk_token_len_small = (
            self.config.infer_config.input.chunk_token_len_small
        )
        self.developer_mode = self.config.infer_config.developer_mode

        # pre-latch gate knobs (rationale in config/config.py InferConfig)
        self.nonidle_prob_gate = float(self.config.infer_config.nonidle_prob_gate)
        self.nonidle_rms_gate = float(self.config.infer_config.nonidle_rms_gate)
        self.nonidle_text_bypass = bool(
            self.config.infer_config.nonidle_text_bypass
        )
        self.mistake_no_text_reset = bool(
            self.config.infer_config.mistake_no_text_reset
        )
        # D's own streak threshold, deliberately > max_mistake_num (rationale
        # in config/config.py InferConfig)
        self.mistake_no_text_num = int(
            self.config.infer_config.mistake_no_text_num
        )
        # set when an escalated mistake_no_text distrust demands a reset();
        # consumed by state_predict right after infer() returns (reset must
        # not happen mid-infer, see _state_predict)
        self._pending_reset = False
        self.commit_no_text_reset = bool(
            self.config.infer_config.commit_no_text_reset
        )
        self.asr_seed_trim = bool(self.config.infer_config.asr_seed_trim)
        # synthetic seed length of buffer_for_asr (re-seeded on every
        # reset()/clear_turn(), never truncated — see _trim_seed)
        self.asr_seed_len = int(1.6 * self.sampling_rate)

        # --- turn-boundary energy guards (2026-11-05 计数指令事故) ---
        # wait_rms_veto: an idle-CLASSIFIED chunk only counts toward the
        # wait-path silence confirmation if its RMS is also below
        # nonidle_rms_gate. Without this the 5-chunk streak can complete
        # on a real-speech onset the model briefly misclassified as idle
        # (2026-11-04 counting incident: the streak-closing "idle" chunk
        # was the onset of 「重」 at rms 0.0388) — the wait commit then
        # resets mid-word and eats the onset for the next turn. Breaths
        # (~0.004) still count; real onsets have never been observed
        # below the gate (min 0.012). Cost at real EOTs: the 1-2 decay
        # chunks right after speech no longer count, so wait commits
        # land ~160-320 ms later.
        self.wait_rms_veto = bool(self.config.infer_config.wait_rms_veto)
        # complete_defer: a <|user_complete|> landing on an ENERGETIC chunk
        # (rms >= complete_defer_rms_gate) may be riding the onset of the
        # user's next word — hold the commit for up to
        # complete_defer_max_chunks idle-classified chunks; a nonidle or
        # incomplete classification in the meantime cancels it (the turn
        # continues). Calibration (results/blip_gate_replay, 2026-11-05):
        # real-EOT complete chunks sit at rms <= 0.045 (fluent eotbench
        # speaker) and <= 0.005 (live user), the mid-word misfire at
        # 0.086 — the gate sits between them. A real EOT above the gate
        # only costs max_chunks * 160 ms.
        self.complete_defer = bool(self.config.infer_config.complete_defer)
        self.complete_defer_rms_gate = float(
            self.config.infer_config.complete_defer_rms_gate
        )
        self.complete_defer_max_chunks = int(
            self.config.infer_config.complete_defer_max_chunks
        )
        # None = no complete is being deferred; otherwise the number of
        # idle-classified chunks seen since the deferral started
        self.pending_complete_cnt = None
        # experimental (default 0 = off): seconds of the previous turn's
        # REAL audio carried in front of the fresh cascade seed at
        # reset(), so the next turn's recognition re-hears the boundary
        # (a commit that still cuts mid-word orphans the next turn's
        # onset — the 「煮慢一点」 mishearing of 「数慢一点」). The
        # duplicated trailing words are stripped from the next commit
        # via _strip_commit_overlap (char-level, zh-oriented).
        self.asr_seed_backoff_s = float(self.config.infer_config.asr_seed_backoff_s)
        # last committed turn text (normalized punctuation-stripped form
        # lives only inside _strip_commit_overlap); cross-turn by design
        self.last_commit_text = ""

    def _init_prompt_embeds(self):
        self.input_text_tokens = torch.tensor(
            self.model.tokenizer.encode("<|task_duplex_predict|><|punctuation_off|>")
        ).to(self.device)
        self.audio_pad_token = torch.tensor(
            self.model.tokenizer.encode("<|padding|>")
        ).to(self.device)
        self.audio_bos_token = torch.tensor(
            self.model.tokenizer.encode("<|begin_of_sentence|>")
        ).to(self.device)
        self.audio_eos_token = torch.tensor(
            self.model.tokenizer.encode("<|end_of_sentence|>")
        ).to(self.device)

        self.action_speak_token = torch.tensor(
            self.model.tokenizer.encode("<|user_complete|>")
        ).to(self.device)
        self.action_wait_token = torch.tensor(
            self.model.tokenizer.encode("<|user_incomplete|>")
        ).to(self.device)
        self.non_idle_token = torch.tensor(
            self.model.tokenizer.encode("<|user_nonidle|>")
        ).to(self.device)

        self.text_embeds = self.embed_tokens_func(self.input_text_tokens)
        self.audio_pad_embeds = self.embed_tokens_func(self.audio_pad_token)
        self.audio_bos_embeds = self.embed_tokens_func(self.audio_bos_token)
        self.audio_eos_embeds = self.embed_tokens_func(self.audio_eos_token)
        self.action_speak_embeds = self.embed_tokens_func(self.action_speak_token)
        self.action_wait_embeds = self.embed_tokens_func(self.action_wait_token)
        self.non_idle_embeds = self.embed_tokens_func(self.non_idle_token)

        # The 7 state tokens, for the pre-latch probability gate
        # (_apply_prelatch_gates): argmax over the full vocab discards how
        # confident the model actually is about "user is speaking".
        mc = self.config.model_config
        self.state_token_ids = torch.tensor(
            [
                mc.user_idle_token_id,
                mc.user_nonidle_token_id,
                mc.user_backchannel_token_id,
                mc.user_complete_token_id,
                mc.user_incomplete_token_id,
                mc.assistant_backchannel_token_id,
                mc.assistant_interrupt_token_id,
            ],
            device=self.device,
        )
        self.state_token_nonidle_idx = 1

    def reset(self):
        # experimental cross-turn left context (asr_seed_backoff_s, default
        # 0 = off): carry the previous turn's final seconds of REAL audio
        # behind the fresh synthetic cascade seed, so the next turn's
        # recognition re-hears the boundary. A commit that still cut mid-
        # word orphans the next turn's onset — the 「煮慢一点」 mishearing
        # of 「数慢一点」. _strip_commit_overlap removes the duplicated
        # trailing words from the next commit's text. The synthetic seed
        # always sits at the FRONT of the buffer, so the tail is real
        # audio as long as any real audio exists after the seed. On the
        # constructor's first reset() the buffers do not exist yet —
        # nothing to carry.
        boff = (
            int(self.asr_seed_backoff_s * self.sampling_rate)
            if self.asr_seed_backoff_s > 0
            else 0
        )
        tail = None
        if boff and hasattr(self, "cascade_buffer"):
            real_len = len(self.cascade_buffer) - self.cascade_seed_len
            if real_len > 0:
                tail = self.cascade_buffer[len(self.cascade_buffer) - min(boff, real_len) :]
        seed_len = int(3.2 * self.sampling_rate) - (len(tail) if tail is not None else 0)
        self.buffer = (
            np.random.randn(
                self.config.infer_config.input.audio_back_size
                + self.config.infer_config.input.audio_ahead_size
            )
            * 0.0001
        )
        self.buffer_for_asr = np.random.randn(int(1.6 * self.sampling_rate)) * 0.00001
        seed = np.random.randn(seed_len) * 0.00001
        self.cascade_buffer = (
            np.concatenate([seed, tail]) if tail is not None else seed
        )
        # how much of cascade_buffer's front is the synthetic seed (0 once
        # the 3.2s rolling truncation in _asr has evicted it) — see _trim_seed
        self.cascade_seed_len = len(seed)
        self.speech_detected = False
        self.past_state = None
        self.history_chunks = []
        self.wait_idle_cnt = 0
        self.monitoring_wait_silence = False
        # a deferred <|user_complete|> never survives a reset — reset ends
        # a turn, and pending turns are only held WITHIN one
        self.pending_complete_cnt = None

    def clear_turn(self):
        self.buffer_for_asr = np.random.randn(int(1.6 * self.sampling_rate)) * 0.00001
        self.speech_detected = False
        self.wait_idle_cnt = 0
        self.monitoring_wait_silence = False
        self.pending_complete_cnt = None

    def _log(self, *args, **kwargs):
        # only for debugging, too much logging will slow down the inference
        if self.developer_mode:
            print(*args, **kwargs)

    def get_rms(self, audio_chunk):
        if audio_chunk.dtype == np.int16:
            samples = audio_chunk.astype(np.float32) / 32768.0
        elif audio_chunk.dtype == np.uint8:
            samples = (audio_chunk.astype(np.float32) - 128.0) / 128.0
        elif audio_chunk.dtype == np.float32:
            samples = np.clip(audio_chunk, -1.0, 1.0)
        else:
            print(audio_chunk.dtype)
            raise ValueError("Unsupported PCM format")
        return np.sqrt(np.mean(samples**2))

    def rms_db(self, y):
        """Calculate RMS dBFS of the audio"""
        rms = np.sqrt(np.mean(y**2))
        if rms < 1e-9:  # Avoid log(0)
            return -np.inf
        return 20 * np.log10(rms)

    def snapshot_runtime(self):
        return {
            "buffer": self.buffer,
            "buffer_for_asr": self.buffer_for_asr,
            "cascade_buffer": self.cascade_buffer,
            "cascade_seed_len": self.cascade_seed_len,
            "speech_detected": self.speech_detected,
            "past_state": self.past_state,
            "history_chunks": self.history_chunks,
            "wait_idle_cnt": self.wait_idle_cnt,
            "monitoring_wait_silence": self.monitoring_wait_silence,
            "pending_complete_cnt": self.pending_complete_cnt,
            "last_commit_text": self.last_commit_text,
        }

    def restore_runtime(self, state):
        if not state:
            self.reset()
            return
        self.buffer = state.get("buffer", self.buffer)
        self.buffer_for_asr = state.get("buffer_for_asr", self.buffer_for_asr)
        self.cascade_buffer = state.get("cascade_buffer", self.cascade_buffer)
        self.cascade_seed_len = state.get("cascade_seed_len", 0)
        self.speech_detected = state.get("speech_detected", False)
        self.past_state = state.get("past_state")
        self.history_chunks = state.get("history_chunks", [])
        self.wait_idle_cnt = state.get("wait_idle_cnt", 0)
        self.monitoring_wait_silence = state.get("monitoring_wait_silence", False)
        self.pending_complete_cnt = state.get("pending_complete_cnt")
        self.last_commit_text = state.get("last_commit_text", "")

    def get_chunk(self):
        if (
            len(self.buffer)
            >= self.config.infer_config.input.chunk_size
            + self.config.infer_config.input.audio_back_size
            + self.config.infer_config.input.audio_ahead_size
        ):
            audio_back = self.buffer[
                : self.config.infer_config.input.audio_back_size
            ].astype(np.float32)

            process_chunk = self.buffer[
                self.config.infer_config.input.audio_back_size : self.config.infer_config.input.audio_back_size
                + self.config.infer_config.input.chunk_size
            ].astype(np.float32)

            audio_ahead = self.buffer[
                self.config.infer_config.input.audio_back_size
                + self.config.infer_config.input.chunk_size : self.config.infer_config.input.audio_back_size
                + self.config.infer_config.input.chunk_size
                + self.config.infer_config.input.audio_ahead_size
            ].astype(np.float32)

            self.buffer = self.buffer[self.config.infer_config.input.chunk_size :]

            # print(process_chunk.dtype)
            return True, process_chunk, audio_back, audio_ahead

        return False, None, None, None

    def process(self, audio_chunk, max_wait_num=None):
        """Process audio chunk, return predict state.

        max_wait_num: optional per-call override of the wait-path silence
        confirmation length (chunks). None -> config default. The model is
        shared across sessions, so this knob travels WITH the chunk instead
        of living on the model instance.

        The returned dict may carry four extra keys (omitted when empty):
          "gate": which blip gate rewrote this chunk's outcome
                  (prob / rms / mistake_no_text / commit_no_text)
          "diag": {p_nonidle, rms} for every pre-latch nonidle chunk — the
                  calibration signal for the gate thresholds (Stage 0)
          "state_log": {from, to} when the dispatched state token changed
                  this chunk — turn.log's always-on STATE lines, so a turn
                  boundary can be read back without developer_mode
          "commit": {trigger, text} when this chunk committed a turn —
                  turn.log's always-on TURN lines
        """
        assert audio_chunk.dtype == np.float32
        # print(audio_chunk.shape)
        self.buffer = np.concatenate([self.buffer, audio_chunk])
        # per-chunk transients, consumed at the end of this call
        self.last_gate = None
        self.last_diag = None
        self.last_state_log = None
        self.last_commit_log = None
        delta_text = ""
        asr_buffer = ""
        predicted_state = {
            "state": "blank",
            "asr_segment": delta_text,
            "asr_buffer": asr_buffer,
        }

        start_prediction, process_chunk, audio_back, audio_ahead = self.get_chunk()

        if start_prediction:
            t_start = time.time()
            predicted_state = self.state_predict(
                process_chunk, audio_back, audio_ahead, max_wait_num
            )
            if self.last_gate is not None:
                predicted_state["gate"] = self.last_gate
            if self.last_diag is not None:
                predicted_state["diag"] = self.last_diag
            if self.last_state_log is not None:
                predicted_state["state_log"] = self.last_state_log
            if self.last_commit_log is not None:
                predicted_state["commit"] = self.last_commit_log
            self._log(f"[Timing] Total chunk: {time.time() - t_start:.4f}s\n\n")

        return predicted_state

    def _recognize(self, audio):
        """Serialized cascade-ASR call (asr_lock — see _init_load_model)."""
        with self.asr_lock:
            return self.cascade_asr.recognize(audio, self.sampling_rate)

    def correct_turn_text(self, audio):
        """Whole-turn offline re-recognition (two-pass ASR 修正, 2026-11-05).

        Live text comes from a 3.2s rolling window with speech chunks glued
        (no pauses) and LCS-diffed against history — measured on 37 real
        turns, recognizing the natural whole-turn audio with this same
        paraformer instance is markedly more accurate. Runs on a server
        worker thread right after the commit message goes out; asr_lock
        serializes against the websocket loop's recognize calls (worst
        case it delays the next chunk's feed_audio by one recognition,
        ~0.1-0.4s, surfaced by turn.log's SLOW feed_audio line).

        audio: raw 16k float32 turn audio (engine-accumulated, pauses
        intact — paraformer's internal VAD uses them). Returns the
        corrected text, or "" when unusable (too short / recognition
        failed) — callers treat "" as "keep the live text".
        """
        if audio is None or len(audio) < 0.3 * self.sampling_rate:
            return ""
        try:
            return self._recognize(audio).strip()
        except Exception as exc:
            print(
                f"[DuplexServer] turn correction failed: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            return ""

    def state_predict(self, process_chunk, audio_back, audio_ahead, max_wait_num=None):
        # prev state for the always-on STATE transition record. past_state
        # is per-session (travels through snapshot/restore_runtime) and is
        # None right after a reset — "start" in the log.
        prev_state = (self.past_state or {}).get("state")
        state, delta_text, asr_buffer = self.infer(
            process_chunk, audio_back, audio_ahead
        )
        if state != prev_state:
            self.last_state_log = {
                "from": _short_state(prev_state),
                "to": _short_state(state),
            }

        if self._pending_reset:
            # escalated mistake_no_text distrust (see _state_predict): the
            # model was stuck claiming speech with no words ever — start
            # clean. Done here, outside infer(), like the far-field reset.
            self._pending_reset = False
            self.reset()
            self.last_gate = {"gate": "mistake_no_text_reset"}
            return {
                "state": "idle",
                "asr_segment": "",
                "asr_buffer": "",
            }

        if (
            self.get_rms(process_chunk) < self.config.infer_config.far_field_threshold
            and not self.speech_detected
            and state == "<|user_nonidle|>"
        ):
            self._log(f"Far-field speech detected: {self.get_rms(process_chunk)}")
            # process_chunk = np.zeros_like(process_chunk)
            self.reset()
            return {
                "state": "idle",
                "asr_segment": "",
                "asr_buffer": "",
            }

        self.past_state["history_len"] += 1

        self.history_chunks.append((process_chunk, state))
        if len(self.history_chunks) > 5:
            self.history_chunks.pop(0)

        if state == "<|user_idle|>":
            self._log("Silence detected")

            if self.pending_complete_cnt is not None:
                # deferred <|user_complete|> (complete_defer): the model now
                # agrees the user went quiet — confirm the held commit. The
                # wait path stays out of it while a complete is being held:
                # this turn is in complete-confirmation, not wait-confirmation.
                self.pending_complete_cnt += 1
                if self.pending_complete_cnt >= self.complete_defer_max_chunks:
                    self.pending_complete_cnt = None
                    return self._complete_commit(
                        delta_text, asr_buffer, process_chunk
                    )

            elif self.monitoring_wait_silence:
                # wait_rms_veto: an idle CLASSIFICATION is not silence —
                # onsets the model mislabels still carry speech energy. A
                # chunk only counts toward the silence confirmation if it
                # is also below the latch RMS gate (2026-11-04 counting
                # incident: the streak-closing "idle" chunk was 「重」's
                # onset at rms 0.0388; breaths ~0.004 still count, real
                # onsets have never been observed under the gate). Cost:
                # decay chunks right after speech end no longer count
                # either, so real-EOT wait commits land ~160-320 ms later.
                if (
                    not self.wait_rms_veto
                    or self.get_rms(process_chunk) < self.nonidle_rms_gate
                ):
                    self.wait_idle_cnt += 1
                elif self.wait_idle_cnt > 0:
                    # energetic "idle" chunk broke a counting streak — the
                    # model misclassified a speech onset as silence (see the
                    # 2026-11-04 counting incident). Surfaced as a gate so
                    # turn.log's GATE lines show the veto doing its job.
                    self.last_gate = {"gate": "wait_rms_veto"}
                effective_wait = max_wait_num or self.config.infer_config["max_wait_num"]
                if self.wait_idle_cnt >= effective_wait:
                    self._log("Continuous silence after wait, triggering reply")
                    if self.speech_detected:
                        segment = self._recognize(
                            self._trim_seed(self.buffer_for_asr, self.asr_seed_len),
                        )
                        segment = self._strip_commit_overlap(segment)
                        if self.commit_no_text_reset and not segment.strip():
                            # Empty recognition: nothing was actually said —
                            # drop the turn instead of committing a phantom.
                            self._log("wait commit suppressed: empty text")
                            self.reset()
                            self.last_gate = {
                                "gate": "commit_no_text",
                                "trigger": "wait",
                            }
                            return {
                                "state": "idle",
                                "asr_segment": delta_text,
                                "asr_buffer": asr_buffer,
                            }
                        self.last_commit_text = segment
                        self.last_commit_log = {
                            "trigger": "wait",
                            "text": segment,
                        }
                        # self.clear_turn()
                        self.reset()
                        return {
                            "state": "speak",
                            "trigger": "wait",
                            # "wait": the boundary was confirmed by
                            # max_wait_num consecutive idle chunks after an
                            # <|user_incomplete|> — silence-verified, the app
                            # can reply without extra confirmation
                            "text": segment,
                            "asr_segment": delta_text,
                            "asr_buffer": asr_buffer,
                        }

            if (
                self.past_state["history_len"] > 200
                and not self.monitoring_wait_silence
                and not self.speech_detected
            ):
                self.reset()

        elif state == "<|user_nonidle|>":
            self._log("Speech detected", self.get_rms(process_chunk.astype(np.float32)))
            self.speech_detected = True

            if self.pending_complete_cnt is not None:
                # deferred complete was premature — the user kept talking;
                # the turn continues (the 「十|数」 misfire shape)
                self._log("deferred complete cancelled: speech resumed")
                self.pending_complete_cnt = None
                self.last_gate = {"gate": "complete_defer_cancel"}

            if self.monitoring_wait_silence:
                self.monitoring_wait_silence = False
                self.wait_idle_cnt = 0

            # semantic vad, yield nonidle after the first word or character finished
            # in case it's cut off in previous chunks
            to_concat = []
            if len(self.history_chunks) >= 2:
                prev_chunk, prev_state = self.history_chunks[-2]
                if prev_state in ["<|user_idle|>", "<|user_backchannel|>"]:
                    candidates = []
                    for i in range(len(self.history_chunks) - 2, -1, -1):
                        c, s = self.history_chunks[i]
                        if s in ["<|user_idle|>", "<|user_backchannel|>"]:
                            candidates.append(c)
                            if len(candidates) >= 5:
                                break
                        else:
                            break
                    to_concat = candidates[::-1]

            if to_concat:
                self.buffer_for_asr = np.concatenate(
                    [self.buffer_for_asr, *to_concat, process_chunk]
                )
            else:
                self.buffer_for_asr = np.concatenate(
                    [self.buffer_for_asr, process_chunk]
                )
            return {
                "state": "nonidle",
                "asr_segment": delta_text,
                "asr_buffer": asr_buffer,
            }

        elif state == "<|user_backchannel|>":
            # self._log("Backchannel detected")
            if not self.speech_detected:
                self.reset()
            if self.monitoring_wait_silence:
                self.wait_idle_cnt = 1

        elif state == "<|user_complete|>":
            self._log("User finished speaking, should reply now")
            if self.speech_detected:
                if (
                    self.complete_defer
                    and self.get_rms(process_chunk) >= self.complete_defer_rms_gate
                ):
                    # energetic complete (complete_defer): this chunk may
                    # hold the onset of the user's next word — the
                    # 「十|数」 misfire shape (rms 0.086 on the 数 onset,
                    # vs <= 0.045 at every real-EOT complete in
                    # calibration). Hold the commit for idle confirmation
                    # (cancelled by a nonidle/incomplete classification).
                    # The chunk still lands in buffer_for_asr so the
                    # confirmed commit recognizes the full turn; the model
                    # narrative itself is untouched — only the returned
                    # action waits.
                    self._log("complete deferred: energetic chunk")
                    self.pending_complete_cnt = 0
                    self.last_gate = {
                        "gate": "complete_defer",
                        "rms": round(float(self.get_rms(process_chunk)), 4),
                    }
                    self.buffer_for_asr = np.concatenate(
                        [self.buffer_for_asr, process_chunk]
                    )
                    return {
                        "state": "nonidle",
                        "asr_segment": delta_text,
                        "asr_buffer": asr_buffer,
                    }
                return self._complete_commit(delta_text, asr_buffer, process_chunk)
            # self.clear_turn()
            self.reset()

        elif state == "<|user_incomplete|>":
            self._log("User hasn't finished speaking, wait")
            if self.pending_complete_cnt is not None:
                # model says the user isn't done — deferred complete was
                # premature, the turn continues
                self.pending_complete_cnt = None
                self.last_gate = {"gate": "complete_defer_cancel"}
            # self.reset()
            self.monitoring_wait_silence = True
            self.wait_idle_cnt = 0
            self.buffer_for_asr = np.concatenate([self.buffer_for_asr, process_chunk])

        else:
            self._log("Unknown state")
            self.reset()

        return {
            "state": "idle",
            "asr_segment": delta_text,
            "asr_buffer": asr_buffer,
        }

    def _complete_commit(self, delta_text, asr_buffer, process_chunk):
        """Commit the turn a <|user_complete|> ended — the direct path
        (complete branch) and the deferred confirmation (complete_defer,
        idle branch) share this. Recognizes buffer_for_asr + the current
        chunk, applies the empty-text guard, records last_commit_text for
        the backoff dedup, resets and returns the speak event."""
        self.buffer_for_asr = np.concatenate([self.buffer_for_asr, process_chunk])
        segment = self._recognize(
            self._trim_seed(self.buffer_for_asr, self.asr_seed_len),
        )
        segment = self._strip_commit_overlap(segment)
        if self.commit_no_text_reset and not segment.strip():
            # Empty recognition: nothing was actually said —
            # drop the turn instead of committing a phantom.
            self._log("complete commit suppressed: empty text")
            self.reset()
            self.last_gate = {
                "gate": "commit_no_text",
                "trigger": "complete",
            }
            return {
                "state": "idle",
                "asr_segment": delta_text,
                "asr_buffer": asr_buffer,
            }
        self.last_commit_text = segment
        self.last_commit_log = {"trigger": "complete", "text": segment}
        # self.clear_turn()
        self.reset()
        return {
            "state": "speak",
            "trigger": "complete",
            # "complete": <|user_complete|> fired the reply the
            # instant the model saw fit — zero silence confirmation,
            # so a mid-sentence pause can look like a finished turn.
            # The app applies a grace window to these (see
            # _run_speech_pipeline) to avoid barging in.
            "text": segment,
            "asr_segment": delta_text,
            "asr_buffer": asr_buffer,
        }

    def _strip_commit_overlap(self, segment):
        """With asr_seed_backoff_s > 0 the previous turn's final audio is
        re-recognized at the head of the new turn's segment — drop the
        duplicated head by matching it against the previous commit's tail.
        Char-level match after punctuation stripping (1 char = 1 token for
        zh; an English word counts as one char — acceptable for an
        experimental, default-off knob; revisit with token-level LCS if it
        ever goes on by default)."""
        if not (self.asr_seed_backoff_s > 0 and self.last_commit_text and segment):
            return segment
        prev = zh_remove_punc(self.last_commit_text)
        cur = zh_remove_punc(segment)
        for k in range(min(8, len(prev), len(cur)), 0, -1):
            if prev[-k:] == cur[:k]:
                return segment[k:].lstrip() or segment
        return segment

    @torch.no_grad()
    def infer(self, audio_chunk, audio_back, audio_ahead):
        self.cascade_buffer = np.concatenate([self.cascade_buffer, audio_chunk])

        # init
        if self.past_state is None:
            self.past_state = {
                "input_embeds": self.text_embeds,
                "past_key_values": None,
                "delta_text": [],
                "cascade_text": "",
                "state": "",
                "history_len": 0,
                "mistake_len": 0,
                "distrust_cnt": 0,
                "checkpoint": None,
            }

        audio_tokens = self._audio_to_tokens(audio_back, audio_chunk, audio_ahead)

        audio_embeds = self._tokens_to_embeds(audio_tokens)

        self.past_state["input_embeds"] = torch.cat(
            (self.past_state["input_embeds"], audio_embeds), dim=0
        ).unsqueeze(0)

        delta_text = self._asr(audio_embeds)

        state = self._state_predict(delta_text, audio_chunk)

        del audio_embeds
        torch.cuda.empty_cache()
        return state, delta_text, self.past_state["cascade_text"]

    def _audio_to_tokens(
        self, audio_back: np.ndarray, audio_chunk: np.ndarray, audio_ahead: np.ndarray
    ):
        """
        audio waveform → discrete speech tokens
        """
        audio_segment = np.concatenate([audio_back, audio_chunk, audio_ahead], axis=0)
        # print(audio_segment.shape)

        start_index = len(audio_back) // self.model.token_samples
        end_index = min(
            start_index + 2,
            math.ceil(audio_segment.shape[0] / self.model.token_samples),
        )

        valid_range = (start_index, end_index)

        pooling_kernel_size = self.model.glm_tokenizer.config.pooling_kernel_size or 1
        stride = (
            self.model.glm_tokenizer.conv1.stride[0]
            * self.model.glm_tokenizer.conv2.stride[0]
            * pooling_kernel_size
            * self.model.feature_extractor.hop_length
        )

        features = self.model.feature_extractor(
            [audio_segment],
            sampling_rate=self.sampling_rate,
            return_attention_mask=True,
            return_tensors="pt",
            device=self.device,
            padding="longest",
            pad_to_multiple_of=stride,
        ).to(self.device)

        outputs = self.model.glm_tokenizer(**features)
        speech_tokens = outputs.quantized_token_ids

        attention_mask = features.attention_mask[
            :,
            :: self.model.glm_tokenizer.conv1.stride[0]
            * self.model.glm_tokenizer.conv2.stride[0],
        ]
        attention_mask = attention_mask[
            :, :: self.model.glm_tokenizer.config.pooling_kernel_size
        ]

        assert attention_mask.shape == speech_tokens.shape
        assert len(speech_tokens) == 1

        speech_token = speech_tokens[0][attention_mask[0].bool()].tolist()
        audio_tokens = speech_token[valid_range[0] : valid_range[1]]

        return audio_tokens

    def _tokens_to_embeds(self, tokens):
        tokens = torch.tensor(tokens, device=self.device)
        embeds = self.model.glm_tokenizer.codebook(tokens)
        embeds = self.model.audio_projector(embeds)

        if embeds.shape[0] != self.chunk_token_len_small:
            embeds = torch.cat(
                (
                    embeds,
                    self.audio_pad_embeds.expand(
                        self.chunk_token_len_small - embeds.shape[0], -1
                    ),
                ),
                dim=0,
            )

        return embeds

    def _asr(self, audio_embeds):
        # 1. Run model to see if it predicts speech (1 step)
        t_llm_check = time.time()
        with torch.no_grad():
            outputs = self.model.llm(
                inputs_embeds=self.past_state["input_embeds"],
                past_key_values=self.past_state["past_key_values"],
                use_cache=True,
            )
            logits = outputs.logits[0]
            current_kv = outputs.past_key_values
            pred = torch.argmax(logits, -1)[-1]
            self._log(f"[Timing] LLM check: {time.time() - t_llm_check:.4f}s")

            delta_text = ""
            need_correction = False
            corrected_prev_delta = ""

            if pred != self.model.asr_eos_token_id:
                # Speech detected -> Cascade ASR
                t_asr = time.time()
                full_text = remove_leading_backchannel(
                    self._recognize(
                        self._trim_seed(self.cascade_buffer, self.cascade_seed_len),
                    )
                )
                self._log(f"[Timing] Cascade ASR: {time.time() - t_asr:.4f}s")

                # Process Text Delta
                history_text = self.past_state.get("cascade_text", "")
                norm_full_text = split_cn_en(zh_norm(zh_remove_punc(full_text.strip())))
                norm_history_text = split_cn_en(
                    zh_norm(zh_remove_punc(history_text.strip()))
                )

                if len(norm_full_text) >= 5 and len(norm_history_text) >= 5:
                    backup_norm_full_text = norm_full_text.copy()
                    backup_norm_history_text = norm_history_text.copy()
                    norm_full_text, norm_history_text = get_lcs_substrings(
                        norm_full_text, norm_history_text
                    )

                prev_delta = (
                    self.past_state["delta_text"][-1]
                    if self.past_state["delta_text"]
                    else ""
                )
                prev_delta_split = split_cn_en(prev_delta)
                len_prev = len(prev_delta_split)

                if len_prev > len(norm_history_text):
                    norm_full_text = backup_norm_full_text
                    norm_history_text = backup_norm_history_text

                history_base = (
                    norm_history_text[:-len_prev] if len_prev > 0 else norm_history_text
                )

                if len(norm_full_text) > len(norm_history_text):
                    current_segment_in_full = norm_full_text[
                        len(history_base) : len(history_base) + len_prev
                    ]
                    if current_segment_in_full == prev_delta_split:
                        # 3.1 Consistent
                        delta_text = "".join(
                            [
                                (s + " ") if check_en(s) else s
                                for s in norm_full_text[len(norm_history_text) :]
                            ]
                        ).strip()
                    else:
                        # 3.2 Inconsistent
                        need_correction = True
                        corrected_prev_delta = "".join(
                            [
                                (s + " ") if check_en(s) else s
                                for s in current_segment_in_full
                            ]
                        ).strip()
                        delta_text = "".join(
                            [
                                (s + " ") if check_en(s) else s
                                for s in norm_full_text[len(history_base) + len_prev :]
                            ]
                        ).strip()

                elif len(norm_full_text) == len(norm_history_text):
                    current_segment_in_full = norm_full_text[len(history_base) :]
                    if current_segment_in_full == prev_delta_split:
                        # 4.1 Consistent
                        delta_text = ""
                    else:
                        # 4.2 Inconsistent
                        need_correction = True
                        corrected_prev_delta = "".join(
                            [
                                (s + " ") if check_en(s) else s
                                for s in current_segment_in_full
                            ]
                        ).strip()
                        delta_text = ""

                else:  # len(norm_full) < len(norm_history)
                    # 5
                    need_correction = True
                    remainder = norm_full_text[len(history_base) :]
                    corrected_prev_delta = ""
                    delta_text = "".join(
                        [(s + " ") if check_en(s) else s for s in remainder]
                    ).strip()

                self.past_state["cascade_text"] = "".join(
                    [(s + " ") if check_en(s) else s for s in norm_full_text]
                ).strip()

                self._log(
                    f"[History]: {''.join([(s + ' ') if check_en(s) else s for s in norm_history_text]).strip()}"
                )
                self._log(f"[ASR] Full: {self.past_state['cascade_text']}")
                self._log(f"[Need Correction]: {need_correction}")
                self._log(f"[Prev Delta]: {corrected_prev_delta}")
                self._log(f"[Delta]: {delta_text}")

            if need_correction and self.past_state["checkpoint"] is not None:
                self._log("--- Correction Triggered ---")
                self.past_state["past_key_values"] = self.past_state["checkpoint"]

                embeds_list = []
                # 1. Corrected Prev Text
                if corrected_prev_delta:
                    ids = self.model.tokenizer.encode(
                        corrected_prev_delta, add_special_tokens=False
                    )
                    if ids:
                        t_ids = torch.tensor(ids, dtype=torch.long).to(self.device)
                        embeds_list.append(self.embed_tokens_func(t_ids))

                # 2. EOS
                embeds_list.append(self.audio_eos_embeds)
                # 3. NonIdle
                embeds_list.append(self.non_idle_embeds)
                # 4. Audio (Current Chunk)
                embeds_list.append(audio_embeds)

                correction_input = torch.cat(embeds_list, dim=0).unsqueeze(0)

                outputs = self.model.llm(
                    inputs_embeds=correction_input,
                    past_key_values=self.past_state["past_key_values"],
                    use_cache=True,
                )
                self.past_state["past_key_values"] = outputs.past_key_values

                if self.past_state["delta_text"]:
                    self.past_state["delta_text"][-1] = corrected_prev_delta
            else:
                self.past_state["past_key_values"] = current_kv

            # Save Checkpoint for NEXT chunk (KV after Audio, before Text)
            self.past_state["checkpoint"] = self.past_state["past_key_values"]

            # cache 3.2s
            self.past_state["delta_text"].append(delta_text)
            max_len = int(3.2 * self.sampling_rate)
            if len(self.cascade_buffer) > max_len:
                self.cascade_buffer = self.cascade_buffer[-max_len:]
                # the synthetic seed (if any) just got evicted from the
                # front — stop trimming it off in _asr (see _trim_seed)
                self.cascade_seed_len = 0

                delta_list = self.past_state["delta_text"][-20:]
                total_tokens = 0
                for t in delta_list:
                    total_tokens += len(split_cn_en(t))

                cascade_tokens = split_cn_en(self.past_state["cascade_text"])
                if total_tokens > 0:
                    keep_tokens = cascade_tokens[-total_tokens:]
                else:
                    keep_tokens = []

                update_text = "".join(
                    [(s + " ") if check_en(s) else s for s in keep_tokens]
                ).strip()
                self.past_state["cascade_text"] = update_text

            self._log(
                f"[Concat]: {''.join([(s + ' ') if check_en(s) else s for s in self.past_state['delta_text']]).strip()}"
            )

            # Prepare input for next step (State Prediction)
            # If delta_text exists, we concat text embeds + eos embeds to save one forward pass
            input_embeds_next = self.audio_eos_embeds.unsqueeze(0)

            if delta_text:
                ids = self.model.tokenizer.encode(delta_text, add_special_tokens=False)
                input_ids = torch.tensor(ids, dtype=torch.long).to(self.device)

                if len(input_ids) > 0:
                    embeds = self.embed_tokens_func(input_ids)
                    embeds = embeds.unsqueeze(0)
                    # Concat text embeds and EOS
                    input_embeds_next = torch.cat((embeds, input_embeds_next), dim=1)

            self.past_state["input_embeds"] = input_embeds_next

            return delta_text

    def _idle_pred(self):
        """A 0-dim idle-token tensor, same shape as torch.argmax's output."""
        return torch.tensor(
            self.config.model_config.user_idle_token_id, device=self.device
        )

    def _apply_prelatch_gates(self, logits_last, pred, state, delta_text, audio_chunk):
        """Pre-latch blip gates: keep breath/laugh residue from latching
        speech_detected (the "对对对。" phantom-turn root cause).

        The latch in state_predict fires on the FIRST nonidle chunk with no
        further evidence, so a breath chunk that barely wins the argmax can
        start a phantom turn. Before the latch, a nonidle chunk must earn it
        via BOTH:
          - model confidence: p(nonidle) over the 7 state tokens >= prob gate
          - energy: chunk RMS >= rms gate
        Words (delta_text) used to bypass both gates, but the cascade ASR
        hallucinates single words on breath chunks exactly in this pre-latch
        window (2026-11-04 "结说。" incident: latch chunk rms 0.004,
        p_nonidle 0.914, hallucinated "去" — the text bypass overrode a
        failing energy gate). Real onsets have never been observed below the
        rms gate (measured minimum 0.012 vs gate 0.01), so the bypass
        protected a case that does not occur while opening this one.
        nonidle_text_bypass restores the old behavior if a genuinely
        sub-gate-energy speech regime ever shows up.
        Gate thresholds live in infer_config; 0 disables a gate.

        Returns (state, pred). pred is overridden alongside state because the
        next-step input embeds are built from it — the model's KV narrative
        must match the state this state machine actually acted on (same
        convention as the complete/incomplete arbitration below).
        """
        if state != "<|user_nonidle|>" or self.speech_detected:
            return state, pred

        # calibration signal (Stage 0): always measure, gating is separate —
        # the 7-way softmax is ~free next to the forward pass it follows.
        # float() keeps np.float32 out of the JSON response.
        probs = torch.softmax(logits_last[self.state_token_ids].float(), dim=-1)
        p_nonidle = float(probs[self.state_token_nonidle_idx].item())
        rms = float(self.get_rms(audio_chunk))
        self.last_diag = {"p_nonidle": round(p_nonidle, 4), "rms": round(rms, 5)}

        if delta_text and self.nonidle_text_bypass:
            return state, pred  # ASR already saw words — real speech

        if self.nonidle_prob_gate > 0.0 and p_nonidle < self.nonidle_prob_gate:
            self._log(f"[Gate] prob: p_nonidle={p_nonidle:.3f} rms={rms:.5f}")
            self.last_gate = {
                "gate": "prob",
                "p_nonidle": round(p_nonidle, 4),
                "rms": round(rms, 5),
            }
            return "<|user_idle|>", self._idle_pred()
        if self.nonidle_rms_gate > 0.0 and rms < self.nonidle_rms_gate:
            self._log(f"[Gate] rms: p_nonidle={p_nonidle:.3f} rms={rms:.5f}")
            self.last_gate = {
                "gate": "rms",
                "p_nonidle": round(p_nonidle, 4),
                "rms": round(rms, 5),
            }
            return "<|user_idle|>", self._idle_pred()
        return state, pred

    def _trim_seed(self, buf, seed_len):
        """Drop the synthetic-noise seed from a recognition input.

        reset()/clear_turn() seed buffer_for_asr (1.6s) and cascade_buffer
        (3.2s) with synthetic noise to give paraformer a noise floor. That
        noise can never contain speech, but prepended to near-silence it is
        exactly the input paraformer hallucinates short text from — trim it.
        seed_len is 0 once the rolling truncation has evicted the seed
        (cascade_buffer only); callers pass self.asr_seed_len /
        self.cascade_seed_len accordingly.
        """
        if not self.asr_seed_trim or seed_len <= 0 or len(buf) <= seed_len:
            return buf
        return buf[seed_len:]

    def _state_predict(self, delta_text, audio_chunk):
        # 4. State Prediction (Feed EOS or Text+EOS)
        t_state = time.time()
        with torch.no_grad():
            outputs = self.model.llm(
                inputs_embeds=self.past_state["input_embeds"],
                past_key_values=self.past_state["past_key_values"],
                use_cache=True,
            )
            logits = outputs.logits[0]
            pred = torch.argmax(logits, -1)[-1]
            state = self.model.tokenizer.decode(pred)

            # pre-latch blip gates — must run BEFORE the mistake_len update
            # below so a downgraded chunk resets the streak instead of
            # feeding it
            state, pred = self._apply_prelatch_gates(
                logits[-1], pred, state, delta_text, audio_chunk
            )

        if state == "<|user_nonidle|>" and not delta_text:
            self.past_state["mistake_len"] += 1
        else:
            self.past_state["mistake_len"] = 0
            if delta_text:
                # words arrived — the nonidle run is real, forget any distrust
                self.past_state["distrust_cnt"] = 0

        mistake_trigger = (
            self.past_state["mistake_len"]
            >= self.config.infer_config.max_mistake_num
        )
        no_text_trigger = (
            self.mistake_no_text_reset
            and self.past_state["mistake_len"] >= self.mistake_no_text_num
        )
        if (
            self.past_state["state"] == "<|user_nonidle|>" and state == "<|user_idle|>"
        ) or mistake_trigger:
            if no_text_trigger and not self.past_state.get("cascade_text", "").strip():
                # The model has claimed speech for mistake_no_text_num chunks
                # but the cascade ASR never saw a word: distrust the nonidle
                # and end the streak as idle — WITHOUT consuming the turn.
                # Arm the wait path instead: the ASR runs every chunk (see
                # infer()), so a real utterance whose text is merely lagging
                # still commits at the wait commit once words arrive, while a
                # blip dies there via the empty-text guard. No reset() here —
                # a slow-ASR real onset stays latched.
                self._log("[Gate] mistake_no_text: nonidle streak without text")
                self.last_gate = {"gate": "mistake_no_text"}
                pred = self._idle_pred()
                state = "<|user_idle|>"
                self.past_state["mistake_len"] = 0
                self.past_state["input_embeds"] = self.embed_tokens_func(
                    pred.unsqueeze(0)
                )
                self.past_state["distrust_cnt"] = (
                    self.past_state.get("distrust_cnt", 0) + 1
                )
                if self.past_state["distrust_cnt"] >= 2:
                    # Second distrust on the same run and still no word ever:
                    # the model is stuck claiming speech over silence — the
                    # wait path can never fire (it needs idle chunks the
                    # model refuses to predict). Escalate to a full reset so
                    # the session cannot hang in nonidle.
                    self._pending_reset = True
                else:
                    self.monitoring_wait_silence = True
                    self.wait_idle_cnt = 0
            elif (
                logits[-1, self.config.model_config.user_complete_token_id]
                > logits[-1, self.config.model_config.user_incomplete_token_id]
            ):
                state = "<|user_complete|>"
                self.past_state["input_embeds"] = self.action_speak_embeds
            else:
                state = "<|user_incomplete|>"
                self.past_state["input_embeds"] = self.action_wait_embeds
        else:
            # Update input_embeds for next turn (state token embedding)
            self.past_state["input_embeds"] = self.embed_tokens_func(pred.unsqueeze(0))

        self.past_state["past_key_values"] = outputs.past_key_values
        self.past_state["state"] = state
        self._log(f"[Timing] State pred: {time.time() - t_state:.4f}s")
        return state


def load_turn_model(config_path: str = "config/config.yaml"):
    """
    Called upon process startup
    """
    turn_model = TurnModel(config_path=config_path)

    print(f"[TurnModel] loaded on {turn_model.device}")
    return turn_model


if __name__ == "__main__":
    model = load_turn_model(config_path="config/config.yaml")

    print("model loaded OK")
