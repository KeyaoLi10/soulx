import os
import json
import argparse
from glob import glob

import soundfile as sf
import torch, torchaudio
from tqdm import tqdm

import string
from zhon.hanzi import punctuation
import re

MODEL_NAME = ""
prefix = ""


def split_cn_en(text: str):
    pattern = r"[\u4e00-\u9fff]|[A-Za-z]+|[0-9]+"
    return re.findall(pattern, text)


def zh_remove_punc(text):
    punctuation_all = punctuation + string.punctuation
    for x in punctuation_all:
        text = text.replace(x, "")

    text = text.replace("  ", " ")
    return text


def get_time_aligned_transcription(data_path, task):
    # Collect all output.wav files under the root directory
    _resample_buffer: dict[int, torchaudio.transforms.Resample] = {}
    audio_paths = sorted(glob(f"{data_path}/*/{MODEL_NAME}{prefix}output.wav"))

    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks

    vad_pipeline = pipeline(
        task=Tasks.voice_activity_detection,
        model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        model_revision="v2.0.4",
        device=f"cuda",
        disable_pbar=True,
        disable_update=True,
    )

    asr_pipeline = pipeline(
        task=Tasks.auto_speech_recognition,
        model="iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        model_revision="v2.0.4",
        device=f"cuda",
        disable_pbar=True,
        disable_update=True,
    )

    for audio_path in tqdm(audio_paths):
        # print(audio_path)
        # Read the audio file (waveform and sample rate)
        wav, sr = torchaudio.load(audio_path)
        # If multichannel audio, convert to mono by averaging channels
        if wav.shape[0] != 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
            assert wav.shape[0] == 1

        if sr != 16000:
            if sr not in _resample_buffer:
                _resample_buffer[sr] = torchaudio.transforms.Resample(
                    orig_freq=sr, new_freq=16000
                )
            wav = _resample_buffer[sr](wav).squeeze()
            sr = 16000
        else:
            wav = wav.squeeze()

        # Default offset is zero (no cropping)
        assert sr == 16000
        sample_per_ms = 16000 // 1000
        offset = 0.0

        if task == "user_interruption":
            # Load the interrupt metadata to get [start, end] timestamps
            meta_path = audio_path.replace(
                f"{MODEL_NAME}{prefix}output.wav", "interrupt.json"
            )
            with open(meta_path, "r") as f:
                interrupt_meta = json.load(f)

            # We only care about the end of the interruption
            _, end_interrupt = interrupt_meta[0]["timestamp"]
            offset = end_interrupt

            # Compute the sample index to start from, and crop the waveform
            start_idx = int(end_interrupt * sr)
            wav = wav[start_idx:]

        vad_result = vad_pipeline(wav)

        vad_chunks = vad_result[0]["value"]

        asr_results = [
            asr_pipeline(
                wav[chunk[0] * sample_per_ms : chunk[1] * sample_per_ms],
                return_timestamps=True,
            )
            for chunk in vad_chunks
        ]

        # Build the output dict, adjusting each timestamp by the offset
        chunks = []
        text = ""

        for i in range(len(asr_results)):
            if not asr_results[i]:
                continue
            vad_chunk = vad_chunks[i]
            asr_result = asr_results[i][0]

            asr_text = asr_result["text"].strip()
            # print(asr_text)

            text += asr_text
            norm_text = split_cn_en(zh_remove_punc(asr_text))
            # print(norm_text)

            timestamps = [
                [
                    timestamp[0] + vad_chunk[0],
                    timestamp[1] + vad_chunk[0],
                ]
                for timestamp in asr_result["timestamp"]
            ]

            for i in range(len(norm_text)):
                start_time = timestamps[i][0] / 1000 + offset
                end_time = timestamps[i][1] / 1000 + offset
                word = norm_text[i]

                chunks.append(
                    {
                        "text": word,
                        "timestamp": [start_time, end_time],
                    }
                )

        output_dict = {
            "text": text.strip(),
            "chunks": chunks,
        }

        # Write the JSON result next to the WAV file
        result_path = audio_path.replace(
            f"{MODEL_NAME}{prefix}output.wav", f"{prefix}output.json"
        )
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(output_dict, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transcribe full audio or only after a user interruption"
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root folder containing subfolders with output.wav (and interrupt.json)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="full",
        choices=["full", "user_interruption"],
        help="Choose 'full' for entire transcript or 'user_interruption' to crop before ASR",
    )
    args = parser.parse_args()

    get_time_aligned_transcription(args.root_dir, args.task)
