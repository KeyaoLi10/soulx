import os, sys
import grpc
import requests
import io, wave
import time
import re

# Add path for CosyVoice to sys.path if not present
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
COSY_GRPC_PATH = os.path.abspath(
    os.path.join(CURRENT_DIR, "../modules/CosyVoice/async_cosyvoice/runtime/async_grpc")
)

if COSY_GRPC_PATH not in sys.path:
    sys.path.append(COSY_GRPC_PATH)

try:
    import cosyvoice_pb2
    import cosyvoice_pb2_grpc
except ImportError:
    print(
        f"Warning: Failed to import cosyvoice_pb2 from {COSY_GRPC_PATH}. CosyVoice client will not work."
    )

from modules.utils.MyTn.cn_tn import TextNorm
from modules.utils.text_utils import split_cn_en

MORE_PUNCT = "'\",;:、，；：\n"


def _clean_tts_text(text: str) -> str:
    """Light cleanup shared by the HTTP TTS clients: strip markdown-ish
    characters the TTS would read aloud, then drop a trailing pause punct."""
    text = (
        text.replace("*", "")
        .replace("`", "")
        .replace("#", "")
        .replace("-", " ")
        .strip()
    )
    if text and text[-1] in MORE_PUNCT:
        text = text[:-1]
    return text


class IndexTTS_VLLM:
    def __init__(self, speaker: str = "elva", api_url: str = "http://0.0.0.0:6006/tts"):
        self.speaker = speaker
        self.api_url = api_url
        self.normalizer = TextNorm()
        self.MORE_PUNCT = "'\",;:、，；：\n"

        data = {"text": "This is for warm up!", "character": self.speaker}
        try:
            for _ in range(10):
                response = requests.post(self.api_url, json=data)
        except Exception:
            pass

    def synthesize(self, text: str, sample_rate: int = 24000, streaming=None):
        text = self.normalizer(text).replace("*", "").replace("-", " ").strip()
        if text[-1] in self.MORE_PUNCT:
            text = text[:-1]
        if not text:
            return

        data = {"text": text, "character": self.speaker}

        try:
            # start_time = time.time()
            response = requests.post(self.api_url, json=data)
            # print(f"IndexTTS_VLLM inference time: {time.time() - start_time} seconds")

        except Exception as e:
            print(f"IndexTTS_VLLM inference failed: {e}")
            return

        # Convert wav bytes to int16 pcm
        try:
            with io.BytesIO(response.content) as wav_buffer:
                with wave.open(wav_buffer, "rb") as wav_file:
                    yield wav_file.readframes(wav_file.getnframes())
        except Exception as e:
            print(f"Failed to convert WAV to PCM: {e}")
            # Fallback or return empty bytes if conversion fails
            return


class FireRedTTS:
    """Client for the local FireRedTTS OpenAI-compatible service.

    Same synthesize() interface as IndexTTS_VLLM: yields int16 pcm byte
    chunks at 24 kHz mono. The server streams a chunked WAV whose first 44
    bytes are a header — stripped here. Text normalization is done by the
    TTS server itself (split_and_normalize), so only light cleanup is needed.
    """

    def __init__(
        self,
        speaker: str = "f531",
        api_url: str = "http://127.0.0.1:8081/v1/audio/speech",
    ):
        self.speaker = speaker
        self.api_url = api_url
        self.MORE_PUNCT = MORE_PUNCT

    def synthesize(self, text: str, sample_rate: int = 24000, streaming=None,
                   options=None):
        # options (VoxCPM live-tuning: cfg/timesteps/voice_desc) is ignored
        # by FireRedTTS — kept in the signature so both engines are
        # drop-in interchangeable per session.
        text = _clean_tts_text(text)
        if not text:
            return

        response = None
        try:
            response = requests.post(
                self.api_url,
                json={"input": text, "voice": self.speaker, "stream": True},
                stream=True,
                timeout=180,
            )
            response.raise_for_status()
        except Exception as e:
            print(f"FireRedTTS inference failed: {e}")
            return

        header_skipped = False
        try:
            for chunk in response.iter_content(chunk_size=None):
                if not chunk:
                    continue
                if not header_skipped:
                    chunk = chunk[44:]
                    header_skipped = True
                    if not chunk:
                        continue
                yield chunk
        finally:
            # close on EVERY path — when the caller abandons this generator
            # mid-stream (barge-in breaks out of the loop), the socket must
            # still be closed or the TTS server is left streaming to a dead
            # reader
            if response is not None:
                response.close()


class VoxCPM_TTS:
    """Client for the local VoxCPM2 TTS service (services/voxcpm_tts_server.py).

    Same synthesize() interface as FireRedTTS: yields int16 pcm byte chunks
    at 24 kHz mono. The server already resamples VoxCPM2's 48 kHz output and
    streams RAW pcm — no WAV header to strip, unlike the FireRedTTS wire
    format. `options` carries live-tunable params through to the server:
    {"cfg_value": float, "inference_timesteps": int, "voice_desc": str}
    (empty/missing fields fall back to the server's env defaults).
    """

    def __init__(self, api_url: str = "http://127.0.0.1:8085/synthesize"):
        self.api_url = api_url

    def synthesize(self, text: str, sample_rate: int = 24000, streaming=None,
                   options=None):
        text = _clean_tts_text(text)
        if not text:
            return

        response = None
        try:
            response = requests.post(
                self.api_url,
                json={"input": text, **(options or {})},
                stream=True,
                timeout=180,
            )
            response.raise_for_status()
        except Exception as e:
            print(f"VoxCPM_TTS inference failed: {e}")
            return

        try:
            for chunk in response.iter_content(chunk_size=None):
                if chunk:
                    yield chunk
        finally:
            # close on every path, including caller-abandoned streams
            # (barge-in) — otherwise the server streams to a dead reader
            if response is not None:
                response.close()


class Cosyvoice_Streaming_VLLM:
    def __init__(self, host: str = "localhost", port: int = 6006):
        self.host = host
        self.port = port
        self.normalizer = TextNorm()
        self.MORE_PUNCT = "'\",!?;:、，！？；：\n"
        self.streaming_threshold = 10

        try:
            for i in range(10):
                response = self.synthesize("This is for warm up!")
                for _ in response:
                    pass
        except Exception:
            pass

    def detect_language_accent(self, text: str) -> str:
        if not text:
            return "elva-female-zh"

        # Count Chinese characters
        zh_chars = re.findall(r"[\u4e00-\u9fff]", text)
        num_zh = len(zh_chars)

        # Count English words
        en_words = re.findall(r"[a-zA-Z]+", text)
        num_en = len(en_words)

        if num_zh == 0:
            return "elva-female-en"

        if num_zh >= num_en:
            return "elva-female-zh"

        return "elva-female-en"

    def synthesize(self, text: str, sample_rate: int = 24000, streaming=None):
        text = self.normalizer(text).replace("*", "").replace("-", " ").strip()
        if text[-1] in self.MORE_PUNCT:
            text = text[:-1]
        if not text:
            return

        if streaming is None:
            streaming = False
            if len(split_cn_en(text)) >= self.streaming_threshold:
                streaming = True

        try:
            with grpc.insecure_channel(f"{self.host}:{self.port}") as channel:
                stub = cosyvoice_pb2_grpc.CosyVoiceStub(channel)

                req = cosyvoice_pb2.Request()
                req.tts_text = text
                req.stream = streaming
                req.speed = 1.0
                req.text_frontend = True
                req.format = "pcm"
                req.zero_shot_by_spk_id_request.spk_id = self.detect_language_accent(
                    text
                )

                for response in stub.Inference(req):
                    if response.tts_audio:
                        # int16 pcm
                        yield response.tts_audio

        except Exception as e:
            print(f"CosyVoice streaming inference failed: {e}")
            return


# test
if __name__ == "__main__":
    import time

    tts = IndexTTS_VLLM("elva", api_url="http://0.0.0.0:6006/tts")
    text = "Next time you can go earlier or choose a time with fewer people."
    start_time = time.time()
    wav = tts.synthesize(text)
    end_time = time.time()
    print(f"Response time: {end_time - start_time} seconds")
