import requests

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


class VoxCPM_TTS:
    """Client for the local VoxCPM2 TTS service (services/voxcpm_tts_server.py).

    Yields int16 pcm byte chunks at 24 kHz mono. The server already resamples
    VoxCPM2's 48 kHz output and streams RAW pcm — no WAV header to strip.
    `options` carries live-tunable params through to the server:
    {"cfg_value": float, "inference_timesteps": int, "lora": str}
    (voice is fixed server-side by the anchor reference — not a request
    option since 2026-10-29). "lora" selects the LoRA checkpoint when the
    server runs in dual-checkpoint mode (2026-11-09); the server clamps
    unknown names to its default.

    2026-11-09: the FireRedTTS / IndexTTS / Cosyvoice clients were removed
    along with the FireRedTTS service (:8081) — VoxCPM2 is the only TTS.
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
