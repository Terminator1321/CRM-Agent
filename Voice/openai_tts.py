"""
Voice/openai_tts.py

OpenAI Text-to-Speech module.

Replaces the custom Sarvam TTS integration with OpenAI's TTS API.
Returns raw PCM16 audio bytes (24kHz mono) ready to be streamed over WebSocket
directly to the browser — no intermediate WAV container or base64 encoding.

Environment variables (set in .env):
    OPENAI_API_KEY  - required
    TTS_MODEL       - optional, "tts-1" (fast) or "tts-1-hd" (high quality).
                      Defaults to "tts-1" for lowest latency in voice chat.
    TTS_VOICE       - optional, one of: alloy, ash, ballad, coral, echo,
                      fable, onyx, nova, sage, shimmer, verse.
                      Defaults to "alloy".
"""

import os
import logging

from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("openai-tts")

TTS_MODEL = os.environ.get("TTS_MODEL", "tts-1")
TTS_VOICE = os.environ.get("TTS_VOICE", "alloy")

_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


async def synthesize(text: str) -> bytes:
    """Synthesize text to raw PCM16 audio bytes using OpenAI TTS.

    Uses response_format="pcm" which returns raw signed 16-bit little-endian
    PCM at 24kHz mono — the browser AudioContext can decode this directly.
    No WAV header, no base64, no intermediate files.

    Args:
        text: The text to synthesize. Should be a single sentence or short
              paragraph (the TTSFilter in ws_voice.py handles chunking).

    Returns:
        Raw PCM16 bytes at 24kHz mono.
    """
    if not text or not text.strip():
        return b""

    try:
        response = await _client.audio.speech.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            input=text.strip(),
            response_format="pcm",   # raw PCM16 @ 24kHz — lowest latency path
        )
        audio_bytes = response.content
        logger.debug("TTS synthesized %d chars -> %d bytes PCM", len(text), len(audio_bytes))
        return audio_bytes
    except Exception as exc:
        logger.error("OpenAI TTS failed: %s", exc)
        raise
