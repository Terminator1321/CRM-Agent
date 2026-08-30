"""
Voice/openai_stt.py

OpenAI Whisper Speech-to-Text module.

Replaces the custom Sarvam STT integration with OpenAI's hosted transcription
API (gpt-4o-mini-transcribe / whisper-1). Accepts raw audio bytes as sent by
the browser (WebM/Opus container) and returns {"transcript": str}.

Environment variables (set in .env):
    OPENAI_API_KEY              - required
    WHISPER_MODEL               - optional, defaults to gpt-4o-mini-transcribe
"""

import io
import os
import logging

from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("openai-stt")

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "gpt-4o-mini-transcribe")

_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


async def transcribe(audio_bytes: bytes, language: str = "en") -> dict:
    """Transcribe raw audio bytes to text using OpenAI Whisper.

    Args:
        audio_bytes: Raw audio bytes. The browser sends WebM/Opus, which
                     OpenAI's API handles natively — no re-encoding needed.
        language:    BCP-47 language tag hint (e.g. "en", "hi"). Optional;
                     passing it slightly improves accuracy and latency.

    Returns:
        {"transcript": str}  — empty string if nothing was detected.
    """
    if not audio_bytes:
        return {"transcript": ""}

    # Wrap bytes in a file-like object. The .name tells the API the format.
    # OpenAI supports: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, webm.
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "audio.webm"

    try:
        result = await _client.audio.transcriptions.create(
            model=WHISPER_MODEL,
            file=audio_file,
            language=language,
            response_format="text",
        )
        # response_format="text" returns a plain string, not a Transcription object
        transcript = result if isinstance(result, str) else getattr(result, "text", "")
        logger.debug("STT transcript (%d bytes -> %d chars)", len(audio_bytes), len(transcript))
        return {"transcript": transcript.strip()}
    except Exception as exc:
        logger.error("OpenAI STT failed: %s", exc)
        raise
