"""
STT.py

Speech-to-text backed by OpenAI's audio model (gpt-4o-mini-transcribe).
Provides the `assistant.whisper.transcribe_file(path)` interface that
server.py's `/query` endpoint expects -- previously referenced but never
actually implemented (no local Whisper model was ever wired up), so this
also fixes that endpoint rather than just relabeling it.

.env file should contain:
    OPENAI_API_KEY=your_key_here

Optional override:
    OPENAI_STT_MODEL=gpt-4o-mini-transcribe   # or "whisper-1"
"""

import os

from openai import OpenAI

DEFAULT_STT_MODEL = "gpt-4o-mini-transcribe"


class OpenAISTT:
    def __init__(self, model=None):
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not found.")

        self.client = OpenAI(api_key=api_key)
        self.model = model or os.getenv("OPENAI_STT_MODEL", DEFAULT_STT_MODEL)

    def transcribe_file(self, audio_path: str) -> str:
        """Transcribes an audio file (wav/mp3/m4a/webm/ogg, etc.) to text."""
        with open(audio_path, "rb") as f:
            result = self.client.audio.transcriptions.create(
                model=self.model,
                file=f,
            )
        return (result.text or "").strip()


# Backward-compatible alias in case anything refers to this as "Whisper".
Whisper = OpenAISTT
