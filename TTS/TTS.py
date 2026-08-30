"""
TTS.py

Text-to-speech backed by OpenAI's audio model (gpt-4o-mini-tts). This
replaces the previous ElevenLabs-based implementation (ELTTS/elevenlabs
SDK) -- same public interface (`synthesize_to_file`, `play_file`,
`speak`), so anything that used to call ELTTS keeps working unchanged
against OpenAITTS.

.env file should contain:
    OPENAI_API_KEY=your_key_here

Optional overrides:
    TTS_VOICE=alloy            # one of OPENAI_VOICES below
    OPENAI_TTS_MODEL=gpt-4o-mini-tts
"""

import os
import tempfile

from openai import OpenAI

# Valid voices for OpenAI's TTS models as of this writing.
OPENAI_VOICES = {
    "alloy", "ash", "ballad", "coral", "echo", "fable",
    "onyx", "nova", "sage", "shimmer", "verse",
}

DEFAULT_VOICE = "alloy"
DEFAULT_TTS_MODEL = "gpt-4o-mini-tts"


class OpenAITTS:
    def __init__(self, voice=DEFAULT_VOICE, model=None):
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not found.")

        self.client = OpenAI(api_key=api_key)

        # Old ElevenLabs ("Rachel") or Kokoro ("af_heart") voice names
        # don't mean anything to OpenAI's TTS -- fall back to a valid
        # default instead of sending a request that will 400.
        self.voice = voice if voice in OPENAI_VOICES else DEFAULT_VOICE
        self.model = model or os.getenv("OPENAI_TTS_MODEL", DEFAULT_TTS_MODEL)

    def synthesize_to_file(self, text: str, output_path=None):

        if output_path is None:
            fd, output_path = tempfile.mkstemp(suffix=".mp3")
            os.close(fd)

        with self.client.audio.speech.with_streaming_response.create(
            model=self.model,
            voice=self.voice,
            input=text,
        ) as response:
            response.stream_to_file(output_path)

        return output_path

    def synthesize_stream(self, text, response_format="pcm"):
        if not text.strip(): return
        with self.client.audio.speech.with_streaming_response.create(model=self.model, voice=self.voice, input=text, response_format=response_format) as response:
            for chunk in response.iter_bytes(chunk_size=4096):
                if chunk: yield chunk

    def play_file(self, audio_path):
        from playsound import playsound
        playsound(audio_path)

    def speak(self, text):
        if not text.strip():
            return text

        path = self.synthesize_to_file(text)

        try:
            self.play_file(path)
        finally:
            if os.path.exists(path):
                os.remove(path)

        return text


# Backward-compatible alias -- older code imported the ElevenLabs-backed
# class under this name.
ELTTS = OpenAITTS
# tts works
