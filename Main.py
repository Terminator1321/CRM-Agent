"""
Main.py

magnacrm
Full voice assistant pipeline:
    Microphone audio -> OpenAI STT -> LangChain prompt -> LLM (OpenAI) -> OpenAI TTS -> speakers

Requirements:
    pip install langchain-core openai sounddevice numpy soundfile

Make sure OPENAI_API_KEY is set in your .env file (see LLM.py). Both the
STT and TTS steps use OpenAI's hosted audio models (gpt-4o-mini-transcribe
and gpt-4o-mini-tts respectively) -- no local model download needed and
no separate ElevenLabs key required.

Note: `self.llm.model` below is a LangChain-compatible chat model. It isn't
defined on the plain LLM class in LLM.py — server.py monkey-patches
`LLM.model` as a property (see OpenAIChatModel in server.py) before this
module is imported, so that LangChain's `|` chaining works the same way
here as it does in the FastAPI server.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from LLM.LLM import LLM, GENERAL_CRM_PROMPT
from TTS.TTS import OpenAITTS
from TTS.STT import OpenAISTT

class VoiceAssistant:
    def __init__(self, whisper_model: str = None, llm_model: str = "gpt-4o-mini", system_prompt: str = GENERAL_CRM_PROMPT, record_duration: int = 5, tts_voice: str = "alloy", speak_replies: bool = True,):
        self.record_duration = record_duration
        self.speak_replies = speak_replies

        self.llm = LLM(model=llm_model, system_prompt=system_prompt)
        self._tts_voice = tts_voice
        self._tts = None
        # `whisper_model` here names an OpenAI transcription model (e.g.
        # "gpt-4o-mini-transcribe" or "whisper-1"), not a local Whisper
        # checkpoint -- kept as a constructor param for backward
        # compatibility with existing callers/env wiring.
        self._stt_model = whisper_model
        self._whisper = None

        # Pass the system prompt as a literal SystemMessage rather than a
        # ("system", text) template tuple. from_messages() f-string-parses
        # template *strings* looking for {variable} placeholders, and the
        # system prompt now contains literal JSON examples (e.g. {"type":
        # "pie", ...}) for the chart-block instructions -- those curly
        # braces would be misread as template variables and raise
        # "Invalid format specifier ... Nested replacement fields are not
        # allowed." A SystemMessage's content is used as-is, with no
        # template parsing, so this sidesteps that entirely.
        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=self.llm.system_prompt),
                ("human", "{input}"),
            ]
        )

        # LLM output feeds straight into TTS.speak(), which plays audio and
        # passes the text through unchanged so the chain's return value is
        # still the reply text.
        chain = self.prompt | self.llm.model | StrOutputParser()
        if self.speak_replies:
            chain = chain | RunnableLambda(self.tts.speak)
        self.chain = chain

    @property
    def tts(self):
        if self._tts is None:
            self._tts = OpenAITTS(voice=self._tts_voice)
        return self._tts

    @property
    def whisper(self):
        """Named `whisper` for backward compatibility with existing
        callers (e.g. server.py's `/query` endpoint), even though this is
        OpenAI's hosted transcription model rather than local Whisper."""
        if self._whisper is None:
            self._whisper = OpenAISTT(model=self._stt_model)
        return self._whisper


    def process_text(self, text: str) -> str:
        return self.chain.invoke({"input": text})

    # def run_once(self):
        # text = self.whisper.listen(duration=self.record_duration)
        # print(f"You said: {text}")

        # if not text:
        #     print("[No speech detected, skipping]\n")
        #     return None

        # response = self.process_text(text)
        # print(f"AI: {response}\n")
        # return response

    def run_loop(self):
        print("Voice assistant started. Speak after the recording prompt. Press Ctrl+C to stop.\n")
        while True:
            try:
                self.run_once()
            except KeyboardInterrupt:
                print("\nStopped.")
                break
            except Exception as e:
                print(f"[Error: {e}]\n")


if __name__ == "__main__":
    assistant = VoiceAssistant(whisper_model="gpt-4o-mini-transcribe", llm_model="gpt-4o-mini", record_duration=5, tts_voice="alloy", speak_replies=True,)
    assistant.run_loop()
