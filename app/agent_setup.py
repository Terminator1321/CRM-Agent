"""Builds the VoiceAssistant, patches LLM.model to use OpenAIChatModel, and
loads/indexes all CRM + web tools. Runs its setup at import time."""
from langchain_core.output_parsers import StrOutputParser

from LLM.LLM import LLM

from . import config
from .llm_model import OpenAIChatModel
from .logging_setup import logger


@property
def _model(self):
    return OpenAIChatModel(
        model_name=self.model_name,
        temperature=self.temperature,
        api_key=self.api_key,
        base_url=self.base_url,
    )


LLM.model = _model

from Main import VoiceAssistant  # noqa: E402  (must follow the LLM.model patch above)
from CRM_Unified.tools import CRM_TOOLS  # noqa: E402
from CRM_Unified.tool_rag import ToolRAG  # noqa: E402
from CRM_Unified.document_rag import DocumentRAG  # noqa: E402
from web.web_tool import WEB_TOOLS  # noqa: E402
from mailer.email_tools import EMAIL_TOOLS  # noqa: E402

logger.info("Loading VoiceAssistant agent (STT=%s, LLM=%s)...", config.WHISPER_MODEL, config.LLM_MODEL)
assistant = VoiceAssistant(
    whisper_model=config.WHISPER_MODEL,
    llm_model=config.LLM_MODEL,
    tts_voice=config.TTS_VOICE,
    speak_replies=False,
)
text_chain = assistant.prompt | assistant.llm.model | StrOutputParser()

ALL_TOOLS = [*CRM_TOOLS, *WEB_TOOLS, *EMAIL_TOOLS]
ALL_REQUIRED_FIELDS: dict = {}
ALL_FIELD_PARSERS: dict = {}

tool_rag = None
if ALL_TOOLS:
    logger.info("Indexing %d CRM/web tool(s) for retrieval...", len(ALL_TOOLS))
    tool_rag = ToolRAG(ALL_TOOLS, top_k=config.TOOL_RAG_TOP_K, min_score=config.TOOL_RAG_MIN_SCORE)
tool_map = {tool.name: tool for tool in ALL_TOOLS}
logger.info("Loaded %d CRM/web tools.", len(ALL_TOOLS))

# Dedicated LLM instance for document OCR/extraction, kept history-free.
llm_ocr_engine = LLM()

doc_rag = DocumentRAG(
    chunk_size=config.DOC_RAG_CHUNK_SIZE,
    chunk_overlap=config.DOC_RAG_CHUNK_OVERLAP,
    top_k=config.DOC_RAG_TOP_K,
    min_score=config.DOC_RAG_MIN_SCORE,
)
