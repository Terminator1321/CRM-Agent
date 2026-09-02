"""All environment-driven constants used across the app package."""
import os

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "gpt-4o-mini-transcribe")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
TTS_VOICE = os.environ.get("TTS_VOICE", "alloy")

LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "4096"))
MAX_COMPLETION_ROUNDS = int(os.environ.get("LLM_MAX_COMPLETION_ROUNDS", "4"))
LLM_REQUEST_TIMEOUT_SECONDS = float(os.environ.get("LLM_REQUEST_TIMEOUT_SECONDS", "90"))

TOOL_RAG_BYPASS_THRESHOLD = 100
TOOL_RAG_TOP_K = int(os.environ.get("TOOL_RAG_TOP_K", "3"))
TOOL_RAG_MIN_SCORE = float(os.environ.get("TOOL_RAG_MIN_SCORE", "0.25"))

DOC_RAG_CHUNK_SIZE = int(os.environ.get("DOC_RAG_CHUNK_SIZE", "800"))
DOC_RAG_CHUNK_OVERLAP = int(os.environ.get("DOC_RAG_CHUNK_OVERLAP", "120"))
DOC_RAG_TOP_K = int(os.environ.get("DOC_RAG_TOP_K", "5"))
DOC_RAG_MIN_SCORE = float(os.environ.get("DOC_RAG_MIN_SCORE", "0.2"))

MAX_HISTORY_TOKENS = 60000

CORS_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]

os.environ.setdefault("LANGCHAIN_PROJECT", "magma-assistance")
LANGSMITH_TRACING_ENABLED = (
    os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true"
    and bool(os.environ.get("LANGCHAIN_API_KEY"))
)
