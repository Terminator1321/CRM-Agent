"""Configures logging and exposes the shared app logger."""
import logging
import os

from . import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
for _voice_logger in ("ws_voice", "openai-stt", "openai-tts"):
    logging.getLogger(_voice_logger).setLevel(logging.DEBUG)

logger = logging.getLogger("agent-server")

if config.LANGSMITH_TRACING_ENABLED:
    logger.info(
        "LangSmith tracing ENABLED -- project='%s', endpoint='%s'",
        os.environ.get("LANGCHAIN_PROJECT"),
        os.environ.get("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
    )
else:
    logger.info(
        "LangSmith tracing disabled (set LANGCHAIN_TRACING_V2=true and "
        "LANGCHAIN_API_KEY in .env to enable)."
    )
