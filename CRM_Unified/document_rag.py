import logging
import os
import re

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("crm-document-rag")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_MODEL_PATH = os.path.join(_THIS_DIR, "models", DEFAULT_MODEL_NAME)

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120
DEFAULT_TOP_K = 5
DEFAULT_MIN_SCORE = 0.2


def chunk_text(text, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP):
    # Splits text into overlapping word-based chunks
    words = re.split(r"\s+", (text or "").strip())
    words = [w for w in words if w]
    if not words:
        return []
    chunks = []
    step = max(chunk_size - chunk_overlap, 1)
    for start in range(0, len(words), step):
        piece = words[start:start + chunk_size]
        if not piece:
            continue
        chunks.append(" ".join(piece))
        if start + chunk_size >= len(words):
            break
    return chunks


class DocumentRAG:
    # Per-session document chunk index with embedding-based retrieval

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_PATH,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = DEFAULT_MIN_SCORE,
    ):
        if os.path.isdir(model_name):
            logger.info("Loading local embedding model from '%s'...", model_name)
            self.model = SentenceTransformer(model_name)
        else:
            logger.warning(
                "No local model found at '%s' — falling back to downloading "
                "'%s' from the Hugging Face Hub.",
                model_name,
                DEFAULT_MODEL_NAME,
            )
            self.model = SentenceTransformer(DEFAULT_MODEL_NAME)

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.min_score = min_score
        self._index: dict = {}

    def index_document(self, session_id: str, filename: str, text: str):
        # Chunks and embeds a document, replacing any prior index for this session
        chunks = chunk_text(text, self.chunk_size, self.chunk_overlap)
        if not chunks:
            self._index.pop(session_id, None)
            return

        embeddings = self.model.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
        self._index[session_id] = {
            "filename": filename,
            "chunks": chunks,
            "embeddings": np.array(embeddings),
        }
        logger.info("Indexed %d chunk(s) for session '%s' ('%s').", len(chunks), session_id, filename)

    def has_document(self, session_id: str) -> bool:
        # Whether a document is indexed for this session
        return session_id in self._index

    def get_filename(self, session_id: str):
        # Filename of the indexed document, if any
        entry = self._index.get(session_id)
        return entry["filename"] if entry else None

    def clear(self, session_id: str):
        # Drops a session's indexed document
        self._index.pop(session_id, None)

    def retrieve(self, session_id: str, query: str, top_k: int = None, min_score: float = None):
        # Returns the most relevant chunk texts for a query, highest similarity first
        entry = self._index.get(session_id)
        if not entry:
            return []

        top_k = self.top_k if top_k is None else top_k
        min_score = self.min_score if min_score is None else min_score

        query_embedding = self.model.encode([query or ""], normalize_embeddings=True)[0]
        scores = entry["embeddings"] @ query_embedding

        ranked_indices = np.argsort(scores)[::-1]
        selected = [
            entry["chunks"][i] for i in ranked_indices[:top_k] if scores[i] >= min_score
        ]

        logger.debug(
            "Session %r query %r -> %d/%d chunk(s) retrieved",
            session_id, query, len(selected), len(entry["chunks"]),
        )

        return selected

    def retrieve_with_scores(self, session_id: str, query: str, top_k: int = None):
        # Same as retrieve(), but returns (chunk, score) pairs regardless of min_score
        entry = self._index.get(session_id)
        if not entry:
            return []

        top_k = self.top_k if top_k is None else top_k
        query_embedding = self.model.encode([query or ""], normalize_embeddings=True)[0]
        scores = entry["embeddings"] @ query_embedding
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        return [(entry["chunks"][i], float(scores[i])) for i in ranked_indices]
