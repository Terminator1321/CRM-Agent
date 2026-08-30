"""
CRM_Unified/tool_rag.py

Generic tool-retrieval layer: instead of binding every CRM/web tool to the
LLM on every turn (which confuses smaller local models like llama3.2 once
you have more than a handful of tools), this embeds each tool's name +
description once at startup, and at query time retrieves only the top-k
most semantically relevant tools to bind for that turn.

This is domain-agnostic — it doesn't know or care whether the tools are
about leads, deals, contacts, or web research. Feed it any list of
LangChain @tool objects and it works the same way. As you add more tools
(e.g. more CRM DocTypes, more web tools...), you don't touch this file at
all — just pass a bigger tool list in.

Usage:
    from CRM_Unified.tool_rag import ToolRAG
    from CRM_Unified.tools import CRM_TOOLS
    from web.web_tool import WEB_TOOLS

    tool_rag = ToolRAG([*CRM_TOOLS, *WEB_TOOLS])

    candidate_tools = tool_rag.retrieve("who are our open deals this week?")
    llm_with_tools = llm.bind_tools(candidate_tools) if candidate_tools else llm
"""

import logging
import os

# sentence-transformers pulls in transformers, which auto-detects and tries
# to load a TensorFlow backend if TF is installed. On machines with
# TensorFlow + Keras 3 installed, that TF backend import crashes (Keras 3
# isn't supported by transformers' TF integration yet). We only ever need
# the PyTorch backend here, so force transformers to skip TF entirely -
# this must be set before sentence_transformers/transformers are imported.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("crm-tool-rag")

# Downloaded ahead of time by ModelDownload.py into CRM_Unified/models/<name>,
# so this loads from disk instead of hitting the Hugging Face Hub on every
# fresh machine/run.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_MODEL_PATH = os.path.join(_THIS_DIR, "models", DEFAULT_MODEL_NAME)
DEFAULT_TOP_K = 3
DEFAULT_MIN_SCORE = 0.25


class ToolRAG:
    """Retrieves the most relevant tools for a query via embedding
    similarity, instead of binding the entire tool list every time."""

    def __init__(
        self,
        tools,
        model_name: str = DEFAULT_MODEL_PATH,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = DEFAULT_MIN_SCORE,
    ):
        """
        Args:
            tools: list of LangChain @tool-decorated callables. Each must
                have `.name` and `.description` attributes (this is true
                for anything created with @tool).
            model_name: path to a locally downloaded sentence-transformers
                model (see ModelDownload.py), or a model name to fetch from
                the Hugging Face Hub if you'd rather not pre-download it.
                Defaults to the local path this project downloads models
                into (CRM_Unified/models/all-MiniLM-L6-v2).
            top_k: max number of tools to return per query.
            min_score: minimum cosine similarity a tool must hit to be
                considered relevant at all. Tools scoring below this are
                dropped — this is what lets plain chit-chat ("hi", "thanks")
                retrieve zero tools instead of being forced into one.
        """
        if not tools:
            raise ValueError("ToolRAG needs at least one tool to index.")

        self.tools = list(tools)
        self.top_k = top_k
        self.min_score = min_score

        if os.path.isdir(model_name):
            logger.info("Loading local embedding model from '%s'...", model_name)
            self.model = SentenceTransformer(model_name)
        else:
            logger.warning(
                "No local model found at '%s' — falling back to downloading "
                "'%s' from the Hugging Face Hub. Run ModelDownload.py to "
                "cache it locally and avoid this on future runs.",
                model_name,
                DEFAULT_MODEL_NAME,
            )
            self.model = SentenceTransformer(DEFAULT_MODEL_NAME)

        self._index_tools()

    def _tool_text(self, tool) -> str:
        """Text used to represent a tool for embedding. Combining name +
        description tends to retrieve better than description alone,
        since users often phrase things close to the tool's name."""
        return f"{tool.name}: {tool.description}"

    def _index_tools(self):
        texts = [self._tool_text(tool) for tool in self.tools]
        self.embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        logger.info("Indexed %d tool(s) for retrieval.", len(self.tools))

    def add_tools(self, new_tools):
        """Adds more tools after construction (e.g. a new domain's tools
        registered later at runtime) and re-embeds just those."""
        if not new_tools:
            return
        new_embeddings = self.model.encode(
            [self._tool_text(tool) for tool in new_tools],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self.tools.extend(new_tools)
        self.embeddings = np.vstack([self.embeddings, new_embeddings])
        logger.info("Added %d tool(s); %d total now indexed.", len(new_tools), len(self.tools))

    def retrieve(self, query: str, top_k: int = None, min_score: float = None):
        """Returns the list of tool objects most relevant to `query`,
        ranked highest similarity first. Returns an empty list if nothing
        clears `min_score` (i.e. the query doesn't need any ERP tool)."""
        top_k = self.top_k if top_k is None else top_k
        min_score = self.min_score if min_score is None else min_score

        query_embedding = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.embeddings @ query_embedding  # cosine sim (both sides normalized)

        ranked_indices = np.argsort(scores)[::-1]

        selected = [
            self.tools[i] for i in ranked_indices[:top_k] if scores[i] >= min_score
        ]

        logger.debug(
            "Query %r -> tools %s",
            query,
            [(self.tools[i].name, round(float(scores[i]), 3)) for i in ranked_indices[:top_k]],
        )

        return selected

    def retrieve_with_scores(self, query: str, top_k: int = None):
        """Same as retrieve(), but returns (tool, score) pairs regardless
        of min_score — useful for debugging/tuning the threshold."""
        top_k = self.top_k if top_k is None else top_k

        query_embedding = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.embeddings @ query_embedding
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        return [(self.tools[i], float(scores[i])) for i in ranked_indices]


if __name__ == "__main__":
    # Quick manual smoke test. Run directly: python -m CRM_Unified.tool_rag
    # (uses dummy tools so this can be sanity-checked without a live CRM)
    from langchain_core.tools import tool

    @tool
    def crm_search_deals():
        """Search CRM deals by stage, owner, or organization."""
        return "dummy"

    @tool
    def crm_search_contacts():
        """Search CRM contacts and their linked organizations."""
        return "dummy"

    @tool
    def web_search():
        """Search the public web for general information."""
        return "dummy"

    logging.basicConfig(level=logging.INFO)
    rag = ToolRAG([crm_search_deals, crm_search_contacts, web_search])

    for question in [
        "what deals are open this week?",
        "find the contact for Acme Corp",
        "hey, how's it going?",
    ]:
        results = rag.retrieve_with_scores(question)
        print(f"\nQuery: {question}")
        for t, score in results:
            print(f"  {t.name}: {score:.3f}")
