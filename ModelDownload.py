"""
ModelDownload.py

Pre-downloads/caches the embedding model used by CRM_Unified/tool_rag.py
for CRM/web tool retrieval, so the server doesn't have to fetch it from
the Hugging Face Hub on first use:

    sentence-transformers embedding model (CRM tool retrieval)
        -> ./CRM_Unified/models/all-MiniLM-L6-v2

Requirements:

    pip install huggingface-hub sentence-transformers

Usage:
    python ModelDownload.py
    python ModelDownload.py --tool-rag-dir ./CRM_Unified/models/all-MiniLM-L6-v2
    python ModelDownload.py --skip-tool-rag
"""

import argparse
import os

from huggingface_hub import snapshot_download

TOOL_RAG_REPO_ID = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_TOOL_RAG_DIR = "./CRM_Unified/models/all-MiniLM-L6-v2"


def download_tool_rag_model(local_dir: str = DEFAULT_TOOL_RAG_DIR):
    """Download the tool RAG embedding model into its own directory."""
    print(f"[ToolRAG] Downloading '{TOOL_RAG_REPO_ID}' to '{local_dir}'...")
    os.makedirs(local_dir, exist_ok=True)

    path = snapshot_download(repo_id=TOOL_RAG_REPO_ID, local_dir=local_dir)
    print(f"[ToolRAG] Model downloaded to: {path}\n")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Download the CRM tool-retrieval embedding model."
    )
    parser.add_argument(
        "--tool-rag-dir",
        default=DEFAULT_TOOL_RAG_DIR,
        help=f"Local directory to store the ToolRAG embedding model (default: {DEFAULT_TOOL_RAG_DIR}).",
    )
    parser.add_argument(
        "--skip-tool-rag",
        action="store_true",
        help="Skip downloading the ToolRAG embedding model.",
    )
    args = parser.parse_args()

    if not args.skip_tool_rag:
        download_tool_rag_model(args.tool_rag_dir)


if __name__ == "__main__":
    main()
