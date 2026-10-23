"""Deploy to a Hugging Face Docker Space.

    HF_TOKEN=hf_... python scripts/deploy_hf.py <user>/citewise

Creates the Space if needed, stores GOOGLE_API_KEY from .env as a Space *secret*
(never in the repo), and uploads only what the image needs. The Space README is
generated here because Spaces read their config from its YAML front matter.
"""

from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402

SPACE_README = """---
title: Citewise
emoji: 📎
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
short_description: RAG that answers with receipts - cited, or honestly refused
---

# Citewise — answers with receipts

Ask questions about a fictional company's documents (Halcyon Robotics). Every claim
is cited to its source; when the documents don't contain the answer, Citewise says
so instead of guessing.

Hybrid retrieval (BM25 + Gemini embeddings, fused with RRF) → LLM reranking →
a relevance gate that refuses before the LLM is ever called → grounded generation
with `[n]` citations. Built with LangGraph, FastAPI and Streamlit.
"""

INCLUDE = ["Dockerfile", "requirements.txt", "start.sh", "app/**", "ui/**",
           "corpus/**", "eval/golden.jsonl"]


def main(space: str) -> None:
    settings = get_settings()
    api = HfApi()  # HF_TOKEN env var, or a prior `hf auth login`
    api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)

    if settings.google_api_key:
        api.add_space_secret(space, "GOOGLE_API_KEY", settings.google_api_key)
    else:
        print("warning: no GOOGLE_API_KEY in .env - add it as a Space secret by hand")
    api.add_space_variable(space, "CHAT_MODEL", settings.chat_model)
    api.add_space_variable(space, "EMBEDDING_MODEL", settings.embedding_model)

    api.upload_folder(
        repo_id=space, repo_type="space", folder_path=str(ROOT),
        allow_patterns=INCLUDE, ignore_patterns=["**/__pycache__/**"],
        commit_message="Deploy Citewise",
    )
    api.upload_file(
        repo_id=space, repo_type="space", path_in_repo="README.md",
        path_or_fileobj=SPACE_README.encode(), commit_message="Space card",
    )
    print(f"https://huggingface.co/spaces/{space}")


if __name__ == "__main__":
    if len(sys.argv) != 2 or "/" not in sys.argv[1]:
        sys.exit("usage: python scripts/deploy_hf.py <user>/<space-name>")
    main(sys.argv[1])
