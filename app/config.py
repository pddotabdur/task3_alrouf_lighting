from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # Paths
    docs_dir: str = os.getenv("DOCS_DIR", "data/docs")
    index_path: str = os.getenv("INDEX_PATH", "index/tfidf_index.joblib")

    # Chunking
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))

    # Retrieval
    top_k: int = int(os.getenv("TOP_K", "4"))
    min_score: float = float(os.getenv("MIN_SCORE", "0.18"))

    # LLM
    llm_provider: str = os.getenv("LLM_PROVIDER", "mock")  # openai|mock
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


settings = Settings()
