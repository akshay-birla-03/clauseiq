"""Central configuration, loaded from environment with sane defaults.

The service is designed to run fully offline (no external API keys) using the
local embedding + extractive-LLM fallbacks, and to transparently upgrade to
SentenceTransformers / OpenAI when those are configured.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # Embeddings
    embedding_backend: str = field(
        default_factory=lambda: os.getenv("CLAUSEIQ_EMBEDDING_BACKEND", "auto")
    )  # auto | sentence-transformers | hashing
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "CLAUSEIQ_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    embedding_dim: int = field(
        default_factory=lambda: int(os.getenv("CLAUSEIQ_EMBEDDING_DIM", "384"))
    )

    # LLM
    llm_backend: str = field(
        default_factory=lambda: os.getenv("CLAUSEIQ_LLM_BACKEND", "auto")
    )  # auto | openai | extractive
    llm_model: str = field(
        default_factory=lambda: os.getenv("CLAUSEIQ_LLM_MODEL", "gpt-4o-mini")
    )
    openai_api_key: str | None = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )

    # Retrieval
    chunk_size: int = field(
        default_factory=lambda: int(os.getenv("CLAUSEIQ_CHUNK_SIZE", "700"))
    )
    chunk_overlap: int = field(
        default_factory=lambda: int(os.getenv("CLAUSEIQ_CHUNK_OVERLAP", "120"))
    )
    top_k: int = field(default_factory=lambda: int(os.getenv("CLAUSEIQ_TOP_K", "5")))
    hybrid_alpha: float = field(
        default_factory=lambda: float(os.getenv("CLAUSEIQ_HYBRID_ALPHA", "0.5"))
    )  # weight of dense vs sparse

    # Storage
    data_dir: str = field(
        default_factory=lambda: os.getenv("CLAUSEIQ_DATA_DIR", "data/contracts")
    )
    index_path: str = field(
        default_factory=lambda: os.getenv("CLAUSEIQ_INDEX_PATH", ".clauseiq_index.pkl")
    )

    enable_agent: bool = field(
        default_factory=lambda: _get_bool("CLAUSEIQ_ENABLE_AGENT", True)
    )


settings = Settings()
