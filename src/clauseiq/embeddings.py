"""Pluggable embedding backends.

`auto` uses SentenceTransformers when installed, otherwise falls back to a
deterministic hashing embedding so the whole service (and its test-suite) runs
offline with zero downloads. The hashing embedder is not competitive with a
real model, but it is real, deterministic and dependency-free — good enough for
CI, demos and architecture review.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

from .config import settings

_WORD_RE = re.compile(r"[A-Za-z0-9]+")


class Embedder(Protocol):
    dim: int

    def encode(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """Feature-hashing bag-of-words with L2 normalisation.

    Deterministic and offline. Uses sub-word 3-grams so that morphologically
    related legal terms ("terminate"/"termination") share signal.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _tokens(self, text: str) -> list[str]:
        words = [w.lower() for w in _WORD_RE.findall(text)]
        grams = []
        for w in words:
            grams.append(w)
            padded = f"#{w}#"
            grams.extend(padded[i : i + 3] for i in range(len(padded) - 2))
        return grams

    def encode(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            for tok in self._tokens(text):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                idx = h % self.dim
                sign = 1.0 if (h >> 8) & 1 else -1.0
                vec[idx] += sign
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # lazy import

        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return [list(map(float, v)) for v in vecs]


def build_embedder() -> Embedder:
    backend = settings.embedding_backend
    if backend in ("auto", "sentence-transformers"):
        try:
            return SentenceTransformerEmbedder(settings.embedding_model)
        except Exception:
            if backend == "sentence-transformers":
                raise
    return HashingEmbedder(settings.embedding_dim)
