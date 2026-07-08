"""A small, dependency-free hybrid vector store.

Combines dense cosine similarity (from the embedder) with a BM25 sparse score,
then fuses them with a configurable alpha. This mirrors what a production stack
would do with Qdrant/Elasticsearch, but keeps the reference implementation
inspectable and testable in-process. Persistable via pickle.
"""
from __future__ import annotations

import math
import pickle
import re
from collections import Counter

from .schemas import Chunk, RetrievedChunk

_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


class HybridVectorStore:
    def __init__(self, embedder):
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self.vectors: list[list[float]] = []
        # BM25 state
        self._doc_tokens: list[list[str]] = []
        self._df: Counter = Counter()
        self._avgdl: float = 0.0
        self.k1 = 1.5
        self.b = 0.75

    # ---- indexing -------------------------------------------------------
    def add(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        vecs = self.embedder.encode([c.text for c in chunks])
        for c, v in zip(chunks, vecs):
            self.chunks.append(c)
            self.vectors.append(v)
            toks = _tokenize(c.text)
            self._doc_tokens.append(toks)
            for term in set(toks):
                self._df[term] += 1
        total = sum(len(t) for t in self._doc_tokens)
        self._avgdl = total / max(1, len(self._doc_tokens))
        return len(chunks)

    # ---- scoring --------------------------------------------------------
    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        # vectors are pre-normalised, so dot product == cosine
        return sum(x * y for x, y in zip(a, b))

    def _bm25_scores(self, query_tokens: list[str]) -> list[float]:
        n = len(self._doc_tokens)
        scores = [0.0] * n
        q_terms = set(query_tokens)
        for term in q_terms:
            df = self._df.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for i, toks in enumerate(self._doc_tokens):
                tf = toks.count(term)
                if tf == 0:
                    continue
                dl = len(toks)
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self._avgdl or 1))
                scores[i] += idf * (tf * (self.k1 + 1)) / denom
        return scores

    @staticmethod
    def _minmax(xs: list[float]) -> list[float]:
        if not xs:
            return xs
        lo, hi = min(xs), max(xs)
        if hi - lo < 1e-12:
            return [0.0 for _ in xs]
        return [(x - lo) / (hi - lo) for x in xs]

    def search(
        self, query: str, top_k: int = 5, alpha: float = 0.5
    ) -> list[RetrievedChunk]:
        if not self.chunks:
            return []
        qvec = self.embedder.encode([query])[0]
        dense = [self._cosine(qvec, v) for v in self.vectors]
        sparse = self._bm25_scores(_tokenize(query))
        dn, sn = self._minmax(dense), self._minmax(sparse)
        fused = [alpha * d + (1 - alpha) * s for d, s in zip(dn, sn)]
        order = sorted(range(len(fused)), key=lambda i: fused[i], reverse=True)[:top_k]
        return [
            RetrievedChunk(
                chunk=self.chunks[i],
                score=fused[i],
                dense_score=dense[i],
                sparse_score=sparse[i],
            )
            for i in order
        ]

    # ---- persistence ----------------------------------------------------
    def save(self, path: str) -> None:
        state = {
            "chunks": [c.model_dump() for c in self.chunks],
            "vectors": self.vectors,
            "doc_tokens": self._doc_tokens,
            "df": dict(self._df),
            "avgdl": self._avgdl,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.chunks = [Chunk(**c) for c in state["chunks"]]
        self.vectors = state["vectors"]
        self._doc_tokens = state["doc_tokens"]
        self._df = Counter(state["df"])
        self._avgdl = state["avgdl"]

    def stats(self) -> dict:
        return {
            "chunks": len(self.chunks),
            "docs": len({c.doc_id for c in self.chunks}),
            "vocab": len(self._df),
        }
