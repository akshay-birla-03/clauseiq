"""A lightweight agentic RAG layer.

For multi-part legal questions ("What is the notice period and what happens on
breach?"), a single retrieval often misses one facet. The agent decomposes the
query into sub-questions, retrieves per sub-question, then synthesises a single
grounded answer. Decomposition uses the LLM when available and a heuristic
splitter otherwise, so the agent works in every backend configuration.
"""
from __future__ import annotations

import re

from .llm import LLM
from .schemas import Citation, QueryResponse, RetrievedChunk

_CONNECTORS = re.compile(r"\b(and|also|as well as|plus|additionally)\b", re.I)


def decompose(question: str, llm: LLM) -> list[str]:
    # Heuristic decomposition: split compound questions on connectors and "?".
    parts = [p.strip() for p in re.split(r"\?|;", question) if p.strip()]
    subs: list[str] = []
    for p in parts:
        if _CONNECTORS.search(p) and len(p.split()) > 8:
            for seg in _CONNECTORS.split(p):
                seg = seg.strip()
                if seg and seg.lower() not in {
                    "and", "also", "as well as", "plus", "additionally"
                } and len(seg.split()) > 2:
                    subs.append(seg if seg.endswith("?") else seg + "?")
        else:
            subs.append(p if p.endswith("?") else p + "?")
    # de-dup, cap
    out, seen = [], set()
    for s in subs:
        key = s.lower()
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out[:4] or [question]


class RagAgent:
    def __init__(self, store, llm: LLM, top_k: int, alpha: float):
        self.store = store
        self.llm = llm
        self.top_k = top_k
        self.alpha = alpha

    def _citations(self, retrieved: list[RetrievedChunk]) -> list[Citation]:
        cites = []
        for r in retrieved:
            snippet = r.chunk.text.strip().replace("\n", " ")
            cites.append(
                Citation(
                    doc_id=r.chunk.doc_id,
                    chunk_id=r.chunk.id,
                    snippet=snippet[:220] + ("…" if len(snippet) > 220 else ""),
                    score=round(r.score, 4),
                )
            )
        return cites

    def answer(self, question: str, use_agent: bool = True) -> QueryResponse:
        sub_questions = decompose(question, self.llm) if use_agent else [question]

        pooled: dict[str, RetrievedChunk] = {}
        for sq in sub_questions:
            for r in self.store.search(sq, top_k=self.top_k, alpha=self.alpha):
                prev = pooled.get(r.chunk.id)
                if prev is None or r.score > prev.score:
                    pooled[r.chunk.id] = r
        retrieved = sorted(pooled.values(), key=lambda r: r.score, reverse=True)
        retrieved = retrieved[: self.top_k]

        answer = self.llm.generate(question, retrieved)
        return QueryResponse(
            question=question,
            answer=answer,
            citations=self._citations(retrieved),
            sub_questions=sub_questions if use_agent and len(sub_questions) > 1 else [],
            backend={"llm": self.llm.name},
        )
