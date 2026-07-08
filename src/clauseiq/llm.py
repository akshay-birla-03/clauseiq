"""Pluggable LLM backends behind a single `generate` interface.

`auto` uses OpenAI when OPENAI_API_KEY is set, otherwise falls back to an
`ExtractiveLLM` that composes a grounded answer directly from retrieved context.
The extractive backend keeps the service demoable offline and — importantly —
guarantees answers are grounded in citations rather than hallucinated.
"""
from __future__ import annotations

import re
from typing import Protocol

from .config import settings
from .schemas import RetrievedChunk

SYSTEM_PROMPT = (
    "You are ClauseIQ, a meticulous legal-document analyst. Answer ONLY from the "
    "provided contract context. Cite clauses as [doc_id::chunk]. If the context "
    "does not contain the answer, say so explicitly. Never invent obligations, "
    "dates, or amounts."
)


class LLM(Protocol):
    name: str

    def generate(self, question: str, context: list[RetrievedChunk]) -> str: ...


def _format_context(context: list[RetrievedChunk]) -> str:
    blocks = []
    for r in context:
        blocks.append(f"[{r.chunk.id}]\n{r.chunk.text}")
    return "\n\n---\n\n".join(blocks)


class ExtractiveLLM:
    """Deterministic, grounded fallback.

    Ranks sentences in the retrieved context by lexical overlap with the
    question, returns the best-supported sentences with inline citations.
    """

    name = "extractive"
    _SENT_RE = re.compile(r"(?<=[.;:])\s+|\n+")
    _WORD_RE = re.compile(r"[A-Za-z0-9]+")

    def _kw(self, text: str) -> set[str]:
        stop = {
            "the", "a", "an", "of", "to", "in", "is", "are", "and", "or", "for",
            "on", "by", "with", "this", "that", "shall", "any", "be", "as", "at",
            "what", "which", "who", "when", "how", "does", "do",
        }
        return {w.lower() for w in self._WORD_RE.findall(text)} - stop

    def generate(self, question: str, context: list[RetrievedChunk]) -> str:
        if not context:
            return "The provided contracts do not contain information to answer this question."
        q = self._kw(question)
        scored = []
        for r in context:
            for sent in self._SENT_RE.split(r.chunk.text):
                sent = sent.strip()
                if len(sent) < 20:
                    continue
                overlap = len(q & self._kw(sent))
                if overlap:
                    scored.append((overlap, sent, r.chunk.id))
        scored.sort(key=lambda x: x[0], reverse=True)
        if not scored:
            top = context[0]
            return (
                "No sentence directly matched, but the most relevant clause is "
                f"[{top.chunk.id}]: {top.chunk.text[:280].strip()}…"
            )
        seen, lines = set(), []
        for _, sent, cid in scored[:3]:
            if sent in seen:
                continue
            seen.add(sent)
            lines.append(f"- {sent} [{cid}]")
        return "Based on the contract clauses:\n" + "\n".join(lines)


class OpenAILLM:
    name = "openai"

    def __init__(self, model: str, api_key: str):
        from openai import OpenAI  # lazy import

        self._client = OpenAI(api_key=api_key)
        self._model = model

    def generate(self, question: str, context: list[RetrievedChunk]) -> str:
        prompt = (
            f"Contract context:\n{_format_context(context)}\n\n"
            f"Question: {question}\n\nGrounded answer with citations:"
        )
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content.strip()


def build_llm() -> LLM:
    backend = settings.llm_backend
    if backend in ("auto", "openai") and settings.openai_api_key:
        try:
            return OpenAILLM(settings.llm_model, settings.openai_api_key)
        except Exception:
            if backend == "openai":
                raise
    return ExtractiveLLM()
