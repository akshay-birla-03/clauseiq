"""Clause-aware text chunking.

Legal documents are structured around numbered sections/clauses. We first try
to split on clause boundaries (e.g. "1.", "1.1", "Section 4", "ARTICLE V") and
then pack clauses into token-bounded windows with overlap so that a single
retrieved chunk usually contains a coherent, self-contained clause.
"""
from __future__ import annotations

import re
import uuid

from .schemas import Chunk

_CLAUSE_RE = re.compile(
    r"(?m)^(?:\s*)(?:"
    r"(?:ARTICLE|Article|SECTION|Section)\s+[\dIVXLC]+"  # Article/Section headers
    r"|\d+(?:\.\d+)*\.?\s+[A-Z]"                          # numbered clauses like 1. / 2.3
    r")"
)


def _approx_tokens(text: str) -> int:
    # ~4 chars/token heuristic; avoids a hard tokenizer dependency.
    return max(1, len(text) // 4)


def split_into_clauses(text: str) -> list[str]:
    marks = [m.start() for m in _CLAUSE_RE.finditer(text)]
    if not marks:
        return [text]
    if marks[0] != 0:
        marks = [0] + marks
    segments = []
    for i, start in enumerate(marks):
        end = marks[i + 1] if i + 1 < len(marks) else len(text)
        seg = text[start:end].strip()
        if seg:
            segments.append(seg)
    return segments


def chunk_document(
    doc_id: str, text: str, chunk_size: int = 700, overlap: int = 120
) -> list[Chunk]:
    """Pack clauses into ~chunk_size-token windows with overlap."""
    clauses = split_into_clauses(text)
    chunks: list[Chunk] = []
    window: list[str] = []
    window_tokens = 0
    ordinal = 0

    def flush(carry: list[str]) -> list[str]:
        nonlocal window_tokens, ordinal
        if not window:
            return []
        body = "\n".join(window).strip()
        if body:
            chunks.append(
                Chunk(
                    id=f"{doc_id}::{ordinal}",
                    doc_id=doc_id,
                    text=body,
                    ordinal=ordinal,
                    metadata={"tokens": window_tokens},
                )
            )
            ordinal += 1
        # build overlap carry from tail
        carry_list: list[str] = []
        carry_tokens = 0
        for seg in reversed(window):
            t = _approx_tokens(seg)
            if carry_tokens + t > overlap:
                break
            carry_list.insert(0, seg)
            carry_tokens += t
        return carry_list

    for clause in clauses:
        t = _approx_tokens(clause)
        if window_tokens + t > chunk_size and window:
            window = flush(window)
            window_tokens = sum(_approx_tokens(s) for s in window)
        window.append(clause)
        window_tokens += t

    flush(window)
    # de-dup ids defensively
    seen = set()
    for c in chunks:
        if c.id in seen:
            c.id = f"{doc_id}::{c.ordinal}::{uuid.uuid4().hex[:6]}"
        seen.add(c.id)
    return chunks
