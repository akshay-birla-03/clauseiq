"""Pydantic schemas shared across the API, agent and pipeline layers."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    id: str
    doc_id: str
    text: str
    ordinal: int
    metadata: dict = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
    dense_score: float = 0.0
    sparse_score: float = 0.0


class Citation(BaseModel):
    doc_id: str
    chunk_id: str
    snippet: str
    score: float


class IngestRequest(BaseModel):
    doc_id: str
    text: str
    metadata: dict = Field(default_factory=dict)


class IngestResponse(BaseModel):
    doc_id: str
    chunks_indexed: int


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    use_agent: Optional[bool] = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    sub_questions: list[str] = Field(default_factory=list)
    backend: dict = Field(default_factory=dict)
