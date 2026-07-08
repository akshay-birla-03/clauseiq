"""FastAPI application exposing ingest / query / health endpoints."""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException

from .pipeline import ClauseIQ
from .schemas import (
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)

app = FastAPI(
    title="ClauseIQ",
    description="Agentic RAG for legal & contract document intelligence.",
    version="0.1.0",
)

_engine: ClauseIQ | None = None


def get_engine() -> ClauseIQ:
    global _engine
    if _engine is None:
        _engine = ClauseIQ()
        # auto-ingest bundled contracts if present
        data_dir = _engine.cfg.data_dir
        if os.path.isdir(data_dir):
            _engine.ingest_dir(data_dir)
    return _engine


@app.get("/health")
def health() -> dict:
    eng = get_engine()
    return {
        "status": "ok",
        "embedder": type(eng.embedder).__name__,
        "llm": eng.llm.name,
        "index": eng.store.stats(),
    }


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    eng = get_engine()
    n = eng.ingest_text(req.doc_id, req.text, req.metadata)
    if n == 0:
        raise HTTPException(status_code=400, detail="No chunks produced from text.")
    return IngestResponse(doc_id=req.doc_id, chunks_indexed=n)


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    eng = get_engine()
    if not eng.store.chunks:
        raise HTTPException(status_code=409, detail="Index is empty. Ingest first.")
    return eng.query(req.question, top_k=req.top_k, use_agent=req.use_agent)
