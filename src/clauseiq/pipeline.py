"""Top-level orchestration: build embedder/store/llm/agent, ingest & query."""
from __future__ import annotations

import glob
import os

from .agent import RagAgent
from .chunking import chunk_document
from .config import Settings, settings
from .embeddings import build_embedder
from .llm import build_llm
from .schemas import QueryResponse
from .vectorstore import HybridVectorStore


class ClauseIQ:
    def __init__(self, cfg: Settings | None = None):
        self.cfg = cfg or settings
        self.embedder = build_embedder()
        self.store = HybridVectorStore(self.embedder)
        self.llm = build_llm()
        self.agent = RagAgent(
            self.store, self.llm, self.cfg.top_k, self.cfg.hybrid_alpha
        )

    def ingest_text(self, doc_id: str, text: str, metadata: dict | None = None) -> int:
        chunks = chunk_document(
            doc_id, text, self.cfg.chunk_size, self.cfg.chunk_overlap
        )
        for c in chunks:
            c.metadata.update(metadata or {})
        return self.store.add(chunks)

    def ingest_dir(self, path: str | None = None) -> dict:
        path = path or self.cfg.data_dir
        total = 0
        files = sorted(glob.glob(os.path.join(path, "*.txt")))
        for fp in files:
            with open(fp, "r", encoding="utf-8") as f:
                text = f.read()
            doc_id = os.path.splitext(os.path.basename(fp))[0]
            total += self.ingest_text(doc_id, text, {"source": fp})
        return {"documents": len(files), "chunks": total, **self.store.stats()}

    def query(
        self, question: str, top_k: int | None = None, use_agent: bool | None = None
    ) -> QueryResponse:
        if top_k is not None:
            self.agent.top_k = top_k
        agent_on = self.cfg.enable_agent if use_agent is None else use_agent
        return self.agent.answer(question, use_agent=agent_on)

    def save(self, path: str | None = None) -> None:
        self.store.save(path or self.cfg.index_path)

    def load(self, path: str | None = None) -> None:
        self.store.load(path or self.cfg.index_path)
