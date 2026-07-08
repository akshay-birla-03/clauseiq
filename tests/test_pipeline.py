import os

from clauseiq.agent import decompose
from clauseiq.config import Settings
from clauseiq.eval.harness import evaluate, load_cases
from clauseiq.llm import ExtractiveLLM
from clauseiq.pipeline import ClauseIQ

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "contracts")
CASES = os.path.join(os.path.dirname(__file__), "..", "data", "eval_cases.json")


def _engine():
    cfg = Settings(embedding_backend="hashing", llm_backend="extractive")
    eng = ClauseIQ(cfg)
    eng.ingest_dir(DATA)
    return eng


def test_ingest_produces_chunks():
    eng = _engine()
    assert eng.store.stats()["chunks"] > 0
    assert eng.store.stats()["docs"] == 4


def test_query_returns_grounded_citations():
    eng = _engine()
    resp = eng.query("What is the termination notice period?")
    assert resp.citations
    valid = {c.doc_id for c in eng.store.chunks}
    assert all(c.doc_id in valid for c in resp.citations)


def test_agent_decomposes_compound_question():
    subs = decompose(
        "What is the notice period and what happens on a material breach?",
        ExtractiveLLM(),
    )
    assert len(subs) >= 2


def test_eval_harness_runs():
    eng = _engine()
    results = evaluate(eng, load_cases(CASES), k=5)
    assert results["cases"] == 8
    assert 0.0 <= results["retrieval_hit@k"] <= 1.0
    # hashing embedder + bm25 should retrieve the right doc most of the time
    assert results["retrieval_hit@k"] >= 0.5
