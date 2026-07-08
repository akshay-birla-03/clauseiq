"""A minimal but honest RAG evaluation harness.

Given a gold set of (question, expected_doc_id, expected_keywords), it measures:
  * retrieval_hit@k  — did the relevant doc appear in the top-k chunks?
  * answer_coverage  — fraction of expected keywords present in the answer
  * grounding        — fraction of citations that actually belong to the corpus

These are deliberately simple, transparent metrics — the point is to show a
repeatable evaluation loop, not to claim SOTA numbers.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from ..pipeline import ClauseIQ


@dataclass
class EvalCase:
    question: str
    expected_doc_id: str
    expected_keywords: list[str]


def load_cases(path: str) -> list[EvalCase]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [EvalCase(**c) for c in raw]


def evaluate(engine: ClauseIQ, cases: list[EvalCase], k: int = 5) -> dict:
    hits, coverage, grounded = [], [], []
    valid_doc_ids = {c.doc_id for c in engine.store.chunks}
    for case in cases:
        resp = engine.query(case.question, top_k=k)
        retrieved_docs = {c.doc_id for c in resp.citations}
        hits.append(1.0 if case.expected_doc_id in retrieved_docs else 0.0)
        ans = resp.answer.lower()
        present = sum(1 for kw in case.expected_keywords if kw.lower() in ans)
        coverage.append(present / max(1, len(case.expected_keywords)))
        if resp.citations:
            grounded.append(
                sum(1 for c in resp.citations if c.doc_id in valid_doc_ids)
                / len(resp.citations)
            )
        else:
            grounded.append(0.0)

    def avg(xs):
        return round(sum(xs) / max(1, len(xs)), 4)

    return {
        "cases": len(cases),
        "retrieval_hit@k": avg(hits),
        "answer_coverage": avg(coverage),
        "grounding": avg(grounded),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ClauseIQ RAG evaluation.")
    parser.add_argument("--cases", default="data/eval_cases.json")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    eng = ClauseIQ()
    stats = eng.ingest_dir()
    print("Ingested:", stats)
    results = evaluate(eng, load_cases(args.cases), k=args.k)
    print(json.dumps(results, indent=2))
