"""Command line interface: ingest a folder, ask a question, or run eval."""
from __future__ import annotations

import argparse
import json

from .eval.harness import evaluate, load_cases
from .pipeline import ClauseIQ


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="clauseiq")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ask = sub.add_parser("ask", help="Ingest data dir and answer a question.")
    p_ask.add_argument("question")
    p_ask.add_argument("--data", default=None)
    p_ask.add_argument("--no-agent", action="store_true")

    p_eval = sub.add_parser("eval", help="Run the evaluation harness.")
    p_eval.add_argument("--cases", default="data/eval_cases.json")
    p_eval.add_argument("--data", default=None)

    args = parser.parse_args(argv)
    eng = ClauseIQ()
    stats = eng.ingest_dir(args.data)

    if args.cmd == "ask":
        resp = eng.query(args.question, use_agent=not args.no_agent)
        print(json.dumps(resp.model_dump(), indent=2))
    elif args.cmd == "eval":
        print("Ingested:", stats)
        print(json.dumps(evaluate(eng, load_cases(args.cases)), indent=2))


if __name__ == "__main__":
    main()
