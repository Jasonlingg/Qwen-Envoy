#!/usr/bin/env python3
"""Prepare, execute once, or score one bounded development-only hosted judge check."""
import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.eval.research_judge_plan import CATALOG, prepare  # noqa: E402
from src.eval.research_judge_run import bounded_request, run, write_new  # noqa: E402
from src.eval.research_judge_score import score_saved  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "score"))
    parser.add_argument("--directory", type=Path,
                        default=REPO / "examples/research-judge-super-check/run")
    parser.add_argument("--fixtures", type=Path,
                        default=REPO / "benchmarks/research_judge_checks_v1")
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare(args.fixtures, args.directory, bounded_request(CATALOG), repo=REPO)
        print(json.dumps({k: result[k] for k in (
            "status", "input_tokens_reserved", "maximum_catalog_cost_usd", "limits")}, indent=2))
    elif args.action == "run":
        from dotenv import load_dotenv
        load_dotenv(REPO / ".env")
        print(json.dumps(run(args.directory, args.fixtures, REPO,
                             os.environ.get("NEBIUS_API_KEY")), indent=2))
    else:
        result = score_saved(args.directory, args.fixtures, REPO)
        write_new(args.directory / "comparison.json", result)
        print(json.dumps({k: v for k, v in result.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
