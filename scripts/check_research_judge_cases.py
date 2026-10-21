#!/usr/bin/env python3
"""Check local fixture integrity; optional exclusive export, no model invocation."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.research_judge_checks import check_bundle, judge_inputs  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1]
                        / "benchmarks/research_judge_checks_v1")
    parser.add_argument("--export-judge-inputs", type=Path)
    args = parser.parse_args()
    result = check_bundle(args.root)
    if args.export_judge_inputs:
        payload = json.dumps(judge_inputs(args.root), ensure_ascii=False, indent=2) + "\n"
        with args.export_judge_inputs.open("x") as output:
            output.write(payload)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
