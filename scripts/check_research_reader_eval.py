#!/usr/bin/env python3
"""Check the frozen reader evaluation without network, models or GPU access."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.reader_eval_bundle import check_bundle  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[1]
                        / "benchmarks/research_reader_v1")
    args = parser.parse_args()
    try:
        print(json.dumps(check_bundle(args.root), indent=2))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Invalid evaluation bundle: {exc}\n")


if __name__ == "__main__":
    main()
