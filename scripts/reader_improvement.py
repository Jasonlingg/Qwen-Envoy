"""Prepare and check local failure reviews; never dispatch a model or export SFT."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.reader_improvement import check, prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "check"):
        command = sub.add_parser(name)
        command.add_argument("--run", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if name == "check":
            command.add_argument("--queue", type=Path, required=True)
            command.add_argument("--review", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = (prepare(args.run, args.output) if args.command == "prepare" else
                  check(args.run, args.queue, args.review, args.output))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f"Review check failed: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
