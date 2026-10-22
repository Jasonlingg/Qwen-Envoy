"""Replay original traces, then reviewed corrections, using Docker and local replies."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.reader_correction_replay import replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "queue", "review", "drafts", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        result = replay(args.run, args.queue, args.review, args.drafts, args.output)
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
        parser.exit(2, f"Replay failed; no model request was made: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
