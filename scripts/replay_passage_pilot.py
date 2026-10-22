"""Replay approved passage demonstrations in Docker or export reviewed train-only data."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.passage_pilot_replay import export, replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["replay", "export"])
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; refusing to overwrite")
    if args.phase == "replay":
        result = replay(args.prepared, args.review, args.output)
        print(json.dumps({"status": result["status"], "cases": len(result["cases"]),
                          "new_model_calls": result["new_model_calls"]}))
    else:
        if args.replay_dir is None:
            parser.error("export requires --replay-dir")
        print(json.dumps(export(args.prepared, args.review, args.replay_dir, args.output)))


if __name__ == "__main__":
    main()
