"""Inventory prior exposure and prepare source-review drafts without inference."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.passage_pilot_data import inventory, prepare
from src.eval.reader_improvement import write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["inventory", "prepare"])
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--rejections", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; refusing to overwrite")
    if args.phase == "inventory":
        registry = inventory()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write(args.output, registry)
        print(json.dumps({"files": len(registry["files"]),
                          "excluded_papers": len(registry["excluded_doc_ids"]),
                          "excluded_questions": len(registry["excluded_question_ids"])}))
    else:
        if not args.registry:
            parser.error("prepare requires --registry")
        print(json.dumps(prepare(args.registry, args.output, args.rejections)))


if __name__ == "__main__":
    main()
