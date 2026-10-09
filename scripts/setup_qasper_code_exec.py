"""Download a pinned QASPER split and build a code-execution-protocol benchmark.

Unlike setup_qasper.py (which targets the paused JSON-action protocol in
src/research/), this produces a benchmark.json + corpus that
scripts/run_eval.py and scripts/research_benchmark.py can run and validate
directly, using the same search()/read()/passage() tools as the MuSiQue and
AI-paper pilots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.sft_quality import known_paper_id
from src.research.qasper import (
    QASPER_CONFIG,
    QASPER_DATASET,
    QASPER_REVISION,
    _doc_id,
    build_qasper_code_exec_benchmark,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--num-questions", type=int, default=20)
    parser.add_argument("--min-insufficient", type=int, default=None,
                         help="Oversample this many 'insufficient' (unanswerable) questions "
                              "above QASPER's natural ~16%% rate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--revision", default=QASPER_REVISION)
    parser.add_argument("--local-arrow", type=Path,
                        help="Use an already cached official split; record its SHA-256")
    parser.add_argument("--exclude-benchmark", type=Path, action="append", default=[],
                        help="Exclude papers targeted by prior question sets (repeatable)")
    parser.add_argument(
        "--output", type=Path, required=True, help="New immutable benchmark directory"
    )
    args = parser.parse_args()
    try:
        from datasets import Dataset, load_dataset
    except ImportError as exc:
        parser.error("Install the QASPER dependency with: pip install -e '.[qasper]'")
        raise AssertionError from exc

    rows = Dataset.from_file(str(args.local_arrow)) if args.local_arrow else load_dataset(
        QASPER_DATASET,
        QASPER_CONFIG,
        revision=args.revision,
        split=args.split,
    )
    excluded_docs = set()
    exclusions = []
    for path in args.exclude_benchmark:
        source = json.loads(path.read_text())
        excluded_docs.update(known_paper_id(q["question"]) for q in source["questions"])
        exclusions.append({"path": str(path),
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    if excluded_docs:
        # Paper-level exclusion also removes sibling questions and corpus distractors.
        rows = [row for row in rows
                if _doc_id(row["id"]) not in excluded_docs]
    manifest, benchmark = build_qasper_code_exec_benchmark(
        rows,
        args.output,
        source_split=args.split,
        revision=args.revision,
        num_questions=args.num_questions,
        seed=args.seed,
        min_insufficient=args.min_insufficient,
    )
    selection = {
        "source_split": args.split, "source_revision": args.revision,
        "local_arrow_sha256": (hashlib.sha256(args.local_arrow.read_bytes()).hexdigest()
                               if args.local_arrow else None),
        "exclusions": exclusions, "excluded_doc_ids": sorted(excluded_docs),
        "question_ids": [q["id"] for q in benchmark["questions"]],
        "corpus_hash": manifest["corpus_hash"], "seed": args.seed,
        "benchmark_sha256": hashlib.sha256(
            (args.output / "benchmark.json").read_bytes()
        ).hexdigest(),
    }
    (args.output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "paper_count": manifest["paper_count"],
                "selected_question_count": len(benchmark["questions"]),
                "corpus_hash": manifest["corpus_hash"],
                "benchmark": str(args.output / "benchmark.json"),
                "corpus": str(args.output / "corpus"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
