"""Build an identity-blind semantic review packet for two or more QASPER systems.

Generalizes scripts/prepare_qasper_grpo_semantic_review.py, which is fixed to an
SFT-versus-RL pair, so Claude arms can be judged under the same rubric and packet
format. Candidate order is shuffled per question, and every blind ID resolves only
through the separate key file.

Emits the same `qasper-grpo-blind-review-v1` row shape, so the existing judgment and
scoring flow keeps working.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_qasper_grpo_semantic_review import candidate, load_corpus, load_jsonl, references


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--system", action="append", required=True, metavar="LABEL=PATH",
                        help="Repeatable. Label is hidden from the review packet.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    questions = json.loads(args.benchmark.read_text())["questions"]
    documents = load_corpus(args.corpus)

    systems: dict[str, dict[str, dict]] = {}
    for item in args.system:
        label, _, path = item.partition("=")
        if label in systems:
            raise ValueError(f"Duplicate system label: {label}")
        systems[label] = {row["question_id"]: row for row in load_jsonl(Path(path))}
    if len(systems) < 2:
        raise ValueError("At least two systems are required for a paired review")

    expected = [question["id"] for question in questions]
    for label, rows in systems.items():
        missing = [qid for qid in expected if qid not in rows]
        if missing:
            raise ValueError(f"{label} is missing {len(missing)} question(s), e.g. {missing[0]}")

    rng = random.Random(args.seed)
    review, key = [], []
    counter = 1
    for question in questions:
        labels = list(systems)
        rng.shuffle(labels)
        candidates = []
        for label in labels:
            blind_id = f"R{counter:03d}"
            counter += 1
            candidates.append({"blind_id": blind_id,
                               **candidate(systems[label][question["id"]], documents)})
            key.append({"blind_id": blind_id, "question_id": question["id"], "system": label})
        review.append({
            "question_id": question["id"],
            "question": question["question"],
            "expected_answerability": question["expected_answerability"],
            "references": references(question, documents),
            "candidates": candidates,
        })

    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "review.json").write_text(
        json.dumps({"schema_version": "qasper-grpo-blind-review-v1", "rows": review}, indent=2) + "\n")
    (args.output_dir / "blind-key.json").write_text(
        json.dumps({"schema_version": "qasper-grpo-blind-key-v1", "seed": args.seed,
                    "systems": sorted(systems), "rows": key}, indent=2) + "\n")
    print(json.dumps({"questions": len(review), "systems": sorted(systems),
                      "candidates": len(key)}, indent=2))


if __name__ == "__main__":
    main()
