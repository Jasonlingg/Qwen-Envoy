"""Filter a QASPER synthetic-data pilot using recorded semantic judgments.

The source files are never modified. Rows judged ``reject`` or ``ambiguous`` are
removed from candidates and from their existing paper-disjoint train/validation
split. Unreviewed answerable rows remain eligible for the independent mechanical
audit; the reliability verifier separately enforces the semantic-review coverage
required to open the scaling gate.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.eval.hashing import known_doc_id, sha256


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def conversation_question(row: dict[str, Any]) -> str:
    messages = row.get("messages", [])
    if len(messages) < 2 or messages[1].get("role") != "user":
        raise ValueError("SFT row has no user question at messages[1]")
    return str(messages[1]["content"]).removeprefix("Question: ").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--semantic-review", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    benchmark = json.loads(args.benchmark.read_text())
    questions = {item["id"]: item for item in benchmark["questions"]}
    question_ids_by_text = {item["question"].strip(): item["id"] for item in benchmark["questions"]}
    review_data = json.loads(args.semantic_review.read_text())
    reviews = review_data.get("reviews", [])
    review_ids = [item.get("question_id") for item in reviews]
    duplicates = [item for item, count in Counter(review_ids).items() if item and count > 1]
    if duplicates:
        raise ValueError(f"Duplicate semantic review IDs: {sorted(duplicates)}")
    allowed = {"pass", "reject", "ambiguous"}
    if any(item.get("verdict") not in allowed for item in reviews):
        raise ValueError("Every semantic review must have pass, reject, or ambiguous verdict")
    rejected = {
        item["question_id"]: item
        for item in reviews
        if item["verdict"] in {"reject", "ambiguous"}
    }

    candidates = load_jsonl(args.candidates)
    candidate_ids = {row["question_id"] for row in candidates}
    unknown_reviews = sorted(set(rejected) - candidate_ids)
    if unknown_reviews:
        raise ValueError(f"Rejected reviews are absent from candidates: {unknown_reviews}")
    accepted_candidates = [row for row in candidates if row["question_id"] not in rejected]
    accepted_ids = {row["question_id"] for row in accepted_candidates}

    def filter_split(path: Path) -> list[dict[str, Any]]:
        result = []
        for row in load_jsonl(path):
            question_text = conversation_question(row)
            question_id = question_ids_by_text.get(question_text)
            if question_id is None:
                raise ValueError(f"Split contains an unknown question: {question_text}")
            if question_id in accepted_ids:
                result.append(row)
        return result

    accepted_train = filter_split(args.train)
    accepted_validation = filter_split(args.validation)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = args.output_dir / "candidates.jsonl"
    train_path = args.output_dir / "train.jsonl"
    validation_path = args.output_dir / "val.jsonl"
    write_jsonl(candidates_path, accepted_candidates)
    write_jsonl(train_path, accepted_train)
    write_jsonl(validation_path, accepted_validation)

    expected_split_count = len(accepted_train) + len(accepted_validation)
    if expected_split_count != len(accepted_candidates):
        raise ValueError(
            f"Filtered splits contain {expected_split_count} rows for "
            f"{len(accepted_candidates)} candidates"
        )
    if not accepted_train or not accepted_validation:
        raise ValueError("Filtering emptied a train or validation split")

    accepted_answerability = Counter(
        questions[row["question_id"]]["expected_answerability"] for row in accepted_candidates
    )
    accepted_behaviors = Counter(row.get("teacher_behavior", "unknown") for row in accepted_candidates)
    manifest = {
        "schema_version": "qasper-synthetic-filter-v1",
        "builder": "scripts/filter_qasper_synthetic_data.py",
        "source": {
            "candidates": str(args.candidates),
            "candidates_sha256": sha256(args.candidates),
            "train": str(args.train),
            "train_sha256": sha256(args.train),
            "validation": str(args.validation),
            "validation_sha256": sha256(args.validation),
            "benchmark": str(args.benchmark),
            "benchmark_sha256": sha256(args.benchmark),
            "semantic_review": str(args.semantic_review),
            "semantic_review_sha256": sha256(args.semantic_review),
        },
        "reviewer": review_data.get("reviewer"),
        "reviewer_type": review_data.get("reviewer_type"),
        "accepted": len(accepted_candidates),
        "excluded": len(rejected),
        "excluded_rows": [
            {
                "question_id": question_id,
                "doc_id": known_doc_id(questions[question_id]["question"]),
                "verdict": review["verdict"],
                "notes": review.get("notes", ""),
            }
            for question_id, review in sorted(rejected.items())
        ],
        "answerability": dict(sorted(accepted_answerability.items())),
        "behaviors": dict(sorted(accepted_behaviors.items())),
        "splits": {
            "train": {"count": len(accepted_train), "sha256": sha256(train_path)},
            "validation": {
                "count": len(accepted_validation),
                "sha256": sha256(validation_path),
            },
        },
        "claim_boundary": (
            "Rows passed a source-visible model-assisted semantic audit; this is not an "
            "independent human annotation study. The separate verifier must pass before scaling."
        ),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "accepted": manifest["accepted"],
                "excluded": manifest["excluded"],
                "train": len(accepted_train),
                "validation": len(accepted_validation),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
