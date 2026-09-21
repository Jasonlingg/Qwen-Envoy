"""Build a blind SFT-vs-GRPO semantic review from QASPER eval episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

from src.eval.qasper_reward import parse_strict


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_corpus(path: Path) -> dict[str, dict]:
    documents = {}
    for item in path.glob("*.json"):
        document = json.loads(item.read_text())
        documents[document["doc_id"]] = document
    return documents


def materialize(spans: list[dict], documents: dict[str, dict]) -> list[dict]:
    result = []
    for span in spans:
        doc_id, start, end = span.get("doc_id"), span.get("start"), span.get("end")
        document = documents.get(doc_id)
        valid = bool(
            document is not None
            and type(start) is int
            and type(end) is int
            and 0 <= start < end <= len(document["text"])
        )
        item = {"doc_id": doc_id, "start": start, "end": end, "valid": valid}
        if valid:
            item["quote"] = document["text"][start:end]
        result.append(item)
    return result


def candidate(row: dict, documents: dict[str, dict]) -> dict:
    submissions = [
        step.get("action", "")
        for step in row.get("trajectory", [])
        if str(step.get("action", "")).upper().startswith("SUBMIT:")
    ]
    raw = str(submissions[-1]) if submissions else ""
    parsed = parse_strict(raw)
    if parsed is None:
        answer, citations, evidence = "", [], []
    else:
        answer, citations, evidence = parsed
    return {
        "answer": answer,
        "citations": citations,
        "evidence": materialize(evidence, documents),
        "raw_submission": raw if parsed is None else None,
        "finish": row.get("finish"),
    }


def references(question: dict, documents: dict[str, dict]) -> list[dict]:
    result = []
    for annotation in question["answer_annotations"]:
        result.append(
            {
                "unanswerable": annotation["unanswerable"],
                "answer_type": annotation["answer_type"],
                "answer": annotation["answer_text"],
                "evidence": materialize(annotation["evidence"], documents),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--sft", type=Path, required=True)
    parser.add_argument("--rl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    questions = json.loads(args.benchmark.read_text())["questions"]
    documents = load_corpus(args.corpus)
    sft = {row["question_id"]: row for row in load_jsonl(args.sft)}
    rl = {row["question_id"]: row for row in load_jsonl(args.rl)}
    expected = [question["id"] for question in questions]
    if list(sft) != expected or list(rl) != expected:
        raise ValueError("SFT and RL arms must match the ordered frozen question set")

    rng = random.Random(args.seed)
    review, key = [], []
    counter = 1
    for question in questions:
        systems = ["sft", "rl-5"]
        rng.shuffle(systems)
        candidates = []
        for system in systems:
            blind_id = f"R{counter:03d}"
            counter += 1
            row = sft[question["id"]] if system == "sft" else rl[question["id"]]
            item = {"blind_id": blind_id, **candidate(row, documents)}
            candidates.append(item)
            key.append(
                {"blind_id": blind_id, "question_id": question["id"], "system": system}
            )
        review.append(
            {
                "question_id": question["id"],
                "question": question["question"],
                "expected_answerability": question["expected_answerability"],
                "references": references(question, documents),
                "candidates": candidates,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "review.json").write_text(
        json.dumps({"schema_version": "qasper-grpo-blind-review-v1", "rows": review}, indent=2)
        + "\n"
    )
    (args.output_dir / "blind-key.json").write_text(
        json.dumps(
            {"schema_version": "qasper-grpo-blind-key-v1", "seed": args.seed, "rows": key},
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"questions": len(review), "candidates": len(key)}, indent=2))


if __name__ == "__main__":
    main()
