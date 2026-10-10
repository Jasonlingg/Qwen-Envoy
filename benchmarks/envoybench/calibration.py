"""Prepare a small anonymous human calibration packet without model calls.

Sampling covers answerability and prior verdict strata. Prior verdicts and
model identities are omitted from the reviewer packet. This is a calibration
sample, not a replacement score or a shortcut through the promotion gate.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import defaultdict
from pathlib import Path

from benchmarks.envoybench.diagnostics import known_source_id
from benchmarks.envoybench.frozen_dataset import DEFAULT_DATASET, load_split
from benchmarks.envoybench.judge import ROW_FIELDS
from src.eval.artifacts import configuration_hash
from src.eval.research_review import load_corpus


def prepare_sample(
    review: dict, benchmark: dict, documents: dict[str, dict], *,
    seed: int = 42, question_count: int = 8,
) -> dict:
    if (review.get("status") != "complete"
            or review.get("reviewer_kind") != "model_assisted"
            or review.get("benchmark_hash") != configuration_hash(benchmark)
            or review.get("corpus_hash") != benchmark["corpus_hash"]):
        raise ValueError("calibration needs a completed matching model-assisted review")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in review["rows"]:
        if row.get("verdict") not in {"pass", "partial", "fail"}:
            raise ValueError("model review has an incomplete verdict")
        grouped[row["question_id"]].append(row)
    questions = {q["id"]: q for q in benchmark["questions"]}
    if set(grouped) != set(questions) or not 1 <= question_count <= len(grouped):
        raise ValueError("sample count or full review question coverage is invalid")
    strata: dict[tuple[str, bool], list[str]] = defaultdict(list)
    for qid, rows in grouped.items():
        strata[(questions[qid]["expected_answerability"], any(
            row["verdict"] == "pass" for row in rows
        ))].append(qid)
    rng = random.Random(seed)
    buckets = [sorted(strata[key]) for key in sorted(strata)]
    for bucket in buckets:
        rng.shuffle(bucket)
    chosen: list[str] = []
    while len(chosen) < question_count:
        for bucket in buckets:
            if bucket and len(chosen) < question_count:
                chosen.append(bucket.pop())
    rng.shuffle(chosen)
    samples = []
    for qid in chosen:
        question = questions[qid]
        doc = documents[known_source_id(question, documents)]
        rows = []
        for original in grouped[qid]:
            row = {field: copy.deepcopy(original[field]) for field in ROW_FIELDS}
            row.update(verdict=None, notes="", relevant_source_passage="")
            rows.append(row)
        samples.append({
            "question_id": qid, "question": question["question"],
            "source": {key: doc.get(key, "") for key in ("doc_id", "title", "text")},
            "answers": rows,
            "reference_decision": None, "reference_notes": "",
        })
    return {
        "schema_version": "envoybench-calibration-packet-v1", "status": "incomplete",
        "reviewer_kind": None, "reviewer_id": "", "reviewed_at": "",
        "benchmark_hash": configuration_hash(benchmark),
        "corpus_hash": benchmark["corpus_hash"],
        "results_hash": review["results_hash"], "source_review_hash": configuration_hash(review),
        "seed": seed, "selected_question_ids": chosen,
        "sampling_method": (
            "Seeded round-robin across answerability and whether either prior model verdict "
            "passed. Stratum membership, prior grades, and model identities are hidden. "
            "This sample is not population-representative and cannot produce a promotion score."
        ),
        "instructions": (
            "A person independently checks the full paper and records reference_valid, "
            "unscorable, or unsure with a reason. For each anonymous answer record pass, "
            "partial, fail, or unsure with a reason and source passage. Correct substantive "
            "answers must be supported by their own submitted evidence. Mark yourself as "
            "human only if you personally performed the review. AI assistance must be "
            "disclosed as model_assisted. Leave uncertainty visible; do not guess."
        ),
        "questions": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split", choices=["dev", "test_candidate"], default="test_candidate")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--questions", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    benchmark, _, _, corpus_dir, _ = load_split(args.dataset, args.split)
    packet = prepare_sample(
        json.loads(args.review.read_text()), benchmark, load_corpus(corpus_dir),
        seed=args.seed, question_count=args.questions,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(packet, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(f"Prepared {len(packet['questions'])} question pairs; no human decisions recorded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
