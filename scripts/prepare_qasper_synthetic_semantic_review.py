"""Prepare a source-visible review of synthetic QASPER training trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.hashing import known_doc_id
from src.eval.qasper_reward import parse_strict
from src.eval.sft_quality import trajectory_hash

STOP_WORDS = set(
    "use known paper doc id to answer what which who when where why how do does did is are was "
    "were the a an of in on for from with and or their they this that these those it its as by be "
    "been being have has had can could would should may might about into than then there any all "
    "part based study authors".split()
)


def question_terms(question: str) -> list[str]:
    prompt = question.split("to answer:", 1)[-1].lower()
    return list(
        dict.fromkeys(
            token
            for token in re.findall(r"[a-z0-9][a-z0-9_-]*", prompt)
            if token not in STOP_WORDS and len(token) > 2
        )
    )


def full_text_windows(text: str, question: str, count: int = 6) -> list[dict]:
    terms = question_terms(question)
    windows = []
    for start in range(0, len(text), 400):
        end = min(start + 900, len(text))
        excerpt = text[start:end]
        lowered = excerpt.lower()
        matched = [term for term in terms if term in lowered]
        score = sum(lowered.count(term) for term in terms)
        if score:
            windows.append(
                {"start": start, "end": end, "score": score, "matched_terms": matched, "text": excerpt}
            )
    windows.sort(key=lambda item: (-item["score"], item["start"]))
    return windows[:count]


def evidence_quotes(evidence: list[dict], documents: dict[str, dict]) -> list[dict]:
    result = []
    for item in evidence:
        doc = documents[item["doc_id"]]
        result.append({**item, "text": doc["text"][item["start"]:item["end"]]})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--answerable-sample", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()

    benchmark = json.loads(args.benchmark.read_text())
    questions = {item["id"]: item for item in benchmark["questions"]}
    rows = [json.loads(line) for line in args.candidates.read_text().splitlines() if line.strip()]
    documents = {
        item["doc_id"]: item
        for path in args.corpus.glob("*.json")
        for item in [json.loads(path.read_text())]
    }
    insufficient = [row for row in rows if row["expected_answerability"] == "insufficient"]
    sufficient_by_behavior = defaultdict(list)
    for row in rows:
        if row["expected_answerability"] == "sufficient":
            sufficient_by_behavior[row.get("teacher_behavior", "unknown")].append(row)
    rng = random.Random(args.seed)
    for group in sufficient_by_behavior.values():
        group.sort(key=lambda row: row["question_id"])
        rng.shuffle(group)
    available = sum(len(group) for group in sufficient_by_behavior.values())
    target = min(args.answerable_sample, available)
    sufficient = []
    labels = sorted(sufficient_by_behavior)
    while len(sufficient) < target:
        advanced = False
        for label in labels:
            group = sufficient_by_behavior[label]
            if group and len(sufficient) < target:
                sufficient.append(group.pop())
                advanced = True
        if not advanced:
            break
    selected = sorted(insufficient + sufficient, key=lambda row: row["question_id"])

    review_rows = []
    for row in selected:
        question = questions[row["question_id"]]
        parsed = parse_strict(row["trajectory"][-1]["action"])
        if parsed is None:
            raise ValueError(f"Invalid saved submission: {row['question_id']}")
        answer, citations, evidence = parsed
        doc_id = known_doc_id(question["question"])
        review_rows.append(
            {
                "question_id": row["question_id"],
                "trajectory_sha256": trajectory_hash(row),
                "expected_answerability": question["expected_answerability"],
                "teacher_behavior": row.get("teacher_behavior"),
                "question": question["question"],
                "submitted_answer": answer,
                "submitted_citations": citations,
                "submitted_evidence": evidence_quotes(evidence, documents),
                "official_annotations": question["answer_annotations"],
                "search_actions": [
                    step["action"]
                    for step in row["trajectory"][:-1]
                    if "search_within" in step["action"]
                ],
                "independent_full_text_windows": full_text_windows(
                    documents[doc_id]["text"], question["question"]
                ),
                "document": {
                    "doc_id": doc_id,
                    "title": documents[doc_id].get("title"),
                    "characters": len(documents[doc_id]["text"]),
                    "sha256": hashlib.sha256(documents[doc_id]["text"].encode()).hexdigest(),
                },
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "schema_version": "qasper-synthetic-semantic-review-v1",
                "seed": args.seed,
                "answerable_sample_target": target,
                "criteria": {
                    "sufficient": (
                        "Pass only when the answer addresses the question and every major claim is "
                        "supported by the exact submitted evidence."
                    ),
                    "insufficient": (
                        "Pass only when the official unanswerable label is plausible after inspecting "
                        "independent full-text windows; reject or mark ambiguous when the paper appears "
                        "to answer the question."
                    ),
                },
                "rows": review_rows,
            },
            indent=2,
        )
        + "\n"
    )
    template = {
        "schema_version": "qasper-synthetic-semantic-judgments-v1",
        "reviewer": "",
        "reviewer_type": "",
        "reviews": [
            {
                "question_id": item["question_id"],
                "trajectory_sha256": item["trajectory_sha256"],
                "verdict": "",
                "notes": "",
            }
            for item in review_rows
        ],
    }
    args.output.with_name("semantic-judgments-template.json").write_text(
        json.dumps(template, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "questions": len(review_rows),
                "unanswerable": len(insufficient),
                "answerable_sample": len(sufficient),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
