"""Freeze annotation-rich QASPER training tasks without teacher API calls."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.artifacts import content_hash
from src.eval.sft_quality import known_paper_id
from src.research.qasper import (QASPER_REVISION, _paper_document, _records,
                                 _convert_question, _to_code_exec_question)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def convert(rows, split, excluded_docs=(), excluded_questions=()):
    documents, questions, rejected = {}, [], Counter()
    for row in rows:
        doc, locations = _paper_document(row, split)
        if doc["doc_id"] in excluded_docs:
            rejected["reserved_paper"] += 1
            continue
        documents[doc["doc_id"]] = doc
        for qa in _records(row.get("qas")):
            q = _convert_question(qa, doc, locations, split)
            if q["id"] in excluded_questions:
                rejected["previously_quarantined"] += 1
                continue
            if q["conversion_issues"] or q["ambiguous_text_evidence_count"]:
                rejected["evidence_mapping"] += 1
                continue
            mapped = _to_code_exec_question(q)
            if mapped is None:
                rejected["annotator_disagreement"] += 1
                continue
            if (split == "train" and q["expected_answerability"] == "sufficient" and
                    any(not a["evidence"] for a in q["answer_annotations"])):
                rejected["missing_gold_evidence"] += 1
                continue
            mapped.update(answer_annotations=q["answer_annotations"],
                          target_doc_ids=q["target_doc_ids"], source_split=split)
            questions.append(mapped)
    return documents, questions, dict(rejected)


def main():
    from datasets import Dataset
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--local-arrow", type=Path, required=True)
    p.add_argument("--split", choices=["train", "validation", "test"], default="train")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--exclude-benchmark", type=Path, action="append", default=[])
    p.add_argument("--exclude-sft", type=Path, action="append", default=[])
    p.add_argument("--review", type=Path, action="append", default=[])
    p.add_argument("--select-benchmark", type=Path)
    p.add_argument(
        "--answerability",
        choices=["all", "sufficient", "insufficient"],
        default="all",
        help="Restrict selection by expert answerability label (default: all).",
    )
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--seed", type=int, default=20260918)
    args = p.parse_args()
    excluded_docs, excluded_qs = set(), set()
    inputs = []
    for path in args.exclude_benchmark:
        b = json.loads(path.read_text())
        excluded_docs.update(b.get("reserved_doc_ids", []))
        excluded_docs.update(known_paper_id(q["question"]) for q in b["questions"])
        excluded_qs.update(q["id"] for q in b["questions"])
        inputs.append({"path": str(path), "sha256": sha(path)})
    for path in args.exclude_sft:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            for m in row["messages"]:
                if m["role"] == "user":
                    doc_id = known_paper_id(m["content"])
                    if doc_id:
                        excluded_docs.add(doc_id)
                        break
        inputs.append({"path": str(path), "sha256": sha(path)})
    def rejected_ids(value):
        if isinstance(value, dict):
            if value.get("verdict") in {"quarantine", "quarantined", "exclude", "fail", "partial"}:
                excluded_qs.add(value.get("question_id"))
            for child in value.values():
                rejected_ids(child)
        elif isinstance(value, list):
            for child in value:
                rejected_ids(child)
    for path in args.review:
        review = json.loads(path.read_text())
        # Final decisions only; a repaired passing episode can have failed raw reviews.
        rejected_ids(review.get("reviews", review.get("review", {})))
        inputs.append({"path": str(path), "sha256": sha(path)})
    docs, questions, rejected = convert(Dataset.from_file(str(args.local_arrow)), args.split,
                                        excluded_docs, excluded_qs)
    if args.answerability != "all":
        questions = [
            question
            for question in questions
            if question["expected_answerability"] == args.answerability
        ]
    if args.select_benchmark:
        ids = {q["id"] for q in json.loads(args.select_benchmark.read_text())["questions"]}
        questions = [q for q in questions if q["id"] in ids]
        if {q["id"] for q in questions} != ids:
            raise ValueError("Selected benchmark has questions ineligible for this reward")
        inputs.append({"path": str(args.select_benchmark), "sha256": sha(args.select_benchmark)})
    else:
        rng = random.Random(args.seed)
        questions.sort(key=lambda q:q["id"])
        rng.shuffle(questions)
        if args.answerability == "all":
            # Approximately the natural answerability mix; avoid a refusal-heavy batch.
            no = [q for q in questions if q["expected_answerability"] == "insufficient"]
            yes = [q for q in questions if q["expected_answerability"] == "sufficient"]
            count_no = min(len(no), args.limit//5)
            questions = yes[:args.limit-count_no] + no[:count_no]
        else:
            questions = questions[:args.limit]
        rng.shuffle(questions)
    if not questions:
        raise ValueError("No eligible questions")
    args.output.mkdir(parents=True, exist_ok=False)
    corpus = args.output / "corpus"
    corpus.mkdir()
    for doc in docs.values():
        (corpus / (doc["doc_id"]+".json")).write_text(json.dumps(doc, ensure_ascii=False)+"\n")
    manifest = dict(schema_version="qasper-rl-v1", source_split=args.split,
                    source_revision=QASPER_REVISION, local_arrow_sha256=sha(args.local_arrow),
                    corpus_hash=content_hash(corpus), selection_seed=args.seed,
                    answerability_filter=args.answerability,
                    reserved_doc_ids=sorted(docs), question_ids=[q["id"] for q in questions],
                    counts=dict(Counter(q["expected_answerability"] for q in questions)),
                    rejected=rejected, exclusions=inputs,
                    excluded_doc_ids=sorted(excluded_docs),
                    excluded_question_ids=sorted(x for x in excluded_qs if x),
                    questions=questions)
    (args.output/"benchmark.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({k:v for k,v in manifest.items() if k in
                      {"counts", "rejected", "corpus_hash"}}, indent=2))


if __name__ == "__main__":
    main()
