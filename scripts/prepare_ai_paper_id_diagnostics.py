"""Build oracle-ID and retrieved-ID variants of the frozen AI-paper pilot."""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections import defaultdict
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.eval.research_review import load_corpus


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def build_passages(
    documents: dict[str, dict], chunk_size: int = 512, overlap: int = 64
) -> list[dict]:
    passages = []
    step = chunk_size - overlap
    for doc_id, document in sorted(documents.items()):
        title = document.get("title", doc_id)
        text = document["text"]
        for chunk_id, start in enumerate(range(0, len(text), step)):
            passages.append({
                "doc_id": doc_id,
                "title": title,
                "chunk_id": chunk_id,
                "text": text[start : start + chunk_size],
            })
    return passages


def retrieve_doc_ids(
    question: str, passages: list[dict], index: BM25Okapi, top_docs: int
) -> list[str]:
    scores = index.get_scores(tokenize(question))
    best_by_doc: dict[str, float] = defaultdict(lambda: float("-inf"))
    for passage, score in zip(passages, scores):
        best_by_doc[passage["doc_id"]] = max(best_by_doc[passage["doc_id"]], float(score))
    return sorted(best_by_doc, key=lambda doc_id: (-best_by_doc[doc_id], doc_id))[:top_docs]


def add_candidates(question: dict, candidate_ids: list[str], documents: dict[str, dict]) -> dict:
    result = copy.deepcopy(question)
    original = result["question"]
    candidates = "\n".join(
        f'- doc_id: "{doc_id}"; title: "{documents[doc_id].get("title", doc_id)}"'
        for doc_id in candidate_ids
    )
    result["question_without_candidates"] = original
    result["candidate_doc_ids"] = candidate_ids
    result["question"] = (
        "Candidate papers have already been selected for this episode. Use these exact doc_id "
        "values with read(), passage(), or search_within(); do not invent IDs. You may call "
        "search() if the candidates are insufficient.\n"
        f"{candidates}\n\nResearch question: {original}"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-docs", type=int, default=4)
    args = parser.parse_args()
    if args.top_docs < 1:
        parser.error("--top-docs must be positive")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")

    benchmark = json.loads(args.questions.read_text())
    questions = benchmark["questions"]
    documents = load_corpus(args.corpus)
    passages = build_passages(documents)
    index = BM25Okapi([
        tokenize(f'{item["title"]} {item["title"]} {item["text"]}') for item in passages
    ])

    oracle_questions = []
    retrieved_questions = []
    audit = []
    all_doc_ids = sorted(documents)
    for question in questions:
        required = list(question.get("required_doc_ids", []))
        missing = set(required) - set(documents)
        if missing:
            raise ValueError(f'{question["id"]} references missing documents: {sorted(missing)}')
        retrieved = retrieve_doc_ids(question["question"], passages, index, args.top_docs)
        # An unanswerable snapshot-level question has no gold document. Supplying the complete
        # six-paper catalog keeps the oracle diagnostic focused on reasoning rather than discovery.
        oracle = required or all_doc_ids
        oracle_questions.append(add_candidates(question, oracle, documents))
        retrieved_questions.append(add_candidates(question, retrieved, documents))
        audit.append({
            "question_id": question["id"],
            "required_doc_ids": required,
            "oracle_doc_ids": oracle,
            "retrieved_doc_ids": retrieved,
            "retrieved_required_hits": sorted(set(required) & set(retrieved)),
        })

    args.output.mkdir(parents=True)
    shared = {
        key: value for key, value in benchmark.items() if key != "questions"
    }
    (args.output / "oracle_ids.json").write_text(json.dumps({
        **shared,
        "benchmark_id": f'{benchmark.get("benchmark_id", "ai-paper-pilot")}-oracle-ids',
        "diagnostic": "Ground-truth document IDs; not a deployable product score.",
        "questions": oracle_questions,
    }, indent=2) + "\n")
    (args.output / "retrieved_ids.json").write_text(json.dumps({
        **shared,
        "benchmark_id": f'{benchmark.get("benchmark_id", "ai-paper-pilot")}-retrieved-ids',
        "diagnostic": "BM25 candidate document IDs with no answer-label access.",
        "retrieval": {
            "method": "BM25 over 512-character passages with 64-character overlap and title boost",
            "top_docs": args.top_docs,
        },
        "questions": retrieved_questions,
    }, indent=2) + "\n")
    required_total = sum(len(row["required_doc_ids"]) for row in audit)
    required_hits = sum(len(row["retrieved_required_hits"]) for row in audit)
    (args.output / "retrieval_audit.json").write_text(json.dumps({
        "top_docs": args.top_docs,
        "required_document_recall": required_hits / required_total if required_total else 1.0,
        "rows": audit,
    }, indent=2) + "\n")
    print(f"Wrote diagnostics to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
