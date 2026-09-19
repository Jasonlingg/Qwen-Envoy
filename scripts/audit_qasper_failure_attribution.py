#!/usr/bin/env python3
"""Attribute QASPER agent failures to retrieval, search planning, or synthesis.

This is a diagnostic, not an evaluation metric. It uses QASPER gold answers and
evidence to perform counterfactual retrieval probes, so its output must never be
used as model input on a held-out evaluation run.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


STOPWORDS = {
    "about", "after", "again", "against", "also", "among", "because", "before",
    "being", "between", "both", "could", "does", "from", "have", "into", "only",
    "other", "over", "paper", "question", "should", "some", "such", "than", "that",
    "their", "them", "then", "there", "these", "they", "this", "those", "through",
    "under", "using", "very", "were", "what", "when", "where", "which", "while",
    "with", "would", "yes", "from", "used", "use", "model", "models", "results",
    "bibref", "inlineform", "tabref", "figref",
}


def words(text: str) -> list[str]:
    return re.findall(r"[a-z][a-z0-9_-]{2,}", text.lower())


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def span_coverage(
    evidence_spans: list[tuple[int, int]], observed_spans: list[tuple[int, int]]
) -> float:
    """Return the best fraction of any gold span covered by observed windows."""
    if not evidence_spans or not observed_spans:
        return 0.0
    merged = merge_intervals(observed_spans)
    best = 0.0
    for gold_start, gold_end in evidence_spans:
        covered = sum(
            max(0, min(gold_end, end) - max(gold_start, start))
            for start, end in merged
        )
        best = max(best, covered / max(1, gold_end - gold_start))
    return best


def find_evidence_spans(text: str, notes: list[str]) -> list[tuple[int, int]]:
    spans = []
    for note in dict.fromkeys(notes):
        start = text.find(note)
        if start >= 0:
            spans.append((start, start + len(note)))
    return spans


def parse_observation(value: str) -> Any | None:
    payload = value.split("\n\n\n[Step", 1)[0].strip()
    try:
        return ast.literal_eval(payload)
    except (ValueError, SyntaxError):
        return None


def observation_spans(trajectory: list[dict[str, Any]], doc_id: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for step in trajectory:
        if str(step.get("action", "")).startswith("SUBMIT:"):
            continue
        parsed = parse_observation(str(step.get("observation", "")))
        rows = parsed if isinstance(parsed, list) else [parsed]
        for row in rows:
            if not isinstance(row, dict) or "text" not in row:
                continue
            if row.get("doc_id", doc_id) != doc_id:
                continue
            start = row.get("offset", row.get("start"))
            if isinstance(start, int):
                spans.append((start, start + len(str(row["text"]))))
    return spans


def search_queries(trajectory: list[dict[str, Any]], doc_id: str) -> list[str]:
    queries: list[str] = []
    for step in trajectory:
        try:
            tree = ast.parse(str(step.get("action", "")))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "search_within"
                and len(node.args) >= 2
            ):
                continue
            try:
                called_doc = ast.literal_eval(node.args[0])
                query = ast.literal_eval(node.args[1])
            except (ValueError, TypeError):
                continue
            if called_doc == doc_id and isinstance(query, str):
                queries.append(query)
    return queries


def lexical_search(
    text: str,
    query: str,
    *,
    top_k: int = 3,
    window_size: int = 500,
    step: int = 200,
) -> list[tuple[int, int]]:
    query_terms = query.lower().split()
    windows = []
    for start in range(0, len(text), step):
        window = text[start : start + window_size]
        score = sum(window.lower().count(term) for term in query_terms)
        if score > 0:
            windows.append((score, start, min(len(text), start + window_size)))
    windows.sort(key=lambda item: (-item[0], item[1]))
    return [(start, end) for _, start, end in windows[:top_k]]


def oracle_queries(question: dict[str, Any], text: str) -> list[str]:
    """Construct diagnostic queries from gold material, not production queries."""
    answer_terms = [
        token for token in words(question.get("answer", ""))
        if token not in STOPWORDS and not token.isdigit()
    ]
    evidence = " ".join(question.get("grader_notes", [])[:1])
    counts = Counter(words(text))
    evidence_terms = [
        token for token in words(evidence)
        if token not in STOPWORDS and not token.isdigit()
    ]
    # Test both an answer-bearing query and a rare-evidence query. A single
    # combined query can be dominated by generic answer words in this simple
    # count-based retriever and incorrectly make the backend look incapable.
    rare = sorted(dict.fromkeys(evidence_terms), key=lambda token: (counts[token], evidence_terms.index(token)))
    answer_query = " ".join(dict.fromkeys(answer_terms[:8]))
    evidence_query = " ".join(rare[:8])
    return [query for query in dict.fromkeys((answer_query, evidence_query)) if query]


def expanded(spans: list[tuple[int, int]], text_length: int, amount: int = 500) -> list[tuple[int, int]]:
    return [(max(0, start - amount), min(text_length, end + amount)) for start, end in spans]


def classify(
    actual: float,
    deeper: float,
    wider: float,
    oracle: float,
    *,
    threshold: float,
) -> str:
    if actual >= threshold:
        return "evidence_surfaced__interpretation_or_completeness"
    if deeper >= threshold:
        return "harness_ranking_depth"
    if wider >= threshold:
        return "harness_context_width"
    if oracle >= threshold:
        return "query_planning_or_early_stopping"
    return "retrieval_backend_or_evidence_alignment"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--blind-key", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--system", default="sft_v5_checkpoint50")
    parser.add_argument("--coverage-threshold", type=float, default=0.20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    benchmark = {q["id"]: q for q in json.loads(args.benchmark.read_text())["questions"]}
    runs = {row["question_id"]: row for row in json.loads(args.run.read_text())}
    reviews = json.loads(args.review.read_text())["rows"]
    identities = {
        row["blind_id"]: row["system"]
        for row in json.loads(args.blind_key.read_text())["assignments"]
    }

    rows = []
    for review in reviews:
        if identities.get(review["blind_id"]) != args.system:
            continue
        if review["expected_answerability"] != "sufficient" or review["verdict"] == "pass":
            continue

        qid = review["question_id"]
        question = benchmark[qid]
        run = runs[qid]
        doc_id = question["required_doc_ids"][0]
        document = json.loads((args.corpus / f"{doc_id}.json").read_text())
        text = document["text"]
        gold_spans = find_evidence_spans(text, question.get("grader_notes", []))
        actual_spans = observation_spans(run["trajectory"], doc_id)
        queries = search_queries(run["trajectory"], doc_id)

        top3 = [span for query in queries for span in lexical_search(text, query, top_k=3)]
        top8 = [span for query in queries for span in lexical_search(text, query, top_k=8)]
        oracle = oracle_queries(question, text)
        oracle_spans = [
            span for query in oracle for span in lexical_search(text, query, top_k=3)
        ]

        coverages = {
            "actual_observations": span_coverage(gold_spans, actual_spans),
            "same_queries_top8": span_coverage(gold_spans, top8),
            "same_top3_expanded_context": span_coverage(
                gold_spans, expanded(top3, len(text))
            ),
            "oracle_query_top3": span_coverage(gold_spans, oracle_spans),
        }
        label = classify(
            coverages["actual_observations"],
            coverages["same_queries_top8"],
            coverages["same_top3_expanded_context"],
            coverages["oracle_query_top3"],
            threshold=args.coverage_threshold,
        )
        rows.append({
            "question_id": qid,
            "blind_id": review["blind_id"],
            "verdict": review["verdict"],
            "question": question["question"],
            "reference_answer": question["answer"],
            "predicted_answer": run["predicted_answer"],
            "queries": queries,
            "oracle_queries": oracle,
            "gold_span_count": len(gold_spans),
            "coverage": {key: round(value, 4) for key, value in coverages.items()},
            "attribution": label,
        })

    counts = Counter(row["attribution"] for row in rows)
    payload = {
        "schema_version": "qasper-failure-attribution-v1",
        "warning": "Uses gold answers/evidence for diagnosis; do not expose to evaluated policies.",
        "coverage_threshold": args.coverage_threshold,
        "system": args.system,
        "questions_audited": len(rows),
        "attribution_counts": dict(sorted(counts.items())),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"Audited {len(rows)} non-passing answerable responses")
    for label, count in sorted(counts.items()):
        print(f"  {count:2d}  {label}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
