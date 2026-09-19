"""Compare within-paper retrieval modes on saved agent queries and gold evidence."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

from audit_qasper_failure_attribution import (
    find_evidence_spans,
    merge_intervals,
    span_coverage,
)
from src.env.repl import LocalREPL


CONDITIONS = (
    ("raw_top3", "raw", 3),
    ("raw_top8", "raw", 8),
    ("dedupe_merge_top3", "dedupe_merge", 3),
    ("ranked_diverse_top6", "ranked_diverse", 6),
)


def load_json(path: Path):
    with path.open() as stream:
        return json.load(stream)


def evaluate(
    benchmark_path: Path,
    attribution_path: Path,
    corpus: Path,
) -> dict:
    benchmark = load_json(benchmark_path)
    questions = {question["id"]: question for question in benchmark["questions"]}
    attribution = load_json(attribution_path)
    report = {
        "schema_version": "search-within-offline-ablation-v1",
        "benchmark": str(benchmark_path),
        "attribution": str(attribution_path),
        "corpus": str(corpus),
        "conditions": {},
        "rows": [],
    }

    for label, mode, top_k in CONDITIONS:
        os.environ["ENVOY_SEARCH_WITHIN_MODE"] = mode
        os.environ["ENVOY_SEARCH_WITHIN_TOP_K"] = str(top_k)
        repl = LocalREPL(corpus_path=str(corpus))
        repl.start_session()
        coverages: list[float] = []
        returned_chars: list[int] = []
        unique_chars: list[int] = []
        try:
            for source in attribution["rows"]:
                question = questions[source["question_id"]]
                doc_id = question["required_doc_ids"][0]
                text = load_json(corpus / f"{doc_id}.json")["text"]
                gold = find_evidence_spans(text, question.get("grader_notes", []))
                spans: list[tuple[int, int]] = []
                chars = 0
                for query in source["queries"]:
                    action = (
                        "print(json.dumps(search_within("
                        f"{doc_id!r}, {query!r}, top_k={top_k})))"
                    )
                    hits = json.loads(repl.execute(action))
                    for hit in hits:
                        start = hit["offset"]
                        end = start + len(hit["text"])
                        spans.append((start, end))
                        chars += len(hit["text"])
                unique = sum(end - start for start, end in merge_intervals(spans))
                coverage = span_coverage(gold, spans)
                coverages.append(coverage)
                returned_chars.append(chars)
                unique_chars.append(unique)
                report["rows"].append(
                    {
                        "condition": label,
                        "question_id": question["id"],
                        "coverage": coverage,
                        "returned_text_chars": chars,
                        "unique_text_chars": unique,
                    }
                )
        finally:
            repl.kill_session()

        report["conditions"][label] = {
            "questions": len(coverages),
            "gold_coverage_at_least_0.2": sum(value >= 0.2 for value in coverages),
            "gold_coverage_at_least_0.8": sum(value >= 0.8 for value in coverages),
            "mean_gold_coverage": statistics.mean(coverages),
            "mean_returned_text_chars": statistics.mean(returned_chars),
            "mean_unique_text_chars": statistics.mean(unique_chars),
            "redundant_char_fraction": 1 - sum(unique_chars) / sum(returned_chars),
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("out/research/qasper-code-dev-v2/benchmark.json"),
    )
    parser.add_argument(
        "--attribution",
        type=Path,
        default=Path(
            "out/research/qwen3-qasper-v5-sft-20260919/eval40/"
            "failure-attribution.json"
        ),
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("out/research/qasper-code-dev-v2/corpus"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "out/research/qasper-search-within-ablation-v1/offline-coverage.json"
        ),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    report = evaluate(args.benchmark, args.attribution, args.corpus)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["conditions"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
