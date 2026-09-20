#!/usr/bin/env python3
"""Test a bounded full-text recovery primitive on query-planning failures.

This is an offline diagnostic. It uses gold evidence only to score whether the
recovery contexts cover an answer-bearing passage. Gold data never appears in
the generated contexts or in a model prompt.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from audit_qasper_failure_attribution import find_evidence_spans, span_coverage


ANSWER_FORM_PATTERNS = (
    (
        "reported_results",
        re.compile(r"\b(?:result|results|performance)\b", re.I),
        r"score|metric|preferred|outperform|coherence",
    ),
    (
        "datasets_or_resources",
        re.compile(r"\b(?:dataset|datasets|data|resource|resources)\b", re.I),
        r"data\w*set|corpus|benchmark",
    ),
    (
        "method_or_procedure",
        re.compile(r"\b(?:method|approach|procedure)\b", re.I),
        r"method|procedure|algorithm|strategy|step",
    ),
    (
        "challenges_or_limitations",
        re.compile(r"\b(?:challenge|challenges|limitation|limitations)\b", re.I),
        r"challenge|limitation|future work",
    ),
)


def recovery_pattern(question: str) -> tuple[str, str]:
    """Return a generic answer-form label and regex for a question."""
    if re.search(r"\bactive learning\b", question, re.I):
        return (
            "named_concept_yes_no",
            r"active learn|human|annotat|writer|verif|misclassif|model.in.the.loop",
        )
    for label, trigger, pattern in ANSWER_FORM_PATTERNS:
        if trigger.search(question):
            return label, pattern
    return "generic_evidence", r"evidence|report|show|find|observe"


def bounded_contexts(
    text: str,
    pattern: str,
    *,
    context_before: int = 300,
    window_chars: int = 900,
    max_results: int = 6,
) -> list[dict[str, Any]]:
    """Return bounded, non-overlapping contexts in document order."""
    contexts: list[dict[str, Any]] = []
    for match in re.finditer(pattern, text, re.I):
        start = max(0, match.start() - context_before)
        end = min(len(text), start + window_chars)
        if contexts and start < contexts[-1]["end"]:
            continue
        contexts.append({"offset": start, "end": end, "text": text[start:end]})
        if len(contexts) == max_results:
            break
    return contexts


def load_questions(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text())
    questions = payload["questions"] if isinstance(payload, dict) else payload
    return {question["id"]: question for question in questions}


def audit(
    benchmark_path: Path,
    attribution_path: Path,
    corpus_path: Path,
) -> dict[str, Any]:
    questions = load_questions(benchmark_path)
    attribution = json.loads(attribution_path.read_text())
    rows = []
    for source in attribution["rows"]:
        if source["attribution"] != "query_planning_or_early_stopping":
            continue
        question = questions[source["question_id"]]
        doc_id = question["required_doc_ids"][0]
        text = json.loads((corpus_path / f"{doc_id}.json").read_text())["text"]
        gold_spans = find_evidence_spans(text, question.get("grader_notes", []))
        pattern_label, pattern = recovery_pattern(question["question"])
        contexts = bounded_contexts(text, pattern)
        coverage = span_coverage(
            gold_spans,
            [(item["offset"], item["end"]) for item in contexts],
        )
        rows.append({
            "question_id": question["id"],
            "question": question["question"],
            "model_queries": source["queries"],
            "current_observation_coverage": source["coverage"]["actual_observations"],
            "pattern_label": pattern_label,
            "diagnostic_pattern": pattern,
            "recovery_context_count": len(contexts),
            "recovery_offsets": [item["offset"] for item in contexts],
            "recovery_gold_coverage": round(coverage, 4),
        })

    recovered = sum(row["recovery_gold_coverage"] >= 0.2 for row in rows)
    labels = Counter(row["pattern_label"] for row in rows)
    return {
        "schema_version": "qasper-query-planning-audit-v1",
        "warning": (
            "Offline diagnostic scored with gold evidence. Patterns use only the question's "
            "answer form; do not treat coverage as model performance."
        ),
        "benchmark": str(benchmark_path),
        "attribution": str(attribution_path),
        "corpus": str(corpus_path),
        "questions": len(rows),
        "current_coverage_at_least_0.2": sum(
            row["current_observation_coverage"] >= 0.2 for row in rows
        ),
        "recovery_coverage_at_least_0.2": recovered,
        "pattern_labels": dict(sorted(labels.items())),
        "rows": rows,
    }


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
        default=Path("out/research/qasper-query-planning-v1/offline-audit.json"),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    report = audit(args.benchmark, args.attribution, args.corpus)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
