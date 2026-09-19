"""Build a six-question oracle-evidence diagnostic from the failure audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ATTRIBUTION = "evidence_surfaced__interpretation_or_completeness"


def load_json(path: Path) -> dict:
    with path.open() as stream:
        return json.load(stream)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def prepare(benchmark_path: Path, attribution_path: Path, output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is not empty: {output}")

    benchmark = load_json(benchmark_path)
    questions = {question["id"]: question for question in benchmark["questions"]}
    attribution = load_json(attribution_path)
    selected_ids = [
        row["question_id"]
        for row in attribution["rows"]
        if row["attribution"] == ATTRIBUTION
    ]
    if len(selected_ids) != 6:
        raise ValueError(f"Expected six interpretation failures, found {len(selected_ids)}")

    selected = []
    corpus = output / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    for question_id in selected_ids:
        question = dict(questions[question_id])
        if question.get("expected_answerability") != "sufficient":
            raise ValueError(f"Oracle question is not answerable: {question_id}")
        doc_ids = question.get("required_doc_ids", [])
        notes = list(dict.fromkeys(question.get("grader_notes", [])))
        if len(doc_ids) != 1 or not notes:
            raise ValueError(f"Oracle question needs one document and gold evidence: {question_id}")
        doc_id = doc_ids[0]
        text = "\n\n--- GOLD EVIDENCE EXCERPT ---\n\n".join(notes)
        write_json(
            corpus / f"{doc_id}.json",
            {
                "doc_id": doc_id,
                "title": f"Oracle evidence for {question_id}",
                "text": text,
                "metadata": {
                    "diagnostic_only": True,
                    "source_question_id": question_id,
                    "warning": "Gold evidence; never use for training or held-out scoring.",
                },
            },
        )
        selected.append(question)

    write_json(
        output / "benchmark.json",
        {
            "schema_version": "research-benchmark-v1",
            "benchmark_id": "qasper-oracle-evidence-v1",
            "status": "development_diagnostic_only",
            "training_exclusion": "Contains held-out gold evidence. Never use for training.",
            "source_benchmark": str(benchmark_path),
            "source_attribution": str(attribution_path),
            "forced_first_action": "read(required_doc_id)",
            "questions": selected,
        },
    )


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
            "out/research/qwen3-qasper-v5-sft-20260919/eval40/failure-attribution.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("out/research/qasper-oracle-evidence-v1"),
    )
    args = parser.parse_args()
    prepare(args.benchmark, args.attribution, args.output)
    print(f"Prepared oracle-evidence diagnostic: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
