"""Score a complete v6 holdout run with QASPER Answer F1 diagnostics.

This applies the pinned official QASPER token-F1 function to every original
annotation in the locked 24-question subset. It is not a support judgment or a
score on the full QASPER test split.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from benchmarks.envoybench.frozen_dataset import load_split
from benchmarks.envoybench.package_v6_holdout import DEFAULT_LOCK
from benchmarks.envoybench.qasper_official import (
    EVALUATOR_COMMIT,
    EVALUATOR_SHA256,
    official_evaluator,
)
from benchmarks.envoybench.score import _verify_artifacts
from src.env.reward import parse_submission_details
from src.eval.artifacts import configuration_hash
from src.research.qasper import QASPER_REVISION

REFUSAL_CUE = re.compile(
    r"\b(?:unanswerable|cannot|cant|unable|insufficient|unclear|unknown|"
    r"not enough|not clear|not stated|not specified|not reported|not mentioned|"
    r"not provided|no (?:answer|information|evidence)|"
    r"(?:does not|doesnt) (?:answer|say|name|state|specify|report|provide|"
    r"contain|mention))\b",
    re.IGNORECASE,
)


def score(dataset: Path, run_dir: Path, *, holdout_lock_path: Path = DEFAULT_LOCK) -> dict:
    benchmark, snapshot, benchmark_path, corpus_path, _ = load_split(dataset, "holdout")
    lock = json.loads(holdout_lock_path.read_text())
    lock_sha256 = _sha256(holdout_lock_path)
    ids = [question["id"] for question in benchmark["questions"]]
    answerability = Counter(
        question["expected_answerability"] for question in benchmark["questions"]
    )
    if (
        lock.get("schema_version") != "qasper-v6-holdout-lock-v1"
        or lock.get("source_split") != "test"
        or lock.get("source_revision") != QASPER_REVISION
        or len(ids) != 24
        or ids != lock.get("question_ids")
        or dict(answerability) != lock.get("answerability_counts")
        or benchmark.get("holdout_lock_sha256") != lock_sha256
        or benchmark.get("benchmark_id") != "envoybench-qasper-v6-holdout-v1"
        or benchmark.get("source_revision") != QASPER_REVISION
        or benchmark.get("source_split") != "test"
        or snapshot.get("holdout_lock_sha256") != lock_sha256
        or snapshot.get("source_revision") != QASPER_REVISION
        or benchmark.get("source_benchmark_sha256") != lock.get("benchmark_sha256")
        or benchmark.get("corpus_hash") != lock.get("corpus_hash")
        or benchmark.get("reserved_doc_ids") != lock.get("reserved_doc_ids")
    ):
        raise ValueError("Dataset differs from the pinned 24-question holdout lock")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    results = json.loads((run_dir / "results.json").read_text())
    if (
        manifest.get("status") != "complete"
        or manifest.get("full_split") is not True
        or manifest.get("split") != "holdout"
        or manifest.get("question_ids") != ids
        or manifest.get("benchmark_hash") != configuration_hash(benchmark)
        or manifest.get("benchmark_file_sha256") != _sha256(benchmark_path)
        or manifest.get("corpus_hash") != benchmark["corpus_hash"]
    ):
        raise ValueError("Run is incomplete or differs from the locked holdout")
    _, model_keys = _verify_artifacts(
        benchmark, manifest, results, corpus_path, require_paired=False,
    )
    keyed = {(row["model_key"], row["question_id"]): row for row in results}
    evaluator = official_evaluator()
    per_model: dict[str, list[dict]] = defaultdict(list)
    for question in benchmark["questions"]:
        qid = question["id"]
        references = [item["answer_text"] for item in question["reference_answers"]]
        if not references:
            raise ValueError(f"{qid} has no pinned QASPER answer annotations")
        for model in model_keys:
            result = keyed[(model, qid)]
            if result["status"] == "submitted":
                trajectory = result["trajectory"]
                terminal = trajectory[-1] if trajectory else None
                submission = (
                    parse_submission_details(terminal.get("action", ""))
                    if isinstance(terminal, dict) else None
                )
                if submission is None or submission != (
                    result["predicted_answer"], result["predicted_citations"],
                    result["predicted_evidence"],
                ):
                    raise ValueError(f"{model}/{qid}: submitted fields differ from final action")
            answer = result["predicted_answer"] if result["status"] == "submitted" else ""
            f1 = max(evaluator.token_f1_score(answer, reference) for reference in references)
            normalized = evaluator.normalize_answer(answer)
            literal_refusal = normalized == "unanswerable"
            possible_refusal = not literal_refusal and bool(REFUSAL_CUE.search(normalized))
            per_model[model].append({
                "question_id": qid,
                "expected_answerability": question["expected_answerability"],
                "status": result["status"],
                "answer_f1": f1,
                "refusal_diagnostic": (
                    "literal_unanswerable" if literal_refusal else
                    "possible_refusal_review_needed" if possible_refusal else
                    "not_identified_by_literal_check"
                ),
                "reference_count": len(references),
            })
    systems = {}
    for model, rows in per_model.items():
        strata = {}
        for answerability in ("sufficient", "insufficient"):
            subset = [row for row in rows if row["expected_answerability"] == answerability]
            strata[answerability] = {
                "questions": len(subset),
                "answer_f1_0_100": 100 * sum(row["answer_f1"] for row in subset) / len(subset),
                "literal_unanswerable": sum(
                    row["refusal_diagnostic"] == "literal_unanswerable" for row in subset
                ),
                "possible_refusals_review_needed": sum(
                    row["refusal_diagnostic"] == "possible_refusal_review_needed"
                    for row in subset
                ),
            }
        systems[model] = {
            "answer_f1_0_100": 100 * sum(row["answer_f1"] for row in rows) / len(rows),
            "literal_unanswerable_on_sufficient": strata["sufficient"]["literal_unanswerable"],
            "literal_unanswerable_on_insufficient": strata["insufficient"]["literal_unanswerable"],
            "possible_refusals_review_needed": sum(
                row["refusal_diagnostic"] == "possible_refusal_review_needed" for row in rows
            ),
            "refusal_behavior_review_required": True,
            "strata": strata,
            "question_rows": rows,
        }
    return {
        "schema_version": "qasper-v6-holdout-automatic-v1",
        "run_manifest_sha256": _sha256(run_dir / "manifest.json"),
        "run_results_sha256": _sha256(run_dir / "results.json"),
        "benchmark_sha256": _sha256(benchmark_path),
        "holdout_lock_sha256": lock_sha256,
        "official_evaluator_commit": EVALUATOR_COMMIT,
        "official_evaluator_sha256": EVALUATOR_SHA256,
        "question_count": len(ids),
        "systems": systems,
        "refusal_diagnostic_note": (
            "Literal Unanswerable matches are exact-format diagnostics. Other answers with "
            "refusal wording are flagged for review; unflagged answers may also refuse. "
            "No semantic false-refusal count is inferred."
        ),
        "claim_boundary": (
            "Official QASPER token-F1 function on 24 selected test questions and their "
            "original annotations. This does not measure whether answer claims are "
            "supported by cited passages and is not the full QASPER benchmark. "
            "Literal refusal screening cannot establish false-refusal counts; "
            "promotion requires blind answer-behavior review."
        ),
    }


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    report = score(args.dataset, args.run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: round(value["answer_f1_0_100"], 2)
                      for key, value in report["systems"].items()}))


if __name__ == "__main__":
    main()
