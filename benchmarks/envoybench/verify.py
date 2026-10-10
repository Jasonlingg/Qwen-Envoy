"""Verify one saved EnvoyBench comparison without running a model.

This checks frozen inputs, the complete paired run, exact submitted source
spans, and review/report bindings using the same loader as the Studio. It does
not decide whether a cited passage supports an answer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from benchmarks.envoybench.demo import build_demo_payload
from benchmarks.envoybench.frozen_dataset import DEFAULT_DATASET


def verify_run(
    *, run_dir: Path, dataset: Path = DEFAULT_DATASET,
    review_dir: Path | None = None,
    reference_review: Path | None = None,
    provisional: bool = False,
    judged_review: Path | None = None,
) -> dict:
    """Return a compact audit of checks already enforced by the Studio loader."""
    active = build_demo_payload(
        run_dir=run_dir, dataset=dataset, review_dir=review_dir,
        reference_review_path=reference_review, provisional=provisional,
        judged_review_path=judged_review,
    )["active"]
    if active["kind"] != "envoybench_run":
        raise ValueError("verification requires a saved EnvoyBench run")

    spans = [
        span
        for case in active["cases"]
        for system in case["systems"].values()
        for span in system["evidence"]
    ]
    provenance = active["provenance"]
    models = {}
    for item in active["models"]:
        metrics = item["metrics"]
        models[item["key"]] = {
            "answerable_pass": metrics["answerable_pass"],
            "answerable_reviewed": metrics["answerable_reviewed"],
            "unanswerable_pass": metrics["unanswerable_pass"],
            "unanswerable_reviewed": metrics["unanswerable_reviewed"],
            "classified_abstentions": metrics.get("abstention_count"),
            "execution_error_episode_rate": metrics["execution_error_episode_rate"],
        }

    human_reviewed = provenance["review_kind"] == "human"
    return {
        "schema_version": "envoybench-verification-v1",
        "status": "verified",
        "run_id": provenance["run_id"],
        "split_status": provenance["split_status"],
        "questions": active["question_denominator"],
        "models": models,
        "checks": {
            "frozen_split_benchmark_corpus_hashes": "passed",
            "complete_unique_model_question_pairs": "passed",
            "source_spans_checked": len(spans),
            "source_spans_exact": sum(span["valid"] for span in spans),
            "source_spans_invalid": sum(not span["valid"] for span in spans),
            "results_bound_to_review": review_dir is not None,
            "judge_report_bound_to_review": provenance["judge"] is not None,
        },
        "review": {
            "status": active["score_status"],
            "kind": provenance["review_kind"],
            "human_reviewed": human_reviewed,
        },
        "limits": [
            "Exact spans and document IDs do not prove that a passage supports an answer.",
            "Matching hashes bind selected files; they do not attest that an endpoint ran a model.",
            *(
                ["Saved answers have no external content binding without a review bundle."]
                if review_dir is None else []
            ),
            *(
                ["Model-assisted grades and source references are provisional, not human gold."]
                if not human_reviewed else []
            ),
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--review-dir", type=Path)
    parser.add_argument("--reference-review", type=Path)
    parser.add_argument("--judged-review", type=Path)
    parser.add_argument("--provisional", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = verify_run(
            run_dir=args.run_dir, dataset=args.dataset,
            review_dir=args.review_dir, reference_review=args.reference_review,
            provisional=args.provisional, judged_review=args.judged_review,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
