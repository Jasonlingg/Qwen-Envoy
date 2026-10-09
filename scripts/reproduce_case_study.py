#!/usr/bin/env python3
"""Verify packaged files and recount the September 30 case study, entirely offline.

Run with Python 3.10+: ``python scripts/reproduce_case_study.py [--json]``.
Only the standard library is used. No installation, corpus download, model,
endpoint, or paid API is needed. This reaggregates existing provisional review
labels; it does not independently validate answers, references, or source spans.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path

DEFAULT_RELEASE = Path(__file__).resolve().parents[1] / "release/envoybench-v0.1"
# These seven original files are immutable inputs, even when release prose or
# SHA256SUMS is updated. Trust is rooted in this repository version, not a signature.
FROZEN_SHA256 = {
    "run/manifest.json": "d57e8121e234dae6fefd7dae50f40bb38e320da63e1036c3d50fcef1eea8c2d9",
    "run/results.json": "8f94c52fd7ea9e56540400753234aee10967f97acbcf199e97b30c30ab77ef39",
    "review-prepared/review.json": (
        "6b36b44e3513697d5526734aeb0c1940c24a2f4a3e355e549c9b60ff17f9c0d2"
    ),
    "review-prepared/blind-key.json": (
        "ddc550e7458ec5b6affb9940a303f5a0e3b3578414975c810e7d1984fb5e6190"
    ),
    "review-model-assisted/review.json": (
        "d01b9570e586278b7e841708025f32fc1ec0d4a8e28537f81fae6e8ccec10083"
    ),
    "review-model-assisted/judge-report.json": (
        "1788dffb4888633aa981ee343ca05ff639bba0acf4bd629cba4bba5d38ce2dbf"
    ),
    "review-model-assisted/score.json": (
        "031eaeb7fdf77c37e6102a7061b4fa0f67f636174badaa1a0ab78f68b1459c06"
    ),
}
REVIEW_CONTENT_FIELDS = (
    "blind_id",
    "question_id",
    "question",
    "expected_answerability",
    "reference_answer",
    "grader_notes",
    "status",
    "answer",
    "citations",
    "evidence",
)
# Frozen equivalent of score.py's automatic-v2 error union. Importing score.py
# also imports the live runner and optional dependencies; this audit needs neither.
RUNTIME_ERROR = re.compile(
    r"^(?:Traceback \(most recent call last\):|[\w.]+(?:Error|Exception):)", re.M
)
HARNESS_ERROR = re.compile(
    r"^(?:ERROR: (?:Execution timed out after \d+s|Failed to write step script:)"
    r"|Action rejected:|Tool error:)",
    re.M,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _configuration_hash(value: object) -> str:
    """Use the existing src.eval.artifacts canonical JSON binding format."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksums(release_dir: Path) -> dict[str, str]:
    """Check every manifest entry and reject changed or omitted frozen inputs."""
    root = release_dir.resolve(strict=True)
    entries = {}
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        _require(match is not None, "invalid SHA256SUMS entry")
        digest, name = match.groups()
        path = root / name
        _require(
            not Path(name).is_absolute() and ".." not in Path(name).parts,
            f"unsafe checksum path: {name}",
        )
        _require(path.resolve().is_relative_to(root), f"checksum path escapes release: {name}")
        _require(name not in entries, f"duplicate checksum entry: {name}")
        _require(_file_hash(path) == digest, f"SHA256 mismatch: {name}")
        entries[name] = digest
    for name, digest in FROZEN_SHA256.items():
        _require(entries.get(name) == digest, f"frozen case-study hash missing or changed: {name}")
    return entries


def _structured_tool_error(observation: str) -> bool:
    stdout = observation.split("\nSTDERR:\n", 1)[0]
    stdout = re.split(r"\n\n(?:\n)?(?:\[Step |\*\*\* FINAL STEP|HINT:)", stdout)[0]

    def parse(value: str):
        if value.strip().startswith(("{", "[")):
            for parser in (json.loads, ast.literal_eval):
                try:
                    return parser(value.strip())
                except (ValueError, SyntaxError, RecursionError):
                    pass
        return None

    def has_error(value: object) -> bool:
        records = value if isinstance(value, list) else [value]
        return any(isinstance(record, dict) and bool(record.get("error")) for record in records)

    parsed = parse(stdout)
    return (
        has_error(parsed)
        if parsed is not None
        else any(has_error(parse(line)) for line in stdout.splitlines())
    )


def recorded_error_episode(row: dict) -> bool:
    """Count tool-return, runtime, or failed-harness/endpoint episodes once."""
    if row["status"] == "error":
        return True
    for step in row["trajectory"]:
        if not isinstance(step, dict):
            continue
        observation = str(step.get("observation", ""))
        stderr = observation.split("\nSTDERR:\n", 1)[1] if "\nSTDERR:\n" in observation else ""
        if (
            _structured_tool_error(observation)
            or RUNTIME_ERROR.search(stderr)
            or HARNESS_ERROR.search(observation)
        ):
            return True
    return False


def reproduce(release_dir: Path = DEFAULT_RELEASE) -> dict:
    hashes = verify_checksums(release_dir)
    artifacts = {
        name: json.loads((release_dir / name).read_text(encoding="utf-8")) for name in FROZEN_SHA256
    }
    manifest = artifacts["run/manifest.json"]
    results = artifacts["run/results.json"]
    prepared = artifacts["review-prepared/review.json"]
    key = artifacts["review-prepared/blind-key.json"]
    review = artifacts["review-model-assisted/review.json"]
    judge = artifacts["review-model-assisted/judge-report.json"]
    saved_score = artifacts["review-model-assisted/score.json"]
    results_hash = _configuration_hash({"results": results})
    question_ids = manifest["question_ids"]
    model_keys = [model["key"] for model in manifest["models"]]
    expected_pairs = {(model, question) for model in model_keys for question in question_ids}
    by_pair = {(row["model_key"], row["question_id"]): row for row in results}
    _require(len(question_ids) == len(set(question_ids)) == 40, "expected 40 unique questions")
    _require(set(model_keys) == {"qwen_base", "qwen_v5"}, "unexpected case-study models")
    _require(
        len(results) == len(by_pair) and set(by_pair) == expected_pairs,
        "missing or duplicate model/question result pairs",
    )
    for row in results:
        for field in (
            "run_id",
            "comparison_id",
            "benchmark_id",
            "benchmark_hash",
            "corpus_hash",
            "split",
            "split_status",
        ):
            _require(row[field] == manifest[field], f"result/manifest {field} mismatch")
    for artifact in (prepared, key, review, judge, saved_score):
        _require(artifact["results_hash"] == results_hash, "results/review binding mismatch")
        for field in ("benchmark_id", "benchmark_hash", "corpus_hash"):
            _require(artifact[field] == manifest[field], f"review/manifest {field} mismatch")
    _require(
        review["status"] == "complete"
        and review["reviewer_kind"] == "model_assisted"
        and review["source_reference_status"] == "unreviewed_qasper",
        "expected complete provisional model-assisted review with unreviewed references",
    )
    _require(
        judge["input_review_hash"] == _configuration_hash(prepared),
        "judge/prepared review binding mismatch",
    )
    _require(
        judge["judged_review_hash"] == _configuration_hash(review),
        "judge/completed review binding mismatch",
    )
    assignments = {item["blind_id"]: item for item in key["assignments"]}
    review_rows = {row["blind_id"]: row for row in review["rows"]}
    _require(
        len(assignments) == len(key["assignments"]) == len(results)
        and len(review_rows) == len(review["rows"]) == len(results)
        and set(assignments) == set(review_rows),
        "missing or duplicate blind assignments",
    )
    assigned_pairs = set()
    labels = {}
    models = {
        model: {
            "questions": 0,
            "provisional_passes": 0,
            "answerable_questions": 0,
            "answerable_passes": 0,
            "unanswerable_questions": 0,
            "correct_abstentions": 0,
            "recorded_error_episodes": 0,
            "submissions": 0,
        }
        for model in model_keys
    }
    for blind_id, assignment in assignments.items():
        pair = (assignment["system"], assignment["question_id"])
        _require(
            pair in by_pair and pair not in assigned_pairs, "invalid blind model/question pair"
        )
        assigned_pairs.add(pair)
        result, row = by_pair[pair], review_rows[blind_id]
        _require(
            assignment["result_hash"] == _configuration_hash(result),
            f"{blind_id}: saved result binding mismatch",
        )
        content = {field: row.get(field) for field in REVIEW_CONTENT_FIELDS}
        _require(
            assignment["review_content_hash"] == _configuration_hash(content)
            and row["question_id"] == pair[1],
            f"{blind_id}: review content binding mismatch",
        )
        label, verdict = row["expected_answerability"], row["verdict"]
        _require(
            label in {"sufficient", "insufficient"} and verdict in {"pass", "partial", "fail"},
            f"{blind_id}: invalid answerability/verdict",
        )
        _require(labels.setdefault(pair[1], label) == label, "paired answerability labels differ")
        metrics = models[pair[0]]
        metrics["questions"] += 1
        metrics["provisional_passes"] += verdict == "pass"
        category = "answerable" if label == "sufficient" else "unanswerable"
        metrics[f"{category}_questions"] += 1
        if verdict == "pass":
            metrics["answerable_passes" if label == "sufficient" else "correct_abstentions"] += 1
        metrics["recorded_error_episodes"] += recorded_error_episode(result)
        metrics["submissions"] += result["status"] == "submitted"
    for model, metrics in models.items():
        _require(
            metrics["provisional_passes"]
            == saved_score["provisional_model_assisted"]["systems"][model]["pass"],
            f"{model}: recomputed/saved pass count mismatch",
        )

    return {
        "schema_version": "envoy-case-study-reproduction-v1",
        "status": "verified",
        "scope": "offline artifact integrity and recount of existing provisional judgments",
        "models": models,
        "checks": {
            "release_files_sha256_verified": len(hashes),
            "frozen_inputs_pinned": len(FROZEN_SHA256),
            "complete_unique_model_question_pairs": len(by_pair),
            "result_review_and_judge_bindings": "passed",
            "source_corpus_and_spans_revalidated": False,
        },
        "provenance": {
            **{
                field: manifest[field]
                for field in (
                    "run_id",
                    "comparison_id",
                    "created_at_utc",
                    "git_commit",
                    "benchmark_id",
                    "benchmark_hash",
                    "corpus_hash",
                    "split",
                    "split_status",
                    "question_ids",
                    "seed",
                    "seed_note",
                    "max_steps",
                    "system_prompt_sha256",
                    "tool_preamble_sha256",
                    "tool_search_version",
                    "legacy_reward_version",
                    "legacy_reward_is_benchmark_score",
                    "runner_hardware",
                    "models",
                )
            },
            "results_hash": results_hash,
            "review_kind": review["reviewer_kind"],
            "reviewer_id": review["reviewer_id"],
            "rubric_version": review["rubric_version"],
            "source_reference_status": review["source_reference_status"],
            "judge_method_deviation": judge["method_deviation"],
            "judge_limitations": judge["limitations"],
            "sha256sums_sha256": _file_hash(release_dir / "SHA256SUMS"),
            "file_sha256": hashes,
        },
        "limitations": [
            "Semantic verdicts and correct-abstention labels remain provisional "
            "model-assisted judgments; "
            "this is no new independent validation and no human-reviewed model-improvement result.",
            "The corpus and source spans are not revalidated here; "
            "exact spans alone do not prove support.",
            "Hashes establish consistency with the trusted repository version, "
            "not endpoint weight attestation.",
            "Recorded-error counts use the automatic-v2 union of parsed tool errors, "
            "runtime failures, and error-status episodes; unprinted or truncated tool returns "
            "cannot be classified reliably.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument(
        "--json", action="store_true", help="emit machine-readable counts and provenance"
    )
    args = parser.parse_args(argv)
    try:
        report = reproduce(args.release_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}), file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(
            f"Verified {report['checks']['release_files_sha256_verified']} packaged file hashes; "
            "recomputed 40 paired questions (September 30, 2026)."
        )
        print("Metric                                  Base Qwen3-8B   v5 SFT")
        for label, numerator, denominator in (
            ("Provisional passes", "provisional_passes", "questions"),
            ("Provisional answerable passes", "answerable_passes", "answerable_questions"),
            ("Provisional correct abstentions", "correct_abstentions", "unanswerable_questions"),
            ("Episodes with recorded errors", "recorded_error_episodes", "questions"),
            ("Submissions", "submissions", "questions"),
        ):
            counts = [
                f"{report['models'][model][numerator]}/{report['models'][model][denominator]}"
                for model in ("qwen_base", "qwen_v5")
            ]
            print(f"{label:40}{counts[0]:16}{counts[1]}")
        print(f"Run: {report['provenance']['run_id']}")
        print(f"Results SHA256: {report['provenance']['file_sha256']['run/results.json']}")
        print(
            "Existing model-assisted judgments only; no new independent validation. "
            "Corpus and source spans were not revalidated. "
            "Use --json for full provenance and limits."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
