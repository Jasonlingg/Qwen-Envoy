"""Offline, paired EnvoyBench checks and blinded source-supported-answer review.

Automatic checks cover only execution and provenance. Exact document IDs and
character offsets do not establish that a passage supports an answer. The
supported-answer score is produced only from a completed, provenance-labeled
review; model-assisted judgments remain explicitly provisional.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import re
import statistics
from collections import defaultdict
from datetime import date
from itertools import combinations
from math import comb, isfinite
from pathlib import Path

from benchmarks.envoybench.run import load_split
from src.eval.artifacts import configuration_hash, content_hash
from src.eval.research_review import (
    load_corpus,
    materialize_evidence,
    review_markdown,
    score_review,
)
from src.research.benchmark import validate_benchmark

REVIEW_VERSION = "envoybench-blind-review-v1"
KEY_VERSION = "envoybench-blind-key-v1"
AUTOMATIC_VERSION = "envoybench-automatic-v2"
REVIEWED_VERSION = "envoybench-reviewed-score-v1"
PROVISIONAL_SCORE_VERSION = "envoybench-provisional-model-assisted-score-v1"
REFERENCE_VERSION = "envoybench-reference-review-v1"
RUBRIC_VERSION = "envoybench-review-rubric-v1"
RUNTIME_ERROR_MARKER = re.compile(
    r"^(?:Traceback \(most recent call last\):|[\w.]+(?:Error|Exception):)", re.M
)
HARNESS_ERROR_MARKER = re.compile(
    r"^(?:ERROR: (?:Execution timed out after \d+s|Failed to write step script:)"
    r"|Action rejected:|Tool error:)", re.M
)
ENDPOINT_ERROR_MARKER = re.compile(
    r"model endpoint returned HTTP \d+|could not reach model endpoint:|"
    r"invalid chat-completions response shape|"
    r"chat-completions response did not contain text content|"
    r"^(?:APIConnectionError|APITimeoutError|APIStatusError|BadRequestError|"
    r"RateLimitError|AuthenticationError|InternalServerError):", re.I
)
CONTEXT_ERROR_MARKER = re.compile(
    r"maximum context length|context[_ ]length[_ ]exceeded|context window exceeded", re.I
)


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 4) if values else None


def _structured_tool_error_count(observation: str) -> int:
    """Count returned error records, never words inside a retrieved passage.

    The saved protocol has stdout strings rather than typed tool events. Parse
    complete JSON/Python literal return values, including lists and one return
    per line. Do not recurse into source text or arbitrary nested metadata.
    Unparseable/truncated output cannot reliably establish a returned error.
    """
    stdout = observation.split("\nSTDERR:\n", 1)[0]
    stdout = re.split(r"\n\n(?:\n)?(?:\[Step |\*\*\* FINAL STEP|HINT:)", stdout)[0]

    def parse(value: str):
        if not value.strip().startswith(("{", "[")):
            return None
        for parser in (json.loads, ast.literal_eval):
            try:
                return parser(value.strip())
            except (ValueError, SyntaxError, RecursionError):
                pass
        return None

    def count(value: object) -> int:
        records = value if isinstance(value, list) else [value]
        return sum(isinstance(record, dict) and bool(record.get("error"))
                   for record in records)

    parsed = parse(stdout)
    if parsed is not None:
        return count(parsed)
    return sum(count(parse(line)) for line in stdout.splitlines())


def _error_diagnostics(row: dict) -> dict:
    tool_returns = tool_steps = runtime_steps = any_steps = 0
    for step in row["trajectory"]:
        if not isinstance(step, dict):
            continue
        observation = str(step.get("observation", ""))
        returned = _structured_tool_error_count(observation)
        # Python exceptions are emitted on stderr; treating arbitrary source
        # prose containing "SyntaxError" or "timed out" as failures is wrong.
        stderr = observation.split("\nSTDERR:\n", 1)[-1] if (
            "\nSTDERR:\n" in observation
        ) else ""
        runtime = bool(RUNTIME_ERROR_MARKER.search(stderr)
                       or HARNESS_ERROR_MARKER.search(observation))
        tool_returns += returned
        tool_steps += bool(returned)
        runtime_steps += runtime
        any_steps += bool(returned) or runtime
    failed = row["status"] == "error"
    endpoint = failed and bool(ENDPOINT_ERROR_MARKER.search(str(row.get("error", ""))))
    context = endpoint and bool(CONTEXT_ERROR_MARKER.search(str(row.get("error", ""))))
    other = failed and not endpoint
    categories = [name for name, present in (
        ("tool_return_error", tool_steps), ("runtime_error", runtime_steps),
        ("endpoint_error", endpoint), ("context_limit_error", context),
        ("other_harness_error", other),
    ) if present]
    return {
        "tool_error_returns": tool_returns,
        "tool_error_steps": tool_steps,
        "tool_error_episode": bool(tool_steps),
        "runtime_error_steps": runtime_steps,
        "runtime_error_episode": bool(runtime_steps),
        "endpoint_error_episode": endpoint,
        "context_limit_error_episode": context,
        "other_harness_error_episode": other,
        "error_categories": categories,
        # Compatibility fields represent the union, not only Python exceptions.
        "execution_error_steps": any_steps,
        "execution_error_episode": bool(any_steps) or failed,
    }


def _strict_evidence(items: object, documents: dict[str, dict]) -> list[dict]:
    """Reuse span materialization and also detect altered submitted quote text."""
    resolved = materialize_evidence(items, documents)
    if not isinstance(items, list):
        return resolved
    for original, passage in zip(items, resolved, strict=True):
        if (passage.get("valid") and isinstance(original, dict)
                and "quote" in original and original["quote"] != passage["quote"]):
            passage["snapshot_quote"] = passage.pop("quote")
            passage["valid"] = False
            passage["error"] = "submitted quote does not match frozen source text"
    return resolved


def _review_content(row: dict) -> dict:
    """Fields that must stay fixed after the blind bundle is generated."""
    return {name: row.get(name) for name in (
        "blind_id", "question_id", "question", "expected_answerability",
        "reference_answer", "grader_notes", "status", "answer", "citations", "evidence",
    )}


def _verify_artifacts(benchmark: dict, run_manifest: dict, results: list[dict],
                      corpus_dir: Path, *,
                      require_paired: bool = True) -> tuple[dict[str, dict], list[str]]:
    validate_benchmark(benchmark)
    if not isinstance(run_manifest, dict) or not isinstance(results, list):
        raise ValueError("run manifest and results must be JSON objects/list")
    if not corpus_dir.is_dir():
        raise ValueError("frozen corpus directory does not exist")
    corpus_hash = content_hash(corpus_dir)
    benchmark_hash = configuration_hash(benchmark)
    if benchmark["corpus_hash"] != corpus_hash:
        raise ValueError("benchmark corpus hash differs from frozen corpus bytes")
    for field, expected in (
        ("benchmark_id", benchmark["benchmark_id"]),
        ("benchmark_hash", benchmark_hash),
        ("corpus_hash", corpus_hash),
    ):
        if run_manifest.get(field) != expected:
            raise ValueError(f"run manifest {field} differs from frozen benchmark")
    if run_manifest.get("schema_version") != "envoybench-run-v1":
        raise ValueError("unsupported EnvoyBench run manifest")
    if run_manifest.get("status") != "complete":
        raise ValueError("run manifest is incomplete")
    if (run_manifest.get("full_split") is not True
            or run_manifest.get("subset_smoke") is not False):
        raise ValueError("subset smoke is diagnostic, not a full benchmark comparison")
    documents = load_corpus(corpus_dir)
    if len(documents) != len(list(corpus_dir.glob("*.json"))):
        raise ValueError("frozen corpus has duplicate document IDs")
    question_ids = [question["id"] for question in benchmark["questions"]]
    question_set = set(question_ids)
    question_text = {question["id"]: question["question"] for question in benchmark["questions"]}
    if run_manifest.get("split_question_count") != len(question_ids):
        raise ValueError("run manifest split count differs from benchmark")
    result_keys = []
    for row in results:
        if not isinstance(row, dict):
            raise ValueError("each result must be an object")
        model_key, question_id = row.get("model_key"), row.get("question_id")
        if not isinstance(model_key, str) or not model_key:
            raise ValueError("each result needs a non-empty model_key")
        if question_id not in question_set:
            raise ValueError("result has an unknown question_id")
        if row.get("question") != question_text[question_id]:
            raise ValueError(f"{question_id}/{model_key} question text mismatch")
        if row.get("schema_version") != "envoybench-run-v1":
            raise ValueError(f"{question_id}/{model_key} run schema mismatch")
        for field, expected in (
            ("benchmark_id", benchmark["benchmark_id"]),
            ("benchmark_hash", benchmark_hash),
            ("corpus_hash", corpus_hash),
            ("run_id", run_manifest.get("run_id")),
            ("comparison_id", run_manifest.get("comparison_id")),
            ("split", run_manifest.get("split")),
            ("split_status", run_manifest.get("split_status")),
        ):
            if row.get(field) != expected:
                raise ValueError(f"{question_id}/{model_key} {field} mismatch")
        if not isinstance(row.get("predicted_answer"), str):
            raise ValueError(f"{question_id}/{model_key} has no answer text field")
        if not isinstance(row.get("predicted_citations"), list):
            raise ValueError(f"{question_id}/{model_key} has invalid citations")
        if not isinstance(row.get("predicted_evidence"), list):
            raise ValueError(f"{question_id}/{model_key} has invalid evidence")
        if not isinstance(row.get("trajectory"), list):
            raise ValueError(f"{question_id}/{model_key} has invalid trajectory")
        if row.get("status") not in {"submitted", "no_submission", "error", "escalated"}:
            raise ValueError(f"{question_id}/{model_key} has an unknown status")
        duration = row.get("duration_seconds")
        if type(duration) not in {int, float} or not isfinite(duration) or duration < 0:
            raise ValueError(f"{question_id}/{model_key} has invalid duration")
        result_keys.append((model_key, question_id))
    if len(result_keys) != len(set(result_keys)):
        raise ValueError("results contain duplicate model/question pairs")
    models = sorted({model for model, _ in result_keys})
    if require_paired and len(models) < 2:
        raise ValueError("paired review needs at least two models")
    if not models:
        raise ValueError("saved run needs at least one model")
    if set(result_keys) != {(model, question) for model in models for question in question_set}:
        raise ValueError("results are not a complete paired model/question matrix")
    if set(run_manifest.get("question_ids", [])) != question_set:
        raise ValueError("run manifest question IDs do not match the benchmark")
    if len(run_manifest["question_ids"]) != len(question_ids):
        raise ValueError("run manifest repeats a question ID")
    declared = run_manifest.get("models")
    if not isinstance(declared, list):
        raise ValueError("run manifest has no model identities")
    declared_keys = {item.get("key") if isinstance(item, dict) else item for item in declared}
    if declared_keys != set(models) or len(declared) != len(models):
        raise ValueError("manifest model keys do not match result model keys")
    return documents, models


def _mechanical_row(row: dict, question: dict, documents: dict[str, dict]) -> dict:
    evidence = _strict_evidence(row["predicted_evidence"], documents)
    citations = row["predicted_citations"]
    valid_citations = [item for item in citations
                       if isinstance(item, str) and item in documents]
    valid_spans = [item for item in evidence if item["valid"]]
    required = set(question.get("required_doc_ids", []))
    steps = row["trajectory"]
    actions = [str(step.get("action", "")).strip() for step in steps
               if isinstance(step, dict)]
    explored = [action for action in actions if action and not action.upper().startswith(
        "SUBMIT:"
    )]
    seen_actions = set()
    repeated = 0
    for action in explored:
        repeated += action in seen_actions
        seen_actions.add(action)
    return {
        "question_id": row["question_id"],
        "status": row["status"],
        "submitted": row["status"] == "submitted" and bool(row["predicted_answer"].strip()),
        "citation_count": len(citations),
        "valid_citation_id_count": len(valid_citations),
        "evidence_span_count": len(evidence),
        "valid_evidence_span_count": len(valid_spans),
        "question_has_valid_evidence": bool(valid_spans),
        "required_document_hits": len(required & set(valid_citations)),
        "required_document_count": len(required),
        "trajectory_steps": len(steps),
        "exploration_steps": len(explored),
        "repeated_action_steps": repeated,
        **_error_diagnostics(row),
        "verifier_rejections": len(row.get("verifier_events") or []),
        "duration_seconds": float(row["duration_seconds"]),
        "duration_observed": row["duration_seconds"] > 0,
    }


def _automatic_score(benchmark: dict, results: list[dict],
                     documents: dict[str, dict], models: list[str]) -> dict:
    questions = {question["id"]: question for question in benchmark["questions"]}
    by_model: dict[str, list[dict]] = defaultdict(list)
    for result in results:
        by_model[result["model_key"]].append(
            _mechanical_row(result, questions[result["question_id"]], documents)
        )
    systems = {}
    for model in models:
        rows = sorted(by_model[model], key=lambda row: row["question_id"])
        total_spans = sum(row["evidence_span_count"] for row in rows)
        total_citations = sum(row["citation_count"] for row in rows)
        total_steps = sum(row["trajectory_steps"] for row in rows)
        total_exploration = sum(row["exploration_steps"] for row in rows)
        required = sum(row["required_document_count"] for row in rows)
        observed_durations = [row["duration_seconds"] for row in rows
                              if row["duration_observed"]]
        timing_complete = len(observed_durations) == len(rows)
        observed_total = round(sum(observed_durations), 4)
        systems[model] = {
            "question_count": len(rows),
            "submission_rate": _ratio(sum(row["submitted"] for row in rows), len(rows)),
            "valid_citation_id_rate": _ratio(
                sum(row["valid_citation_id_count"] for row in rows), total_citations
            ),
            "valid_evidence_span_rate": _ratio(
                sum(row["valid_evidence_span_count"] for row in rows), total_spans
            ),
            "questions_with_valid_evidence_rate": _ratio(
                sum(row["question_has_valid_evidence"] for row in rows), len(rows)
            ),
            "required_document_recall": _ratio(
                sum(row["required_document_hits"] for row in rows), required
            ),
            "execution_error_step_rate": _ratio(
                sum(row["execution_error_steps"] for row in rows), total_steps
            ),
            "execution_error_episode_rate": _ratio(
                sum(row["execution_error_episode"] for row in rows), len(rows)
            ),
            "execution_error_step_count": sum(row["execution_error_steps"] for row in rows),
            "execution_error_episode_count": sum(
                row["execution_error_episode"] for row in rows
            ),
            "tool_error_return_count": sum(row["tool_error_returns"] for row in rows),
            **{
                f"{category}_episode_{suffix}": (
                    count if suffix == "count" else _ratio(count, len(rows))
                )
                for category in ("tool_error", "runtime_error", "endpoint_error",
                                 "context_limit_error", "other_harness_error")
                for count in [sum(row[f"{category}_episode"] for row in rows)]
                for suffix in ("count", "rate")
            },
            **{
                f"{category}_step_{suffix}": (
                    count if suffix == "count" else _ratio(count, total_steps)
                )
                for category in ("tool_error", "runtime_error")
                for count in [sum(row[f"{category}_steps"] for row in rows)]
                for suffix in ("count", "rate")
            },
            "repeated_action_step_rate": _ratio(
                sum(row["repeated_action_steps"] for row in rows), total_exploration
            ),
            "verifier_rejections": sum(row["verifier_rejections"] for row in rows),
            "mean_trajectory_steps": _mean([row["trajectory_steps"] for row in rows]),
            "timing_complete": timing_complete,
            "timed_question_count": len(observed_durations),
            "missing_duration_question_count": len(rows) - len(observed_durations),
            "mean_duration_seconds": _mean(observed_durations) if timing_complete else None,
            "total_episode_seconds": observed_total if timing_complete else None,
            "observed_mean_duration_seconds": _mean(observed_durations),
            "observed_total_episode_seconds": observed_total,
            "question_rows": rows,
        }
    return {
        "schema_version": AUTOMATIC_VERSION,
        "benchmark_id": benchmark["benchmark_id"],
        "benchmark_hash": configuration_hash(benchmark),
        "corpus_hash": benchmark["corpus_hash"],
        "systems": systems,
        "diagnostic_definitions": {
            "execution_error": (
                "Compatibility union of returned tool errors, runtime-error steps, "
                "and failed harness/endpoint episodes; categories can overlap."
            ),
            "tool_error": (
                "Nonempty error fields in complete parsed stdout return records, "
                "not error-related words in source passages; unprinted or truncated "
                "returns cannot be counted from saved stdout."
            ),
            "context_limit_error": "Subset of endpoint-error episodes, not an additional failure.",
            "endpoint_error": (
                "Failed episodes with an explicit model-endpoint transport/response "
                "error. Unclassified failures remain other_harness_error."
            ),
            "duration": (
                "Zero recorded duration is missing timing, not a zero-cost attempt. "
                "Full means, totals and cost estimates require every question's timing. "
                "Observed fields contain only the recorded positive-duration subset."
            ),
        },
        "note": (
            "Exact spans, citation IDs, retrieval overlap, and tool errors are mechanical "
            "checks. They do not establish whether an answer is true or supported. "
            "MuSiQue reward/answer scores are intentionally excluded."
        ),
    }


def reference_review_template(benchmark: dict) -> dict:
    """A source-only sheet to complete before opening any model answers."""
    validate_benchmark(benchmark)
    return {
        "schema_version": REFERENCE_VERSION,
        "benchmark_id": benchmark["benchmark_id"],
        "benchmark_hash": configuration_hash(benchmark),
        "corpus_hash": benchmark["corpus_hash"],
        "status": "incomplete",
        "reviewer_id": "",
        "reviewed_at": "",
        "instructions": (
            "Inspect each original QASPER question, source paper, annotated answers, "
            "and evidence before opening model outputs. Mark reference_valid only when "
            "the converted frozen text supports a clear scoring interpretation; otherwise "
            "mark unscorable and explain why."
        ),
        "questions": [{
            "question_id": question["id"],
            "question": question["question"],
            "reference_answer": question.get("answer"),
            "reference_answers": question.get("reference_answers", []),
            "source_question_id": question.get("source_question_id"),
            "source_paper_id": question.get("source_paper_id"),
            "gold_evidence": question.get("gold_evidence", []),
            "decision": None,
            "reason": "",
        } for question in benchmark["questions"]],
    }


def _validated_references(benchmark: dict, sheet: dict) -> tuple[set[str], list[dict]]:
    if not isinstance(sheet, dict) or sheet.get("schema_version") != REFERENCE_VERSION:
        raise ValueError("a completed EnvoyBench reference review is required")
    for field, expected in (
        ("benchmark_id", benchmark["benchmark_id"]),
        ("benchmark_hash", configuration_hash(benchmark)),
        ("corpus_hash", benchmark["corpus_hash"]),
    ):
        if sheet.get(field) != expected:
            raise ValueError(f"reference review {field} mismatch")
    if sheet.get("status") != "complete":
        raise ValueError("reference review must be complete before model outputs are opened")
    if not isinstance(sheet.get("reviewer_id"), str) or not sheet["reviewer_id"].strip():
        raise ValueError("reference review needs a reviewer_id")
    try:
        date.fromisoformat(sheet["reviewed_at"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("reference review needs an ISO reviewed_at date") from exc
    items = sheet.get("questions")
    if not isinstance(items, list):
        raise ValueError("reference review questions must be a list")
    expected_ids = {question["id"] for question in benchmark["questions"]}
    seen = set()
    scorable = set()
    excluded = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("reference review row must be an object")
        question_id = item.get("question_id")
        if question_id not in expected_ids or question_id in seen:
            raise ValueError("reference review has duplicate or unknown question ID")
        seen.add(question_id)
        decision = item.get("decision")
        if decision == "reference_valid":
            scorable.add(question_id)
        elif decision == "unscorable":
            reason = item.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(f"{question_id} needs an unscorable reason")
            excluded.append({"question_id": question_id, "reason": reason.strip()})
        else:
            raise ValueError(f"{question_id} needs reference_valid or unscorable")
    if seen != expected_ids:
        raise ValueError("reference review must cover every benchmark question")
    if not scorable:
        raise ValueError("reference review has no scorable questions")
    return scorable, excluded


def prepare_review_bundle(
    benchmark: dict,
    run_manifest: dict,
    results: list[dict],
    corpus_dir: Path,
    reference_validation: dict,
    *,
    seed: int = 42,
) -> tuple[dict, dict, dict]:
    """Build a complete paired, identity-blinded review from frozen run artifacts."""
    documents, models = _verify_artifacts(benchmark, run_manifest, results, corpus_dir)
    scorable, excluded = _validated_references(benchmark, reference_validation)
    return _prepare_bundle(
        benchmark, run_manifest, results, documents, models, scorable, excluded,
        reference_review_hash=configuration_hash(reference_validation),
        source_reference_status="human_validated", seed=seed,
    )


def prepare_provisional_bundle(
    benchmark: dict,
    run_manifest: dict,
    results: list[dict],
    corpus_dir: Path,
    *,
    seed: int = 42,
) -> tuple[dict, dict, dict]:
    """Blind a paired run using QASPER references that no human has validated here.

    This permits an explicitly model-assisted development comparison. It cannot
    produce a human-reviewed answer score or pass the held-out promotion gate.
    """
    documents, models = _verify_artifacts(benchmark, run_manifest, results, corpus_dir)
    question_ids = {question["id"] for question in benchmark["questions"]}
    return _prepare_bundle(
        benchmark, run_manifest, results, documents, models, question_ids, [],
        reference_review_hash=None,
        source_reference_status="unreviewed_qasper", seed=seed,
    )


def _prepare_bundle(
    benchmark: dict,
    run_manifest: dict,
    results: list[dict],
    documents: dict[str, dict],
    models: list[str],
    included_question_ids: set[str],
    excluded: list[dict],
    *,
    reference_review_hash: str | None,
    source_reference_status: str,
    seed: int,
) -> tuple[dict, dict, dict]:
    questions = [question for question in benchmark["questions"]
                 if question["id"] in included_question_ids]
    by_pair = {(row["model_key"], row["question_id"]): row for row in results}
    rng = random.Random(seed)
    review_rows = []
    assignments = []
    for question in questions:
        order = models[:]
        rng.shuffle(order)
        for model in order:
            row = by_pair[(model, question["id"])]
            blind_id = f"R{len(review_rows) + 1:03d}"
            review_row = {
                "blind_id": blind_id,
                "question_id": question["id"],
                "question": question["question"],
                "expected_answerability": question["expected_answerability"],
                "reference_answer": question.get("answer"),
                "grader_notes": question.get("grader_notes", []),
                "status": row["status"],
                "answer": row["predicted_answer"],
                "citations": row["predicted_citations"],
                "evidence": _strict_evidence(row["predicted_evidence"], documents),
                "verdict": None,
                "notes": "",
                "relevant_source_passage": "",
            }
            review_rows.append(review_row)
            assignments.append({
                "blind_id": blind_id,
                "question_id": question["id"],
                "system": model,
                "result_hash": configuration_hash(row),
                "review_content_hash": configuration_hash(_review_content(review_row)),
            })
    binding = {
        "benchmark_id": benchmark["benchmark_id"],
        "benchmark_hash": configuration_hash(benchmark),
        "corpus_hash": benchmark["corpus_hash"],
        "results_hash": configuration_hash({"results": results}),
        "reference_review_hash": reference_review_hash,
    }
    review = {
        "schema_version": REVIEW_VERSION,
        **binding,
        "status": "incomplete",
        "reviewer_kind": (
            "model_assisted" if source_reference_status == "unreviewed_qasper" else None
        ),
        "reviewer_id": "",
        "reviewed_at": "",
        "rubric_version": RUBRIC_VERSION,
        "adjudications": [],
        "excluded_questions": excluded,
        "instructions": {
            "pass": (
                "All important parts correct, material claims supported by paper text, "
                "honest limits; appropriate explanation when evidence is insufficient."
            ),
            "partial": (
                "Useful supported part, but an important requested detail or qualification "
                "is missing; no material false or unsupported claim."
            ),
            "fail": (
                "Wrong or materially unsupported claim, misleading citation, invented result, "
                "false refusal, or no useful answer."
            ),
            "reviewer_kind": (
                "Set to human for a completed source-checked human review, or "
                "model_assisted for explicitly provisional judgments. An unreviewed "
                "QASPER reference bundle cannot be marked human-reviewed."
            ),
            "partial_or_fail": (
                "Record a short reason in notes and a relevant frozen-source passage "
                "or explain why no passage exists."
            ),
            "blinding": (
                "Keep blind-key.json and automatic.json closed until verdicts are locked. "
                "Model names, training stage, reward, and aggregate results are withheld here."
            ),
        },
        "rows": review_rows,
    }
    if source_reference_status == "unreviewed_qasper":
        # The human review editor has a deliberately exact schema. Keep that
        # path unchanged; provisional bundles are filled by a separate grader.
        review["source_reference_status"] = source_reference_status
        review["instructions"]["reference_warning"] = (
            "Original QASPER reference annotations have not been independently "
            "validated for this benchmark. Grade provisionally as model_assisted; "
            "do not report these verdicts as human-reviewed or source-validated."
        )
    key = {"schema_version": KEY_VERSION, **binding,
           "seed": seed,
           "excluded_questions": excluded,
           "assignments": assignments}
    if source_reference_status == "unreviewed_qasper":
        key["source_reference_status"] = source_reference_status
    automatic = _automatic_score(benchmark, results, documents, models)
    for item in run_manifest.get("models", []):
        if not isinstance(item, dict) or item.get("key") not in automatic["systems"]:
            continue
        price = item.get("serving_hourly_usd")
        system = automatic["systems"][item["key"]]
        system["estimated_episode_compute_usd"] = (
            round(system["total_episode_seconds"] * price / 3600, 6)
            if type(price) in {int, float} and isfinite(price) and price >= 0
            and system["timing_complete"] else None
        )
        system["estimated_observed_episode_compute_usd"] = (
            round(system["observed_total_episode_seconds"] * price / 3600, 6)
            if type(price) in {int, float} and isfinite(price) and price >= 0 else None
        )
    automatic["cost_estimate_note"] = (
        "Estimated episode compute cost is the sum of client-observed episode "
        "durations times the declared GPU hourly rate; incomplete timing makes "
        "the full estimate null, and observed cost is only a partial subtotal. "
        "It excludes server startup, idle time, shared utilization, and token "
        "charges; it is not a bill. New harness timings include policy creation "
        "and episode reset; historical successful timings excluded reset."
    )
    automatic["results_hash"] = binding["results_hash"]
    automatic["reference_review_hash"] = binding["reference_review_hash"]
    automatic["source_reference_status"] = source_reference_status
    automatic["question_ids_included"] = sorted(included_question_ids)
    automatic["scorable_question_ids"] = (
        sorted(included_question_ids) if source_reference_status == "human_validated" else None
    )
    automatic["excluded_questions"] = excluded
    if source_reference_status == "unreviewed_qasper":
        automatic["note"] += (
            " QASPER source references are included without independent human "
            "validation; any model-graded answer score is provisional."
        )
    automatic["run_identity"] = {
        "run_id": run_manifest.get("run_id"),
        "comparison_id": run_manifest.get("comparison_id"),
        "split": run_manifest.get("split"),
        "split_status": run_manifest.get("split_status"),
        "seed": run_manifest.get("seed"),
        "max_steps": run_manifest.get("max_steps"),
        "models": run_manifest.get("models"),
        "runner_hardware": run_manifest.get("runner_hardware"),
        "serving_hardware": {
            item["key"]: item.get("serving_hardware")
            for item in run_manifest.get("models", [])
            if isinstance(item, dict) and "key" in item
        },
    }
    return review, key, automatic


def _verify_result_assignments(review: dict, assignments: list[dict],
                               results: list[dict]) -> None:
    """Bind every blind assignment to its exact saved result and model identity."""
    if not isinstance(results, list) or configuration_hash({"results": results}) != review.get(
        "results_hash"
    ):
        raise ValueError("saved results do not match the blind review results_hash")
    by_pair: dict[tuple[str, str], dict] = {}
    for result in results:
        if not isinstance(result, dict):
            raise ValueError("saved results must be objects")
        pair = (result.get("model_key"), result.get("question_id"))
        if (not all(isinstance(value, str) and value for value in pair)
                or pair in by_pair):
            raise ValueError("saved results have duplicate or invalid model/question pairs")
        by_pair[pair] = result
    seen: set[tuple[str, str]] = set()
    for assignment in assignments:
        pair = (assignment.get("system"), assignment.get("question_id"))
        if (not all(isinstance(value, str) and value for value in pair)
                or pair in seen or pair not in by_pair
                or assignment.get("result_hash") != configuration_hash(by_pair[pair])):
            raise ValueError("blind assignment does not match its saved model result")
        seen.add(pair)


def score_completed_review(
    review: dict, key: dict, *, results: list[dict] | None = None,
) -> dict:
    """Aggregate paired verdicts, keeping model-assisted review out of the human gate.

    Supply ``results`` to verify blind-key assignments against the frozen run.
    This is mandatory for model-assisted scores and optional for legacy human
    review bundles.
    """
    if review.get("schema_version") != REVIEW_VERSION or key.get("schema_version") != KEY_VERSION:
        raise ValueError("review and blind key have incompatible schemas")
    for field in ("benchmark_id", "benchmark_hash", "corpus_hash", "results_hash",
                  "reference_review_hash"):
        if review.get(field) != key.get(field):
            raise ValueError(f"review and blind key {field} mismatch")
    source_status = review.get("source_reference_status", "human_validated")
    if (source_status not in {"human_validated", "unreviewed_qasper"}
            or source_status != key.get("source_reference_status", "human_validated")):
        raise ValueError("review and blind key need matching source reference provenance")
    reference_hash = review.get("reference_review_hash")
    if (source_status == "human_validated") != bool(
        isinstance(reference_hash, str) and re.fullmatch(r"[0-9a-f]{64}", reference_hash)
    ):
        raise ValueError("source reference provenance and review hash disagree")
    if review.get("status") != "complete":
        raise ValueError("review must be marked complete before scoring")
    kind = review.get("reviewer_kind")
    if kind not in {"human", "model_assisted"}:
        raise ValueError("reviewer_kind must be human or model_assisted")
    if source_status == "unreviewed_qasper" and kind == "human":
        raise ValueError("unreviewed QASPER references cannot yield a human-reviewed score")
    if kind == "model_assisted" and results is None:
        raise ValueError("model-assisted scoring requires saved results for assignment checks")
    if not isinstance(review.get("reviewer_id"), str) or not review["reviewer_id"].strip():
        raise ValueError("completed review needs a reviewer_id")
    try:
        date.fromisoformat(review["reviewed_at"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("completed review needs an ISO reviewed_at date") from exc
    if review.get("rubric_version") != RUBRIC_VERSION:
        raise ValueError("review rubric version is unknown")
    rows = review.get("rows")
    assignments = key.get("assignments")
    if not isinstance(rows, list) or not isinstance(assignments, list):
        raise ValueError("review rows and blind assignments must be lists")
    row_ids = [row.get("blind_id") for row in rows if isinstance(row, dict)]
    key_ids = [item.get("blind_id") for item in assignments if isinstance(item, dict)]
    if (len(row_ids) != len(rows) or len(key_ids) != len(assignments)
            or len(set(row_ids)) != len(row_ids) or len(set(key_ids)) != len(key_ids)
            or set(row_ids) != set(key_ids)):
        raise ValueError("review and key must contain each blind ID exactly once")
    if review.get("excluded_questions") != key.get("excluded_questions"):
        raise ValueError("review exclusion list changed after blinding")
    if results is not None:
        _verify_result_assignments(review, assignments, results)
    assignments_by_id = {item["blind_id"]: item for item in assignments}
    for row in rows:
        assignment = assignments_by_id[row["blind_id"]]
        if (row.get("question_id") != assignment.get("question_id")
                or configuration_hash(_review_content(row)) != assignment.get(
                    "review_content_hash"
                )):
            raise ValueError(f"{row['blind_id']} blind review content changed")
        if row.get("verdict") in {"partial", "fail"}:
            if not isinstance(row.get("notes"), str) or not row["notes"].strip():
                raise ValueError(f"{row['blind_id']} needs a reason for partial/fail")
            passage = row.get("relevant_source_passage")
            if not isinstance(passage, str) or not passage.strip():
                raise ValueError(
                    f"{row['blind_id']} needs a relevant source passage or absence note"
                )
    raw = score_review(review, key)
    if kind == "model_assisted":
        raw["schema_version"] = PROVISIONAL_SCORE_VERSION
        raw["decision_note"] = (
            "These are provisional model-graded verdicts against unverified source "
            "references unless separately validated. They cannot satisfy the "
            "human-reviewed model-promotion gate."
        )
    by_question: dict[str, dict[str, str]] = defaultdict(dict)
    assignment_by_id = {item["blind_id"]: item for item in assignments}
    for row in rows:
        assignment = assignment_by_id[row["blind_id"]]
        by_question[assignment["question_id"]][assignment["system"]] = row["verdict"]
    raw["paired_pass_tests"] = {}
    for left, right in combinations(sorted(raw["systems"]), 2):
        left_only = sum(
            outcomes[left] == "pass" and outcomes[right] != "pass"
            for outcomes in by_question.values()
        )
        right_only = sum(
            outcomes[right] == "pass" and outcomes[left] != "pass"
            for outcomes in by_question.values()
        )
        discordant = left_only + right_only
        tail = min(left_only, right_only)
        exact_p = min(
            1.0,
            2 * sum(comb(discordant, k) for k in range(tail + 1)) / (2 ** discordant),
        )
        raw["paired_pass_tests"][f"{left}_vs_{right}"] = {
            "left_only_pass": left_only,
            "right_only_pass": right_only,
            "pass_rate_difference_right_minus_left": round(
                (right_only - left_only) / len(by_question), 4
            ),
            "exact_mcnemar_two_sided_p": round(exact_p, 6),
            "scorable_question_count": len(by_question),
        }
    result = {
        "schema_version": REVIEWED_VERSION,
        "benchmark_id": review["benchmark_id"],
        "benchmark_hash": review["benchmark_hash"],
        "corpus_hash": review["corpus_hash"],
        "results_hash": review["results_hash"],
        "reference_review_hash": review["reference_review_hash"],
        "source_reference_status": source_status,
        "result_bindings_verified": results is not None,
        "review_provenance": {"kind": kind, "reviewer_id": review["reviewer_id"].strip()},
        "reviewed_at": review["reviewed_at"],
        "rubric_version": review["rubric_version"],
        "excluded_questions": review.get("excluded_questions", []),
        "human": raw if kind == "human" else None,
        "provisional_model_assisted": raw if kind == "model_assisted" else None,
        "note": (
            "Human supported-answer pass rates require validated source references and "
            "source-checked human verdicts. A model-assisted score is provisional and "
            "cannot pass that gate."
        ),
    }
    return result


def _write_new(path: Path, value: dict | str) -> None:
    with path.open("x", encoding="utf-8") as stream:
        if isinstance(value, str):
            stream.write(value)
        else:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write("\n")


def _load_frozen_cli_benchmark(path: Path, corpus: Path | None = None) -> dict:
    """Bind a CLI review to the tracked split and regenerated corpus hashes."""
    dataset = path.resolve().parent.parent
    split = path.resolve().parent.name
    benchmark, _, benchmark_path, corpus_path, _ = load_split(dataset, split)
    if benchmark_path.resolve() != path.resolve():
        raise ValueError("benchmark path does not match frozen split")
    if corpus is not None and corpus_path.resolve() != corpus.resolve():
        raise ValueError("corpus path does not match frozen split")
    return benchmark


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Create a blind review bundle")
    prepare.add_argument("--benchmark", type=Path, required=True)
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--results", type=Path, required=True)
    prepare.add_argument("--corpus", type=Path, required=True)
    prepare.add_argument("--reference-review", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--seed", type=int, default=42)
    provisional = commands.add_parser(
        "prepare-provisional",
        help="Create a blind model-graded bundle using unreviewed QASPER references",
    )
    for name in ("benchmark", "manifest", "results", "corpus", "output"):
        provisional.add_argument(f"--{name}", type=Path, required=True)
    provisional.add_argument("--seed", type=int, default=42)
    complete = commands.add_parser("score", help="Score a completed review")
    complete.add_argument("--review", type=Path, required=True)
    complete.add_argument("--key", type=Path, required=True)
    complete.add_argument(
        "--results", type=Path,
        help="Saved results.json; required for model-assisted scores and checks blind assignments",
    )
    complete.add_argument("--output", type=Path, required=True)
    reference = commands.add_parser(
        "reference-template", help="Create source-only reference validation sheet"
    )
    reference.add_argument("--benchmark", type=Path, required=True)
    reference.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "reference-template":
        benchmark = _load_frozen_cli_benchmark(args.benchmark)
        _write_new(args.output, reference_review_template(benchmark))
        print(args.output)
        return 0
    if args.command in {"prepare", "prepare-provisional"}:
        benchmark = _load_frozen_cli_benchmark(args.benchmark, args.corpus)
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        benchmark_file_hash = hashlib.sha256(args.benchmark.read_bytes()).hexdigest()
        if benchmark_file_hash != manifest.get("benchmark_file_sha256"):
            raise ValueError("run manifest benchmark file hash differs from frozen split")
        results = json.loads(args.results.read_text(encoding="utf-8"))
        if args.command == "prepare":
            reference_validation = json.loads(args.reference_review.read_text(encoding="utf-8"))
            review, key, automatic = prepare_review_bundle(
                benchmark, manifest, results, args.corpus, reference_validation,
                seed=args.seed,
            )
        else:
            review, key, automatic = prepare_provisional_bundle(
                benchmark, manifest, results, args.corpus, seed=args.seed,
            )
        args.output.mkdir(parents=True, exist_ok=False)
        _write_new(args.output / "review.json", review)
        _write_new(args.output / "blind-key.json", key)
        _write_new(args.output / "automatic.json", automatic)
        _write_new(args.output / "review.md", review_markdown(review))
        print(args.output)
        return 0
    review = json.loads(args.review.read_text(encoding="utf-8"))
    key = json.loads(args.key.read_text(encoding="utf-8"))
    results = json.loads(args.results.read_text(encoding="utf-8")) if args.results else None
    _write_new(args.output, score_completed_review(review, key, results=results))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
