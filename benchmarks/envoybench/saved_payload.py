"""Build and validate saved Studio payloads without starting the web app or a model."""

from __future__ import annotations

import json
import math
from pathlib import Path

from benchmarks.envoybench.diagnostics import (
    case_diagnostics,
    prepared_retrieval_status,
    refusal_baseline,
    response_behavior,
    unknown_diagnostics,
)
from benchmarks.envoybench.frozen_dataset import DEFAULT_DATASET, _file_sha256, load_split
from benchmarks.envoybench.promotion import evaluate_promotion
from benchmarks.envoybench.score import (
    _automatic_score,
    _mechanical_row,
    _strict_evidence,
    _validated_references,
    _verify_artifacts,
    score_completed_review,
)
from src.eval.artifacts import configuration_hash

HERE = Path(__file__).resolve().parent

PRIOR_ART = [
    {
        "name": "QASPER",
        "url": "https://arxiv.org/abs/2105.03011",
        "overlap": (
            "The paper questions, answerability labels, and source annotations "
            "originate here."
        ),
        "difference": "EnvoyBench adapts selected questions into bounded code-execution episodes.",
    },
    {
        "name": "Qwen paper-QA LoRA",
        "url": "https://huggingface.co/whosouravsharma/paper-qa-lora",
        "overlap": "A Qwen adapter has already been trained and evaluated on QASPER questions.",
        "difference": (
            "This comparison includes the agent's search/code trajectory and "
            "requires source-support review."
        ),
    },
    {
        "name": "Aviary / LitQA2",
        "url": "https://arxiv.org/html/2412.21154",
        "overlap": "Scientific-literature agents were trained and compared with untrained agents.",
        "difference": (
            "EnvoyBench uses open-ended QASPER answers and model-authored "
            "Python actions."
        ),
    },
    {
        "name": "PaperArena",
        "url": "https://arxiv.org/abs/2510.10909",
        "overlap": "Multi-step literature QA already includes a code executor and trace analysis.",
        "difference": "Python over the frozen paper tools is EnvoyBench's main action interface.",
    },
    {
        "name": "ResearchQA",
        "url": "https://arxiv.org/html/2607.11074v1",
        "overlap": "Open-ended scientific-paper answers already have quote and refusal checks.",
        "difference": "The bounded tool trajectory is part of the measured behavior here.",
    },
    {
        "name": "AgentHop",
        "url": "https://arxiv.org/html/2609.34428v1",
        "overlap": "Scientific-paper agents already have multi-tool diagnostic evaluation.",
        "difference": "The answer is open-ended and reviewed for source support.",
    },
    {
        "name": "AstaBench",
        "url": "https://github.com/allenai/asta-bench",
        "overlap": "Science-agent benchmarks already use sandboxed code, logs, and cost summaries.",
        "difference": (
            "EnvoyBench is a small paired base-versus-adapter audit on one "
            "evidence task."
        ),
    },
    {
        "name": "Inspect View",
        "url": "https://inspect.aisi.org.uk/log-viewer.html",
        "overlap": "Per-sample trace, score, and metadata inspection is established practice.",
        "difference": "This demo narrows the view to paper-evidence failures in this repository.",
    },
]


def _read_json(path: Path) -> dict | list:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict | list):
        raise ValueError(f"Expected JSON object or list: {path}")
    return value


def _amended_continuation(manifest: dict) -> dict | None:
    """Identify the derived Nebius run from its saved protocol and episode lineage."""
    amendment = manifest.get("protocol_amendment")
    lineage = manifest.get("lineage")
    if amendment is None and lineage is None:
        return None
    if (not isinstance(amendment, dict) or not isinstance(lineage, dict)
            or amendment.get("schema_version") != "qasper-nebius-continuation-merge-v1"
            or lineage.get("schema_version") != amendment["schema_version"]
            or amendment.get("kind") != "cumulative_budget_increase_and_question_39_restart"):
        raise ValueError("Unsupported saved protocol amendment or lineage")
    original, continuation = lineage.get("original"), lineage.get("continuation")
    ids = manifest.get("question_ids")
    if (not isinstance(original, dict) or not isinstance(continuation, dict)
            or not isinstance(ids, list) or len(ids) != 40
            or original.get("retained_question_ids") != ids[:38]
            or continuation.get("question_ids") != ids[38:]
            or not isinstance(original.get("interrupted_question_39"), dict)
            or original["interrupted_question_39"].get("question_id") != ids[38]
            or amendment.get("original_question_39_partial_trace_excluded_from_derived_result")
            is not True
            or amendment.get("original_no_retry_protocol_amended") is not True
            or amendment.get("question_39_restarted_at_step") != 1
            or amendment.get("first_38_source") != "original completed episodes"
            or amendment.get("question_39_source") != "continuation retry"
            or amendment.get("question_40_source") != "continuation first attempt"
            or not isinstance(original.get("run_id"), str)
            or not isinstance(continuation.get("run_id"), str)):
        raise ValueError("Saved protocol amendment does not match the episode lineage")
    old_cap = amendment.get("original_local_estimated_usd_cap")
    new_cap = amendment.get("amended_cumulative_estimated_usd_cap")
    if (type(old_cap) not in {int, float} or type(new_cap) not in {int, float}
            or not math.isfinite(old_cap) or not math.isfinite(new_cap)
            or not 0 < old_cap < new_cap):
        raise ValueError("Saved protocol amendment has invalid cost caps")
    return {
        "kind": "amended_continuation",
        "original_completed_count": len(original["retained_question_ids"]),
        "original_budget_cap_usd": old_cap,
        "amended_budget_cap_usd": new_cap,
        "original_run_id": original["run_id"],
        "continuation_run_id": continuation["run_id"],
        "restarted_question_id": ids[38],
        "first_attempt_question_id": ids[39],
        "original_partial_excluded": True,
    }


def _token_diagnostic_payload(run_dir: Path, dataset: Path) -> dict:
    """Display a complete paired dev smoke separately from benchmark grades."""
    paths = [run_dir / "manifest.json", run_dir / "results.json"]
    if any(path.stat().st_size > 10 * 1024 * 1024 for path in paths):
        raise ValueError("Token diagnostic exceeds the 10 MiB display limit")
    manifest, results = (_read_json(path) for path in paths)
    if not isinstance(manifest, dict) or not isinstance(results, list):
        raise ValueError("Token diagnostic needs a manifest and result rows")
    if (manifest.get("schema_version") != "envoybench-run-v1"
            or manifest.get("status") != "complete" or manifest.get("split") != "dev"
            or manifest.get("subset_smoke") is not True
            or manifest.get("full_split") is not False):
        raise ValueError("Token diagnostic requires a complete development subset smoke")
    benchmark, _, _, _, _ = load_split(dataset, "dev")
    for field, expected in (
        ("benchmark_id", benchmark["benchmark_id"]),
        ("benchmark_hash", configuration_hash(benchmark)),
        ("corpus_hash", benchmark["corpus_hash"]),
    ):
        if manifest.get(field) != expected:
            raise ValueError(f"Token diagnostic {field} differs from frozen development data")
    questions = {question["id"]: question for question in benchmark["questions"]}
    ids = manifest.get("question_ids")
    if (not isinstance(ids, list) or not ids or not all(isinstance(q, str) for q in ids)
            or len(ids) != len(set(ids)) or not set(ids).issubset(questions)):
        raise ValueError("Token diagnostic has invalid question IDs")
    identities = manifest.get("models")
    if not isinstance(identities, list) or not all(isinstance(m, dict) for m in identities):
        raise ValueError("Token diagnostic has invalid model identities")
    keys = [model.get("key") for model in identities]
    if (len(keys) < 2 or not all(isinstance(key, str) and key for key in keys)
            or len(keys) != len(set(keys))):
        raise ValueError("Token diagnostic needs distinct paired models")
    pairs = {}
    for row in results:
        if not isinstance(row, dict):
            raise ValueError("Token diagnostic has an invalid result row")
        key, question_id = row.get("model_key"), row.get("question_id")
        if key not in keys or question_id not in ids or (question_id, key) in pairs:
            raise ValueError("Token diagnostic has an unknown or duplicate model/question pair")
        if row.get("status") != "submitted" or row.get("error"):
            raise ValueError("Token diagnostic requires submissions from every paired model")
        if row.get("question") != questions[question_id]["question"]:
            raise ValueError("Token diagnostic question text differs from frozen data")
        for field in ("schema_version", "run_id", "comparison_id", "benchmark_id",
                      "benchmark_hash", "corpus_hash", "split", "split_status"):
            if row.get(field) != manifest.get(field):
                raise ValueError(f"Token diagnostic row {field} differs from its manifest")
        trajectory = row.get("trajectory")
        if not isinstance(trajectory, list) or not trajectory:
            raise ValueError("Token diagnostic needs saved turns")
        for step in trajectory:
            diagnostic = step.get("logprob_diagnostics") if isinstance(step, dict) else None
            if (not isinstance(diagnostic, dict)
                    or diagnostic.get("schema_version") != "sampled-content-logprobs-v1"):
                raise ValueError("Token diagnostic turn is missing generated-token telemetry")
            reported, valid = (diagnostic.get(name) for name in (
                "reported_token_count", "valid_logprob_count"
            ))
            prefix = diagnostic.get("prefix")
            if (type(reported) is not int or type(valid) is not int
                    or not 0 < valid <= reported or not isinstance(prefix, list)
                    or len(prefix) > min(64, reported)
                    or diagnostic.get("prefix_truncated") is not (reported > len(prefix))):
                raise ValueError("Token diagnostic has invalid token coverage")
            for token in prefix:
                if not isinstance(token, dict):
                    raise ValueError("Token diagnostic has an invalid saved token")
                value = token.get("logprob")
                if value is not None and (
                    type(value) not in {int, float} or not math.isfinite(value) or value > 0
                ):
                    raise ValueError("Token diagnostic has an invalid token logprob")
        pairs[(question_id, key)] = {
            "status": row["status"], "answer": row.get("predicted_answer", ""),
            "trajectory": trajectory, "duration_seconds": row.get("duration_seconds"),
        }
    if set(pairs) != {(question_id, key) for question_id in ids for key in keys}:
        raise ValueError("Token diagnostic has missing paired submissions")
    return {
        "schema_version": "envoybench-token-diagnostic-v1",
        "run_id": manifest["run_id"], "comparison_id": manifest["comparison_id"],
        "created_at_utc": manifest.get("created_at_utc"), "split": "dev",
        "question_count": len(ids), "status": "complete_paired_smoke",
        "scope": "Generated-token likelihood only; no answer-quality or calibration score.",
        "source_sha256": {path.name: _file_sha256(path) for path in paths},
        "models": [{name: identity.get(name) for name in (
            "key", "model_id", "revision", "adapter_id", "adapter_revision",
            "identity_source", "serving_runtime", "serving_hardware", "decoding", "extra_body",
        )} for identity in identities],
        "cases": [{"id": q, "question": questions[q]["question"],
                   "answerability": questions[q].get("expected_answerability"),
                   "systems": {key: pairs[(q, key)] for key in keys}} for q in ids],
    }


def _load_web_evidence_report(path: Path) -> dict:
    """Read one completed local pilot artifact; never fetch or recompute it."""
    if not path.is_file():
        raise ValueError(f"Web-evidence report is not a local file: {path}")
    if path.stat().st_size > 10 * 1024 * 1024:
        raise ValueError("Web-evidence report exceeds the 10 MiB display limit")
    report = _read_json(path)
    if not isinstance(report, dict):
        raise ValueError("Web-evidence report must be a JSON object")
    if report.get("schema_version") != "web-evidence-report-v1":
        raise ValueError("Unsupported web-evidence report schema")
    if report.get("status") != "complete":
        raise ValueError("Web-evidence report must be complete before display")
    if not isinstance(report.get("aggregate"), dict) or not isinstance(report.get("cases"), list):
        raise ValueError("Web-evidence report is missing aggregate metrics or cases")
    if any(
        not isinstance(case, dict)
        or not isinstance(case.get("facts"), list)
        or any(not isinstance(fact, dict) for fact in case["facts"])
        for case in report["cases"]
    ):
        raise ValueError("Web-evidence report contains an invalid case")
    return report


def _candidate_summary(dataset: Path) -> dict:
    index = _read_json(dataset / "manifest.json")
    if not isinstance(index, dict):
        raise ValueError("EnvoyBench data index must be an object")
    entry = index["splits"]["test_candidate"]
    return {
        "question_count": entry["question_count"],
        "corpus_paper_count": entry["corpus_paper_count"],
        "corpus_hash": entry["corpus_hash"],
        "status": entry["status"],
        "reference_review_status": entry["human_review_status"],
        "score": None,
    }


def _excerpt(value: object, limit: int = 1800) -> tuple[str, bool]:
    text = str(value or "")
    return text[:limit], len(text) > limit


def _review_annotations(
    review_dir: Path, results: list[dict], benchmark: dict,
    judged_review_path: Path | None = None,
) -> tuple[dict, dict]:
    review = _read_json(judged_review_path or review_dir / "review.json")
    key = _read_json(review_dir / "blind-key.json")
    if not isinstance(review, dict) or not isinstance(key, dict):
        raise ValueError("Review and blind key must be JSON objects")
    if review.get("results_hash") != configuration_hash({"results": results}):
        raise ValueError("Review bundle does not match the selected run results")
    if review.get("benchmark_hash") != configuration_hash(benchmark):
        raise ValueError("Review bundle does not match the frozen benchmark")
    scored = score_completed_review(review, key, results=results)
    assignments = {item["blind_id"]: item for item in key["assignments"]}
    annotations = {}
    for row in review["rows"]:
        assignment = assignments[row["blind_id"]]
        annotations[(assignment["question_id"], assignment["system"])] = {
            "verdict": row["verdict"],
            "notes": row.get("notes", ""),
        }
    return scored, annotations


def _run_payload(
    run_dir: Path, dataset: Path, review_dir: Path | None,
    reference_review_path: Path | None, *, provisional: bool = False,
    judged_review_path: Path | None = None,
    verified_trace_store: dict[tuple[str, ...], dict] | None = None,
    supplementary: bool = False,
) -> dict:
    manifest = _read_json(run_dir / "manifest.json")
    incomplete = isinstance(manifest, dict) and manifest.get("status") == "incomplete"
    if incomplete and (not supplementary or review_dir is not None
                       or judged_review_path is not None or reference_review_path is not None):
        raise ValueError("incomplete runs are supplementary trace inspection only")
    results_path = run_dir / ("results.partial.json" if incomplete else "results.json")
    if incomplete and (run_dir / "results.json").exists():
        raise ValueError("incomplete run must not also claim completed results")
    results = _read_json(results_path)
    if not isinstance(manifest, dict) or not isinstance(results, list):
        raise ValueError("Run manifest/results have invalid JSON shape")
    amended_continuation = _amended_continuation(manifest)
    if amended_continuation is not None and (incomplete or not supplementary):
        raise ValueError("Amended continuation requires a complete supplementary run")
    split = manifest.get("split")
    if split not in {"dev", "test_candidate"}:
        raise ValueError("Run split must be dev or test_candidate")
    benchmark, _, _, corpus_path, _ = load_split(dataset, split)
    documents, model_keys = _verify_artifacts(
        benchmark, manifest, results, corpus_path, require_paired=not supplementary,
        allow_incomplete=incomplete,
    )
    if incomplete:
        if len(model_keys) != 1:
            raise ValueError("incomplete supplementary inspection needs one declared model")
        questions_by_id = {question["id"]: question for question in benchmark["questions"]}
        automatic = {
            "schema_version": "envoybench-incomplete-inspection-v1",
            "scope": "Per-episode diagnostics only; aggregate scores withheld for incomplete run",
            "systems": {key: {
                "submission_rate": None, "valid_evidence_span_rate": None,
                "execution_error_episode_rate": None,
                "question_rows": [
                    _mechanical_row(row, questions_by_id[row["question_id"]], documents)
                    for row in results if row["model_key"] == key
                ],
            } for key in model_keys},
        }
        components, featured = {}, []
        component_provenance = {"reviewed_question_count": 0}
        behavior, behavior_metrics, behavior_provenance = {}, {}, None
    else:
        automatic = _automatic_score(benchmark, results, documents, model_keys)
        components, featured, component_provenance = case_diagnostics(
            benchmark, results, documents
        )
        behavior, behavior_metrics, behavior_provenance = response_behavior(benchmark, results)
    mechanical = {
        (row["question_id"], model): row
        for model, system in automatic["systems"].items()
        for row in system["question_rows"]
    }
    scored, annotations = (None, {})
    if review_dir is not None:
        scored, annotations = _review_annotations(
            review_dir, results, benchmark, judged_review_path
        )
    judge_provenance = None
    if split == "test_candidate":
        if provisional:
            if scored and not scored.get("provisional_model_assisted"):
                raise ValueError("provisional mode requires model-assisted verdicts")
            if scored:
                if judged_review_path is None:
                    raise ValueError("scored provisional view requires --judged-review")
                judge_report = _read_json(judged_review_path.parent / "judge-report.json")
                if not isinstance(judge_report, dict):
                    raise ValueError("judge report must be an object")
                binding = {
                    "judged_review_hash": configuration_hash(_read_json(judged_review_path)),
                    "benchmark_hash": configuration_hash(benchmark),
                    "corpus_hash": benchmark["corpus_hash"],
                    "results_hash": configuration_hash({"results": results}),
                    "source_reference_status": "unreviewed_qasper",
                    "status": "complete",
                }
                if any(judge_report.get(field) != value for field, value in binding.items()):
                    raise ValueError("judge report does not match the selected provisional run")
                judge_provenance = {
                    "requested_model": judge_report.get("requested_model"),
                    "reported_models": judge_report.get("reported_models"),
                    "system_prompt_sha256": judge_report.get("system_prompt_sha256"),
                    "input_tokens": judge_report.get("input_tokens"),
                    "output_tokens": judge_report.get("output_tokens"),
                    "report_hash": configuration_hash(judge_report),
                    "limitations": judge_report.get("limitations"),
                }
        elif not (scored or {}).get("human"):
            raise ValueError(
                "test_candidate model identities stay closed until blind human "
                "answer review is complete"
            )
    promotion = None
    if split == "test_candidate" and not provisional:
        if review_dir is None or reference_review_path is None:
            raise ValueError("completed candidate review artifacts are required")
        promotion = evaluate_promotion(
            scored,
            _read_json(review_dir / "automatic.json"),
            _read_json(reference_review_path),
        )
    reviewed = (scored or {}).get("human") or (scored or {}).get("provisional_model_assisted")
    model_labels = {
        item["key"]: item.get("label") or item["key"].replace("_", " ").title()
        for item in manifest["models"]
    }
    models = []
    for item in manifest["models"]:
        key = item["key"]
        metrics = automatic["systems"][key]
        timed = [
            row["duration_seconds"] for row in results
            if row["model_key"] == key and row["duration_seconds"] > 0
        ]
        semantic = (reviewed or {}).get("systems", {}).get(key, {})
        verdicts_by_answerability = {}
        for category in ("sufficient", "insufficient"):
            verdicts_by_answerability[category] = [
                annotations[(question["id"], key)]["verdict"]
                for question in benchmark["questions"]
                if question["expected_answerability"] == category
                and (question["id"], key) in annotations
            ]
        models.append({
            "key": key,
            "label": model_labels[key],
            "metrics": {
                "pass": semantic.get("pass"),
                "partial": semantic.get("partial"),
                "fail": semantic.get("fail"),
                "answerable_pass": (
                    verdicts_by_answerability["sufficient"].count("pass") if reviewed else None
                ),
                "answerable_reviewed": (
                    len(verdicts_by_answerability["sufficient"]) if reviewed else None
                ),
                "unanswerable_pass": (
                    verdicts_by_answerability["insufficient"].count("pass") if reviewed else None
                ),
                "unanswerable_reviewed": (
                    len(verdicts_by_answerability["insufficient"]) if reviewed else None
                ),
                "submission_rate": metrics["submission_rate"],
                "valid_evidence_span_rate": metrics["valid_evidence_span_rate"],
                "execution_error_episode_rate": metrics["execution_error_episode_rate"],
                "returned_tool_error_episode_count": metrics.get("tool_error_episode_count"),
                "runtime_error_episode_count": metrics.get("runtime_error_episode_count"),
                "context_error_episode_count": metrics.get("context_limit_error_episode_count"),
                "endpoint_error_episode_count": metrics.get("endpoint_error_episode_count"),
                "mean_duration_seconds": sum(timed) / len(timed) if timed else None,
                "duration_sample_count": len(timed),
                "duration_complete": len(timed) == len(benchmark["questions"]),
                **behavior_metrics.get(key, {}),
            },
        })
    by_pair = {(row["question_id"], row["model_key"]): row for row in results}
    cases = []
    for question in benchmark["questions"]:
        question_id = question["id"]
        systems = {}
        for key in model_keys:
            row = by_pair.get((question_id, key))
            if row is None:
                # Display-only placeholder. Never add it to the saved inference artifacts.
                row = {
                    "question_id": question_id, "model_key": key, "status": "not_attempted",
                    "error": None,
                    "predicted_answer": "", "predicted_citations": [], "predicted_evidence": [],
                    "trajectory": [], "steps": 0, "duration_seconds": 0,
                }
            trajectory = []
            for step in row["trajectory"]:
                observation, truncated = _excerpt(step.get("observation", ""))
                turn = {
                    "step": step.get("step"),
                    "action": step.get("action", ""),
                    "observation": observation,
                    "observation_truncated": truncated,
                    "observation_original_chars": len(str(step.get("observation", "") or "")),
                    "done": bool(step.get("done")),
                }
                reasoning = step.get("reasoning")
                if reasoning is not None:
                    if not isinstance(reasoning, str):
                        raise ValueError("saved generated reasoning must be text")
                    turn["reasoning"], turn["reasoning_truncated"] = _excerpt(reasoning)
                    turn["reasoning_original_chars"] = len(reasoning)
                if isinstance(step.get("logprob_diagnostics"), dict):
                    turn["logprob_diagnostics"] = step["logprob_diagnostics"]
                trajectory.append(turn)
            annotation = annotations.get((question_id, key), {})
            diagnostic = mechanical.get((question_id, key), {})
            semantic_diagnostics = components.get((question_id, key), unknown_diagnostics())
            if (question_id, key) in behavior:
                semantic_diagnostics = {
                    **semantic_diagnostics, "answer_behavior": behavior[(question_id, key)],
                }
            systems[key] = {
                "label": model_labels[key],
                "verdict": annotation.get("verdict"),
                "notes": annotation.get("notes", ""),
                "answer": row["predicted_answer"],
                "citations": row["predicted_citations"],
                "evidence": _strict_evidence(row["predicted_evidence"], documents),
                "steps": row["steps"],
                "duration_seconds": row["duration_seconds"] or None,
                "status": row["status"],
                "error": row.get("error"),
                "error_category": ", ".join(diagnostic.get("error_categories", [])),
                "returned_tool_error_count": diagnostic.get("tool_error_steps"),
                "runtime_error_count": diagnostic.get("runtime_error_steps"),
                "context_error_count": int(diagnostic.get("context_limit_error_episode", False)),
                "diagnostics": semantic_diagnostics,
                "mechanical_diagnostics": diagnostic,
                "policy_metadata": row.get("policy_metadata"),
                "trajectory": trajectory,
            }
        cases.append({
            "id": question_id,
            "question": question["question"],
            "answerability": question["expected_answerability"],
            "focus": "Saved paper-agent run; inspect the code, source spans, and review status.",
            "systems": systems,
        })
        if amended_continuation is not None:
            if question_id == amended_continuation["restarted_question_id"]:
                cases[-1]["episode_origin"] = "Q39 · restarted after original budget stop"
            elif question_id == amended_continuation["first_attempt_question_id"]:
                cases[-1]["episode_origin"] = "Q40 · first attempt in continuation"
            else:
                cases[-1]["episode_origin"] = "Original completed episode"
    if incomplete:
        score_status = "incomplete; saved traces only, no aggregate scores"
    elif scored and scored.get("human"):
        score_status = "human-reviewed"
    elif scored:
        score_status = "provisional model-graded; not human-reviewed"
    elif split == "test_candidate" and provisional:
        score_status = "provisional; automatic diagnostics only"
    else:
        score_status = "unreviewed; automatic diagnostics only"
    payload = {
        "kind": "envoybench_run",
        "title": f"QASPER Agent Studio · {split.replace('_', ' ').title()}",
        "subtitle": "Saved paper-agent evaluation; no inference in this viewer.",
        "supplementary": supplementary,
        "run_status": manifest["status"],
        "score_status": score_status,
        "comparison_mode": (
            "provisional" if provisional and split == "test_candidate" else "reviewed"
        ),
        "question_denominator": len(benchmark["questions"]),
        "models": models,
        "cases": cases,
        "analytical_baselines": [] if incomplete else [refusal_baseline(benchmark)],
        "featured_cases": featured,
        "review_status": (
            f"Separate component diagnoses cover {component_provenance['reviewed_question_count']} "
            f"of {len(benchmark['questions'])} selected cases. These are unblinded AI-assisted "
            "explanations, not an accuracy sample. Other component labels are unreviewed; "
            "the original end-to-end grades are unchanged."
        ),
        "automatic_diagnostics": automatic,
        "retrieval_baseline": prepared_retrieval_status(
            benchmark, manifest,
            HERE.parents[1] / "out/envoybench/retrieval-baseline-prepared-20260930/plan.json",
        ),
        "promotion": promotion,
        "provenance": {
            "run_id": manifest["run_id"],
            "benchmark_id": manifest["benchmark_id"],
            "results_hash": configuration_hash({"results": results}),
            "results_sha256": _file_sha256(results_path),
            "comparison_id": manifest["comparison_id"],
            "benchmark_hash": manifest["benchmark_hash"],
            "corpus_hash": manifest["corpus_hash"],
            "split_status": manifest["split_status"],
            "created_at_utc": manifest["created_at_utc"],
            "seed": manifest["seed"],
            "max_steps": manifest["max_steps"],
            "runner_hardware": manifest.get("runner_hardware"),
            "model_identities": manifest["models"],
            "review_kind": (scored or {}).get("review_provenance", {}).get("kind"),
            "judge": judge_provenance,
            "component_review": component_provenance,
            "behavior_review": behavior_provenance,
            "automatic_metrics_version": automatic["schema_version"],
            "source": f"Local EnvoyBench manifest.json and {results_path.name}",
        },
    }
    if amended_continuation is not None:
        payload["amended_continuation"] = amended_continuation
        payload["provenance"]["protocol_amendment"] = manifest["protocol_amendment"]
        payload["provenance"]["lineage_run_ids"] = {
            "original": amended_continuation["original_run_id"],
            "continuation": amended_continuation["continuation_run_id"],
        }
    if incomplete:
        interrupted = sum(row["status"] == "error" for row in results)
        payload["completion"] = {
            "planned": len(benchmark["questions"]), "recorded": len(results),
            "finished": len(results) - interrupted, "interrupted": interrupted,
            "not_attempted": len(benchmark["questions"]) - len(results),
        }
        payload["review_status"] = (
            "Incomplete attempt. Saved episodes retain their original status and traces. "
            "Unattempted questions are display placeholders, not generated results."
        )
    if incomplete or amended_continuation is not None:
        budget_path = run_dir / "usage-budget.json"
        if amended_continuation is not None and not budget_path.is_file():
            raise ValueError("Amended continuation needs its cumulative usage budget")
        if budget_path.is_file():
            budget = _read_json(budget_path)
            if amended_continuation is not None:
                original_usage = manifest["lineage"]["original"].get("usage")
                continuation_usage = manifest["lineage"]["continuation"].get("usage_delta")
                if (not isinstance(budget, dict)
                        or not isinstance(original_usage, dict)
                        or not isinstance(continuation_usage, dict)):
                    raise ValueError("Amended continuation has invalid cumulative usage")
                budget_config = budget.get("config")
                if (not isinstance(budget_config, dict)
                        or budget_config.get("max_estimated_usd")
                        != amended_continuation["amended_budget_cap_usd"]):
                    raise ValueError("Amended continuation budget cap differs from its manifest")
                for field in ("requests", "estimated_usd", "prompt_tokens", "completion_tokens"):
                    total = budget.get(field)
                    first, second = original_usage.get(field), continuation_usage.get(field)
                    if (type(total) not in {int, float} or type(first) not in {int, float}
                            or type(second) not in {int, float}
                            or not math.isclose(total, first + second, rel_tol=1e-9, abs_tol=1e-9)):
                        raise ValueError(f"Amended cumulative usage differs for {field}")
            payload["usage_budget"] = budget
            payload["provenance"]["usage_budget_sha256"] = _file_sha256(budget_path)
    if verified_trace_store is not None:
        # Keep the verified rows in memory. The lazy route must never re-read a
        # mutable results.json after the artifact and review checks above pass.
        verified_trace_store.update({
            ((manifest["run_id"], row["question_id"], row["model_key"]) if supplementary
             else (row["question_id"], row["model_key"])): {
                "question_id": row["question_id"],
                "model_key": row["model_key"],
                "status": row["status"],
                "error": row.get("error"),
                "steps": row["steps"],
                "duration_seconds": row["duration_seconds"],
                "trajectory": row["trajectory"],
            }
            for row in results
        })
    return payload


def _attach_qasper_score(path: Path, runs: list[dict]) -> None:
    """Attach a saved official metric only to its exact verified run artifacts."""
    if path.stat().st_size > 10 * 1024 * 1024:
        raise ValueError("QASPER score exceeds the 10 MiB display limit")
    report = _read_json(path)
    if (not isinstance(report, dict)
            or report.get("schema_version") != "qasper-agent-study-score-v1"):
        raise ValueError("Unsupported QASPER score schema")
    matches = [run for run in runs
               if run.get("provenance", {}).get("run_id") == report.get("run_id")]
    if len(matches) != 1:
        raise ValueError("QASPER score must identify exactly one loaded saved run")
    run = matches[0]
    if run.get("run_status") == "incomplete":
        raise ValueError("incomplete runs cannot display a full-run QASPER score")
    if "qasper_score" in run:
        raise ValueError("QASPER score already loaded for this run")
    for field in (
        "benchmark_id", "benchmark_hash", "corpus_hash", "results_hash", "results_sha256",
    ):
        if not report.get(field) or report[field] != run["provenance"].get(field):
            raise ValueError(f"QASPER score {field} differs from the saved run")
    ids = [case["id"] for case in run["cases"]]
    if (not isinstance(report.get("question_ids"), list)
            or sorted(report["question_ids"]) != sorted(ids)):
        raise ValueError("QASPER score question IDs differ from the saved run")
    summaries, rows = report.get("models"), report.get("rows")
    keys = {model["key"] for model in run["models"]}
    if not isinstance(summaries, dict) or set(summaries) != keys or not isinstance(rows, list):
        raise ValueError("QASPER score needs the saved run's model summaries and rows")

    def valid_f1(value: object) -> bool:
        return type(value) in {int, float} and math.isfinite(value) and 0 <= value <= 1

    by_pair = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Invalid QASPER score row")
        pair = (row.get("question_id"), row.get("model_key"))
        if (pair in by_pair or pair[0] not in ids or pair[1] not in keys
                or not valid_f1(row.get("answer_f1"))):
            raise ValueError("QASPER score has invalid or duplicate question/model scores")
        by_pair[pair] = row
    if set(by_pair) != {(question_id, key) for question_id in ids for key in keys}:
        raise ValueError("QASPER score is missing question/model scores")
    for model in run["models"]:
        summary = summaries[model["key"]]
        if (not isinstance(summary, dict) or not valid_f1(summary.get("answer_f1"))
                or summary.get("question_count") != len(ids)):
            raise ValueError("Invalid QASPER model score or denominator")
        average = sum(by_pair[(qid, model["key"])]["answer_f1"] for qid in ids) / len(ids)
        if not math.isclose(summary["answer_f1"], average, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("QASPER model score differs from its question scores")
        model["metrics"]["qasper_answer_f1"] = summary["answer_f1"]
    for case in run["cases"]:
        for key, system in case["systems"].items():
            row = by_pair[(case["id"], key)]
            missing = system["status"] != "submitted"
            expected_answer = None if missing else system["answer"]
            if (row.get("predicted_answer") != expected_answer
                    or row.get("status") != system["status"]
                    or row.get("prediction_missing") is not missing
                    or (missing and row["answer_f1"] != 0)):
                raise ValueError("QASPER scored prediction differs from the saved answer")
            system["qasper_answer_f1"] = row["answer_f1"]
    run["qasper_score"] = {
        "schema_version": report["schema_version"],
        "title": "Official QASPER Answer F1",
        "scope": (f"Official metric on {len(ids)} selected QASPER questions; "
                  "not the full benchmark or a support review."),
        "source_sha256": _file_sha256(path),
        "provenance": report.get("provenance", {}),
    }
    run["provenance"]["qasper_official_score"] = run["qasper_score"]


def build_demo_payload(
    *, run_dir: Path | None = None, review_dir: Path | None = None,
    dataset: Path = DEFAULT_DATASET,
    reference_review_path: Path | None = None,
    provisional: bool = False,
    judged_review_path: Path | None = None,
    verified_trace_store: dict[tuple[str, ...], dict] | None = None,
    token_diagnostic_run: Path | None = None,
    supplementary_runs: list[Path] | None = None,
    qasper_scores: list[Path] | None = None,
) -> dict:
    """Build a transparent dashboard payload without executing model actions."""
    if review_dir is not None and run_dir is None:
        raise ValueError("--review-dir requires --run-dir")
    if judged_review_path is not None and review_dir is None:
        raise ValueError("--judged-review requires --review-dir")
    if run_dir is not None:
        run_manifest = _read_json(run_dir / "manifest.json")
        if (isinstance(run_manifest, dict) and run_manifest.get("split") == "test_candidate"
                and not provisional):
            if reference_review_path is None or not reference_review_path.is_file():
                raise ValueError(
                    "test_candidate outputs stay closed until a completed reference "
                    "review sheet is supplied with --reference-review"
                )
            benchmark, _, _, _, _ = load_split(dataset, "test_candidate")
            _validated_references(benchmark, _read_json(reference_review_path))
            if review_dir is None:
                raise ValueError(
                    "test_candidate outputs stay closed until a completed blind "
                    "human answer review is supplied with --review-dir"
                )
    fixture = _read_json(HERE / "demo_fixture.json")
    if not isinstance(fixture, dict) or "active" not in fixture:
        raise ValueError("Bundled recorded-trace fixture is invalid")
    active = (
        _run_payload(
            run_dir, dataset, review_dir, reference_review_path,
            provisional=provisional, judged_review_path=judged_review_path,
            verified_trace_store=verified_trace_store,
        )
        if run_dir is not None else fixture["active"]
    )
    supplements = [
        _run_payload(path, dataset, None, None, provisional=True, supplementary=True,
                     verified_trace_store=verified_trace_store)
        for path in supplementary_runs or []
    ]
    run_ids = [run.get("provenance", {}).get("run_id") for run in [active, *supplements]]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("Loaded saved runs must have distinct run IDs")
    for path in qasper_scores or []:
        _attach_qasper_score(path, [active, *supplements])
    return {
        "schema_version": "envoybench-demo-v1",
        "candidate": _candidate_summary(dataset),
        "active": active,
        "supplementary_runs": supplements,
        "prior_art": PRIOR_ART,
        "token_diagnostic": (
            _token_diagnostic_payload(token_diagnostic_run, dataset)
            if token_diagnostic_run is not None else None
        ),
    }
