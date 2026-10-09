"""Local EnvoyBench Studio: source review and inspectable saved traces.

By default this shows a *historical development* Qwen comparison, not an
EnvoyBench test result. Pass --run-dir after a complete EnvoyBench evaluation to
inspect its actual saved trajectories without running any model calls.
The paper reader stores uploads and bounded live investigations separately from
benchmark results. Live inference requires an explicit operator model config.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import uvicorn
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

from benchmarks.envoybench.answer_review import (
    finalize_answer_review,
    load_answer_review,
    save_answer_verdict,
)
from benchmarks.envoybench.diagnostics import (
    case_diagnostics,
    known_source_id,
    prepared_retrieval_status,
    refusal_baseline,
    response_behavior,
    unknown_diagnostics,
)
from benchmarks.envoybench.paper_api import create_paper_router
from benchmarks.envoybench.promotion import evaluate_promotion
from benchmarks.envoybench.reference_review import (
    finalize_reference_review,
    load_reference_review,
    save_reference_decision,
)
from benchmarks.envoybench.run import DEFAULT_DATASET, _file_sha256, load_split
from benchmarks.envoybench.score import (
    _automatic_score,
    _strict_evidence,
    _validated_references,
    _verify_artifacts,
    score_completed_review,
)
from src.eval.artifacts import configuration_hash
from src.eval.research_review import load_corpus

HERE = Path(__file__).resolve().parent
DEFAULT_REFERENCE_REVIEW = HERE.parents[1] / "out/envoybench/test-reference-review.json"
DEFAULT_PAPER_STATE = HERE.parents[1] / "out/envoybench/paper-workspace"
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
    verified_trace_store: dict[tuple[str, str], dict] | None = None,
) -> dict:
    manifest = _read_json(run_dir / "manifest.json")
    results = _read_json(run_dir / "results.json")
    if not isinstance(manifest, dict) or not isinstance(results, list):
        raise ValueError("Run manifest/results have invalid JSON shape")
    split = manifest.get("split")
    if split not in {"dev", "test_candidate"}:
        raise ValueError("Run split must be dev or test_candidate")
    benchmark, _, _, corpus_path, _ = load_split(dataset, split)
    documents, model_keys = _verify_artifacts(benchmark, manifest, results, corpus_path)
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
            "label": key.replace("_", " ").title(),
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
            row = by_pair[(question_id, key)]
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
            diagnostic = mechanical[(question_id, key)]
            semantic_diagnostics = components.get((question_id, key), unknown_diagnostics())
            if (question_id, key) in behavior:
                semantic_diagnostics = {
                    **semantic_diagnostics, "answer_behavior": behavior[(question_id, key)],
                }
            systems[key] = {
                "label": key.replace("_", " ").title(),
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
            "focus": "Complete paired run; inspect the code, source spans, and verdicts.",
            "systems": systems,
        })
    if scored and scored.get("human"):
        score_status = "human-reviewed"
    elif scored:
        score_status = "provisional model-graded; not human-reviewed"
    elif split == "test_candidate" and provisional:
        score_status = "provisional; automatic diagnostics only"
    else:
        score_status = "unreviewed; automatic diagnostics only"
    payload = {
        "kind": "envoybench_run",
        "title": f"EnvoyBench · {split.replace('_', ' ').title()}",
        "subtitle": "Complete paired run from local saved artifacts; no inference in this viewer.",
        "score_status": score_status,
        "comparison_mode": (
            "provisional" if provisional and split == "test_candidate" else "reviewed"
        ),
        "question_denominator": len(benchmark["questions"]),
        "models": models,
        "cases": cases,
        "analytical_baselines": [refusal_baseline(benchmark)],
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
            "source": "Local EnvoyBench manifest.json and results.json",
        },
    }
    if verified_trace_store is not None:
        # Keep the verified rows in memory. The lazy route must never re-read a
        # mutable results.json after the artifact and review checks above pass.
        verified_trace_store.update({
            (row["question_id"], row["model_key"]): {
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


def build_demo_payload(
    *, run_dir: Path | None = None, review_dir: Path | None = None,
    dataset: Path = DEFAULT_DATASET,
    reference_review_path: Path | None = None,
    provisional: bool = False,
    judged_review_path: Path | None = None,
    verified_trace_store: dict[tuple[str, str], dict] | None = None,
    token_diagnostic_run: Path | None = None,
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
    return {
        "schema_version": "envoybench-demo-v1",
        "candidate": _candidate_summary(dataset),
        "active": active,
        "prior_art": PRIOR_ART,
        "token_diagnostic": (
            _token_diagnostic_payload(token_diagnostic_run, dataset)
            if token_diagnostic_run is not None else None
        ),
    }


def create_app(
    *, run_dir: Path | None = None, review_dir: Path | None = None,
    dataset: Path = DEFAULT_DATASET,
    reference_review_path: Path = DEFAULT_REFERENCE_REVIEW,
    blind_review_dir: Path | None = None,
    provisional: bool = False,
    judged_review_path: Path | None = None,
    paper_state_dir: Path = DEFAULT_PAPER_STATE,
    paper_models_path: Path | None = None,
    web_evidence_report_path: Path | None = None,
    token_diagnostic_run: Path | None = None,
) -> FastAPI:
    if provisional and blind_review_dir is not None:
        raise ValueError("provisional named outputs cannot share a blind human review session")
    if token_diagnostic_run is not None and blind_review_dir is not None:
        raise ValueError("named token diagnostics cannot share a blind human review session")
    app = FastAPI(title="EnvoyBench Studio")
    verified_trace_store: dict[tuple[str, str], dict] = {}
    app.state.payload = build_demo_payload(
        run_dir=run_dir, review_dir=review_dir, dataset=dataset,
        reference_review_path=reference_review_path, provisional=provisional,
        judged_review_path=judged_review_path,
        verified_trace_store=(verified_trace_store if blind_review_dir is None else None),
        token_diagnostic_run=token_diagnostic_run,
    )
    app.state.verified_trace_store = verified_trace_store
    app.state.web_evidence_report = (
        _load_web_evidence_report(web_evidence_report_path)
        if blind_review_dir is None and web_evidence_report_path is not None else None
    )
    # Named checkpoints and live runs must not leak into a blind review session.
    if blind_review_dir is None:
        app.include_router(create_paper_router(paper_state_dir, paper_models_path))

        @app.get("/papers", response_class=HTMLResponse)
        def paper_reader() -> HTMLResponse:
            return HTMLResponse((HERE / "paper.html").read_text(encoding="utf-8"))

        @app.get("/model", response_class=HTMLResponse)
        def model_page() -> HTMLResponse:
            return HTMLResponse((HERE / "model.html").read_text(encoding="utf-8"))

        @app.get("/api/model-card", response_class=JSONResponse)
        def model_card() -> JSONResponse:
            return JSONResponse(_read_json(HERE / "model_card.json"))

        @app.get("/web-evidence", response_class=HTMLResponse)
        def web_evidence_page() -> HTMLResponse:
            return HTMLResponse((HERE / "web_evidence.html").read_text(encoding="utf-8"))

        @app.get("/api/web-evidence", response_class=JSONResponse)
        def web_evidence_data() -> JSONResponse:
            if app.state.web_evidence_report is None:
                raise HTTPException(
                    status_code=404, detail="No completed web-evidence report loaded"
                )
            return JSONResponse(app.state.web_evidence_report)

        @app.get("/api/demo/trace", response_class=JSONResponse)
        def full_saved_trace(question_id: str, model: str) -> JSONResponse:
            """One verified saved Python-turn trajectory; no inference or file lookup."""
            active = app.state.payload["active"]
            if run_dir is None or active.get("kind") != "envoybench_run":
                raise HTTPException(status_code=404, detail="No saved run loaded")
            case = next(
                (item for item in active["cases"] if item["id"] == question_id), None
            )
            if case is None or model not in case["systems"]:
                raise HTTPException(status_code=404, detail="Unknown question or model")
            result = app.state.verified_trace_store.get((question_id, model))
            if result is None:
                raise HTTPException(status_code=404, detail="No verified trace for selection")
            return JSONResponse(result)

    def source_context() -> tuple[dict, Path]:
        try:
            benchmark, _, _, corpus_dir, _ = load_split(dataset, "test_candidate")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return benchmark, corpus_dir

    def review_response(result: dict) -> dict:
        sheet = result["review"]
        app.state.payload["candidate"]["reference_review_status"] = (
            "complete" if sheet["status"] == "complete" else "pending"
        )
        return {
            **sheet,
            "revision": result["sha256"],
            "decided_count": result["decided_count"],
            "total_count": result["total_count"],
        }

    def require_blind_human_session() -> None:
        if provisional and run_dir is not None:
            raise HTTPException(
                status_code=409,
                detail=(
                    "human review cannot be recorded while provisional named outputs are visible"
                ),
            )

    def blind_context() -> tuple[dict, Path, Path]:
        if blind_review_dir is None:
            raise HTTPException(status_code=404, detail="no prepared blind review bundle selected")
        benchmark, corpus_dir = source_context()
        try:
            source_sheet = load_reference_review(
                reference_review_path, benchmark, corpus_dir
            )["review"]
            _validated_references(benchmark, source_sheet)
            review_path = blind_review_dir / "review.json"
            blind_sheet = load_answer_review(review_path, benchmark, corpus_dir)
            if blind_sheet["reference_review_hash"] != configuration_hash(source_sheet):
                raise ValueError("blind review differs from finalized source review")
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return benchmark, corpus_dir, review_path

    def frozen_paper(question_id: str, benchmark: dict, corpus_dir: Path) -> dict:
        question = next(
            (item for item in benchmark["questions"] if item["id"] == question_id), None
        )
        if question is None:
            raise HTTPException(status_code=404, detail="unknown benchmark question_id")
        documents = load_corpus(corpus_dir)
        try:
            document = documents[known_source_id(question, documents)]
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {
            "doc_id": document["doc_id"],
            "title": document.get("title", ""),
            "text": document["text"],
        }

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse((HERE / "demo.html").read_text(encoding="utf-8"))

    @app.get("/api/demo", response_class=JSONResponse)
    def demo_data() -> JSONResponse:
        return JSONResponse(app.state.payload)

    @app.get("/api/reference-review", response_class=JSONResponse)
    def reference_review_data() -> JSONResponse:
        benchmark, corpus_dir = source_context()
        try:
            result = load_reference_review(reference_review_path, benchmark, corpus_dir)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return JSONResponse(review_response(result))

    @app.get("/api/reference-review/{question_id}/source", response_class=JSONResponse)
    def reference_source(question_id: str) -> JSONResponse:
        benchmark, corpus_dir = source_context()
        return JSONResponse(frozen_paper(question_id, benchmark, corpus_dir))

    @app.post("/api/reference-review/finalize", response_class=JSONResponse)
    def finalize_reference_data(payload: dict = Body(...)) -> JSONResponse:
        require_blind_human_session()
        benchmark, corpus_dir = source_context()
        try:
            result = finalize_reference_review(
                reference_review_path, benchmark, corpus_dir,
                payload.get("reviewer_id"),
                expected_sha256=payload.get("expected_sha256"),
            )
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return JSONResponse(review_response(result))

    @app.post("/api/reference-review/{question_id}", response_class=JSONResponse)
    def save_reference_data(question_id: str, payload: dict = Body(...)) -> JSONResponse:
        require_blind_human_session()
        benchmark, corpus_dir = source_context()
        try:
            result = save_reference_decision(
                reference_review_path, benchmark, corpus_dir,
                question_id, payload.get("decision"), payload.get("reason"),
                payload.get("reviewer_id"),
                expected_sha256=payload.get("expected_sha256"),
            )
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return JSONResponse(review_response(result))

    @app.get("/api/blind-review", response_class=JSONResponse)
    def blind_review_data() -> JSONResponse:
        benchmark, corpus_dir, review_path = blind_context()
        try:
            result = load_answer_review(review_path, benchmark, corpus_dir)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return JSONResponse(result)

    @app.get("/api/blind-review/{blind_id}/source", response_class=JSONResponse)
    def blind_review_source(blind_id: str) -> JSONResponse:
        benchmark, corpus_dir, review_path = blind_context()
        review = load_answer_review(review_path, benchmark, corpus_dir)
        row = next((item for item in review["rows"] if item["blind_id"] == blind_id), None)
        if row is None:
            raise HTTPException(status_code=404, detail="unknown blind_id")
        return JSONResponse(frozen_paper(row["question_id"], benchmark, corpus_dir))

    @app.post("/api/blind-review/finalize", response_class=JSONResponse)
    def finalize_blind_review(payload: dict = Body(...)) -> JSONResponse:
        require_blind_human_session()
        benchmark, corpus_dir, review_path = blind_context()
        try:
            result = finalize_answer_review(
                review_path, benchmark, corpus_dir, payload.get("reviewer_id"),
                expected_sha256=payload.get("expected_sha256"),
            )
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return JSONResponse(result)

    @app.post("/api/blind-review/{blind_id}", response_class=JSONResponse)
    def save_blind_review(blind_id: str, payload: dict = Body(...)) -> JSONResponse:
        require_blind_human_session()
        benchmark, corpus_dir, review_path = blind_context()
        try:
            result = save_answer_verdict(
                review_path, benchmark, corpus_dir, blind_id,
                payload.get("verdict"), payload.get("notes"),
                payload.get("relevant_passage"), payload.get("reviewer_id"),
                expected_sha256=payload.get("expected_sha256"),
            )
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return JSONResponse(result)

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--run-dir", type=Path, help="complete local EnvoyBench run directory")
    parser.add_argument("--review-dir", type=Path, help="completed blind review bundle")
    parser.add_argument(
        "--judged-review", type=Path,
        help="completed model-graded review JSON; key remains in --review-dir",
    )
    parser.add_argument(
        "--provisional", action="store_true",
        help="show a test_candidate run without human review; never a promotion result",
    )
    parser.add_argument(
        "--blind-review-dir", type=Path,
        help="prepared anonymous answer-review bundle to edit before scoring",
    )
    parser.add_argument(
        "--reference-review", type=Path, default=DEFAULT_REFERENCE_REVIEW,
        help="local source-review sheet; created on the first saved decision",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--paper-state-dir", type=Path, default=DEFAULT_PAPER_STATE,
        help="local paper snapshots and saved investigations (separate from the benchmark)",
    )
    parser.add_argument(
        "--paper-models", type=Path,
        help="operator model JSON for live paper runs; omitted means reader and saved runs only",
    )
    parser.add_argument(
        "--web-evidence-report", type=Path,
        help="completed local web-to-evidence pilot report JSON (read-only)",
    )
    parser.add_argument(
        "--token-diagnostic-run", type=Path,
        help="complete paired dev smoke directory; separate from benchmark quality scores",
    )
    args = parser.parse_args()
    app = create_app(
        run_dir=args.run_dir, review_dir=args.review_dir, dataset=args.dataset,
        reference_review_path=args.reference_review,
        blind_review_dir=args.blind_review_dir,
        provisional=args.provisional,
        judged_review_path=args.judged_review,
        paper_state_dir=args.paper_state_dir,
        paper_models_path=args.paper_models,
        web_evidence_report_path=args.web_evidence_report,
        token_diagnostic_run=args.token_diagnostic_run,
    )
    print(f"EnvoyBench Studio: http://127.0.0.1:{args.port}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
