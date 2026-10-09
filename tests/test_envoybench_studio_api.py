"""The local Studio must support source review without exposing candidate outputs."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from benchmarks.envoybench.demo import build_demo_payload, create_app
from benchmarks.envoybench.run import _file_sha256, load_split
from benchmarks.envoybench.score import (
    prepare_provisional_bundle,
    prepare_review_bundle,
    reference_review_template,
    score_completed_review,
)
from src.eval.artifacts import configuration_hash, content_hash


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _candidate_dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "dataset"
    corpus = dataset / "test_candidate/corpus"
    _write(corpus / "paper_a.json", {
        "doc_id": "paper_a", "title": "A frozen paper",
        "text": "The answer is blue.",
    })
    corpus_hash = content_hash(corpus)
    question = {
        "id": "q1", "split": "pilot_evaluation",
        "question": "What is the answer?", "answer": "blue",
        "expected_answerability": "sufficient",
        "expected_citations": ["paper_a"],
        "required_doc_ids": ["paper_a"],
        "minimum_distinct_sources": 1,
        "grader_notes": [], "source_paper_id": "paper_a",
    }
    benchmark = {
        "schema_version": "research-benchmark-v1",
        "benchmark_id": "envoybench-test-fixture",
        "corpus_hash": corpus_hash,
        "reserved_doc_ids": ["paper_a"],
        "questions": [question],
    }
    benchmark_path = dataset / "test_candidate/benchmark.json"
    _write(benchmark_path, benchmark)
    benchmark_hash = _file_sha256(benchmark_path)
    _write(dataset / "test_candidate/manifest.json", {
        "schema_version": "research-snapshot-v1",
        "corpus_hash": corpus_hash,
        "split_status": "unreviewed_candidate_not_scored",
        "benchmark_sha256": benchmark_hash,
        "selected_question_ids": ["q1"],
        "selected_target_paper_ids": ["paper_a"],
        "papers": [{"doc_id": "paper_a"}],
    })
    splits_path = dataset / "splits.json"
    _write(splits_path, {
        "schema_version": "envoybench-splits-v1",
        "splits": {"test_candidate": {"question_ids": ["q1"]}},
    })
    _write(dataset / "manifest.json", {
        "schema_version": "envoybench-data-manifest-v1",
        "frozen_splits_sha256": _file_sha256(splits_path),
        "splits": {"test_candidate": {
            "benchmark": "test_candidate/benchmark.json",
            "corpus": "test_candidate/corpus",
            "manifest": "test_candidate/manifest.json",
            "benchmark_sha256": benchmark_hash,
            "corpus_hash": corpus_hash,
            "question_count": 1,
            "corpus_paper_count": 1,
            "status": "unreviewed_candidate_not_scored",
            "human_review_status": "pending",
        }},
    })
    return dataset


def test_studio_review_route_requires_human_decision_before_finalization(tmp_path: Path) -> None:
    dataset = _candidate_dataset(tmp_path)
    sheet_path = tmp_path / "review.json"
    client = TestClient(create_app(dataset=dataset, reference_review_path=sheet_path))

    initial = client.get("/api/reference-review")
    assert initial.status_code == 200
    assert initial.json()["decided_count"] == 0
    assert initial.json()["status"] == "incomplete"
    assert not sheet_path.exists()
    source = client.get("/api/reference-review/q1/source")
    assert source.status_code == 200
    assert source.json() == {
        "doc_id": "paper_a", "title": "A frozen paper", "text": "The answer is blue."
    }
    assert client.get("/api/reference-review/unknown/source").status_code == 404
    assert client.post("/api/reference-review/finalize", json={
        "reviewer_id": "human-1", "expected_sha256": initial.json()["revision"],
    }).status_code == 409

    saved = client.post("/api/reference-review/q1", json={
        "decision": "reference_valid", "reason": "Source states blue.",
        "reviewer_id": "human-1", "expected_sha256": initial.json()["revision"],
    })
    assert saved.status_code == 200
    assert saved.json()["decided_count"] == 1
    assert sheet_path.exists()
    assert client.post("/api/reference-review/q1", json={
        "decision": "reference_valid", "reason": "",
        "reviewer_id": "human-1", "expected_sha256": initial.json()["revision"],
    }).status_code == 409

    completed = client.post("/api/reference-review/finalize", json={
        "reviewer_id": "human-1", "expected_sha256": saved.json()["revision"],
    })
    assert completed.status_code == 200
    assert completed.json()["status"] == "complete"
    assert client.get("/api/demo").json()["candidate"]["reference_review_status"] == "complete"
    assert client.post("/api/reference-review/q1", json={
        "decision": "unscorable", "reason": "Too late", "reviewer_id": "human-1",
        "expected_sha256": completed.json()["revision"],
    }).status_code == 409

    run_dir = tmp_path / "run"
    _write(run_dir / "manifest.json", {"split": "test_candidate"})
    with pytest.raises(ValueError, match="completed blind human answer review"):
        build_demo_payload(
            run_dir=run_dir, dataset=dataset, reference_review_path=sheet_path
        )


def test_candidate_outputs_are_hidden_before_source_review(tmp_path: Path) -> None:
    dataset = _candidate_dataset(tmp_path)
    run_dir = tmp_path / "run"
    _write(run_dir / "manifest.json", {"split": "test_candidate"})
    with pytest.raises(ValueError, match="outputs stay closed"):
        build_demo_payload(run_dir=run_dir, dataset=dataset)


def test_studio_blind_routes_keep_model_identity_out_of_review(tmp_path: Path) -> None:
    dataset = _candidate_dataset(tmp_path)
    benchmark, _, _, corpus, _ = load_split(dataset, "test_candidate")
    source_sheet = reference_review_template(benchmark)
    source_sheet["status"] = "complete"
    source_sheet["reviewer_id"] = "source-reviewer"
    source_sheet["reviewed_at"] = "2026-09-30"
    source_sheet["questions"][0]["decision"] = "reference_valid"
    source_path = tmp_path / "source-review.json"
    _write(source_path, source_sheet)
    manifest = {
        "schema_version": "envoybench-run-v1", "status": "complete",
        "run_id": "run-one", "comparison_id": "pair-one",
        "benchmark_id": benchmark["benchmark_id"],
        "benchmark_hash": configuration_hash(benchmark),
        "corpus_hash": benchmark["corpus_hash"],
        "split": "test_candidate",
        "split_status": "unreviewed_candidate_not_scored",
        "full_split": True, "subset_smoke": False,
        "split_question_count": 1, "question_ids": ["q1"],
        "models": [{"key": "qwen_base"}, {"key": "qwen_v5"}],
        "created_at_utc": "2026-09-30T00:00:00+00:00",
        "seed": 42, "max_steps": 2,
    }
    results = []
    for model_key, answer in (
        ("qwen_base", "It is red."), ("qwen_v5", "It is blue.")
    ):
        results.append({
            "schema_version": "envoybench-run-v1",
            "run_id": manifest["run_id"],
            "comparison_id": manifest["comparison_id"],
            "benchmark_id": benchmark["benchmark_id"],
            "benchmark_hash": manifest["benchmark_hash"],
            "corpus_hash": benchmark["corpus_hash"],
            "split": "test_candidate", "split_status": manifest["split_status"],
            "question_id": "q1", "question": "What is the answer?",
            "model_key": model_key, "status": "submitted",
            "predicted_answer": answer,
            "predicted_citations": ["paper_a"],
            "predicted_evidence": [{"doc_id": "paper_a", "start": 14, "end": 18}],
            "trajectory": [], "steps": 0, "duration_seconds": 1.0,
        })
    blind_sheet, key, automatic = prepare_review_bundle(
        benchmark, manifest, results, corpus, source_sheet
    )
    blind_dir = tmp_path / "blind"
    _write(blind_dir / "review.json", blind_sheet)
    # The Studio's editing routes must work without the identity key on disk.
    client = TestClient(create_app(
        dataset=dataset, reference_review_path=source_path,
        blind_review_dir=blind_dir,
    ))
    initial = client.get("/api/blind-review")
    assert initial.status_code == 200
    assert initial.json()["decided_count"] == 0
    assert "qwen_base" not in initial.text and "qwen_v5" not in initial.text
    assert "assignments" not in initial.text
    blind_ids = [row["blind_id"] for row in initial.json()["rows"]]
    assert client.get(f"/api/blind-review/{blind_ids[0]}/source").json()["text"] == (
        "The answer is blue."
    )
    first = client.post(f"/api/blind-review/{blind_ids[0]}", json={
        "verdict": "pass", "notes": "", "relevant_passage": "",
        "reviewer_id": "answer-reviewer",
        "expected_sha256": initial.json()["revision"],
    })
    assert first.status_code == 200
    assert first.json()["decided_count"] == 1
    assert client.post("/api/blind-review/finalize", json={
        "reviewer_id": "answer-reviewer",
        "expected_sha256": first.json()["revision"],
    }).status_code == 409
    second = client.post(f"/api/blind-review/{blind_ids[1]}", json={
        "verdict": "fail", "notes": "The paper says blue.",
        "relevant_passage": "paper_a:14-18",
        "reviewer_id": "answer-reviewer",
        "expected_sha256": first.json()["revision"],
    })
    assert second.status_code == 200
    completed = client.post("/api/blind-review/finalize", json={
        "reviewer_id": "answer-reviewer",
        "expected_sha256": second.json()["revision"],
    })
    assert completed.status_code == 200
    assert completed.json()["reviewer_kind"] == "human"
    assert score_completed_review(completed.json(), key)["human"] is not None
    _write(blind_dir / "blind-key.json", key)
    _write(blind_dir / "automatic.json", automatic)
    run_dir = tmp_path / "run"
    _write(run_dir / "manifest.json", manifest)
    _write(run_dir / "results.json", results)
    named = build_demo_payload(
        run_dir=run_dir, review_dir=blind_dir,
        reference_review_path=source_path, dataset=dataset,
    )["active"]
    assert named["kind"] == "envoybench_run"
    assert named["promotion"]["status"] == "inconclusive"  # one synthetic question

    # Opting into a provisional comparison does not invent human source decisions.
    missing_source = tmp_path / "no-human-source-review.json"
    automatic_only = build_demo_payload(
        run_dir=run_dir, dataset=dataset,
        reference_review_path=missing_source, provisional=True,
    )["active"]
    assert automatic_only["comparison_mode"] == "provisional"
    assert automatic_only["promotion"] is None
    assert automatic_only["models"][0]["metrics"]["pass"] is None
    assert not missing_source.exists()

    provisional_review, provisional_key, provisional_automatic = (
        prepare_provisional_bundle(benchmark, manifest, results, corpus)
    )
    judged_review = copy.deepcopy(provisional_review)
    judged_review["status"] = "complete"
    judged_review["reviewer_id"] = "independent-model-judge"
    judged_review["reviewed_at"] = "2026-09-30"
    for row in judged_review["rows"]:
        row["verdict"] = "pass"
    provisional_dir = tmp_path / "provisional"
    _write(provisional_dir / "review.json", provisional_review)
    _write(provisional_dir / "blind-key.json", provisional_key)
    _write(provisional_dir / "automatic.json", provisional_automatic)
    judged_review_path = tmp_path / "judged/review.json"
    _write(judged_review_path, judged_review)
    judge_report = {
        "judged_review_hash": configuration_hash(judged_review),
        "benchmark_hash": configuration_hash(benchmark),
        "corpus_hash": benchmark["corpus_hash"],
        "results_hash": configuration_hash({"results": results}),
        "source_reference_status": "unreviewed_qasper",
        "status": "complete",
        "requested_model": "test-judge",
        "reported_models": ["test-judge-version"],
        "system_prompt_sha256": "test-prompt-hash",
        "input_tokens": 100,
        "output_tokens": 50,
        "limitations": "Model-assisted only",
    }
    _write(judged_review_path.parent / "judge-report.json", judge_report)
    with pytest.raises(ValueError, match="--judged-review requires --review-dir"):
        build_demo_payload(
            run_dir=run_dir, dataset=dataset, provisional=True,
            judged_review_path=judged_review_path,
        )
    provisional = build_demo_payload(
        run_dir=run_dir, review_dir=provisional_dir,
        reference_review_path=missing_source, dataset=dataset, provisional=True,
        judged_review_path=judged_review_path,
    )["active"]
    assert provisional["score_status"] == "provisional model-graded; not human-reviewed"
    assert provisional["provenance"]["review_kind"] == "model_assisted"
    assert provisional["provenance"]["judge"]["requested_model"] == "test-judge"
    assert provisional["provenance"]["judge"]["input_tokens"] == 100
    assert provisional["models"][0]["metrics"]["pass"] == 1
    assert provisional["models"][0]["metrics"]["answerable_pass"] == 1
    assert provisional["models"][0]["metrics"]["answerable_reviewed"] == 1
    assert provisional["models"][0]["metrics"]["unanswerable_reviewed"] == 0
    assert provisional["promotion"] is None
    assert not missing_source.exists()
    judge_report["judged_review_hash"] = "mismatched-review"
    _write(judged_review_path.parent / "judge-report.json", judge_report)
    with pytest.raises(ValueError, match="judge report does not match"):
        build_demo_payload(
            run_dir=run_dir, review_dir=provisional_dir,
            reference_review_path=missing_source, dataset=dataset, provisional=True,
            judged_review_path=judged_review_path,
        )
    with pytest.raises(ValueError, match="cannot share a blind human review session"):
        create_app(
            run_dir=run_dir, dataset=dataset, provisional=True,
            blind_review_dir=blind_dir,
        )
    provisional_client = TestClient(create_app(
        run_dir=run_dir, dataset=dataset, provisional=True,
        reference_review_path=missing_source,
    ))
    assert provisional_client.get("/api/reference-review").status_code == 200
    assert provisional_client.post("/api/reference-review/q1", json={
        "decision": "reference_valid", "reason": "Looks right", "reviewer_id": "x",
    }).status_code == 409
    assert not missing_source.exists()
