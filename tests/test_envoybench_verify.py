"""Saved-run verification must catch altered inputs and review bindings."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.envoybench.run import _file_sha256
from benchmarks.envoybench.score import prepare_provisional_bundle
from benchmarks.envoybench.verify import main, verify_run
from src.eval.artifacts import configuration_hash, content_hash


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    dataset = tmp_path / "data"
    corpus = dataset / "dev/corpus"
    _write(corpus / "paper.json", {
        "doc_id": "paper", "title": "A study", "text": "The answer is blue."
    })
    corpus_hash = content_hash(corpus)
    question = {
        "id": "q1", "split": "pilot_evaluation", "question": "What is the answer?",
        "answer": "blue", "expected_answerability": "sufficient",
        "expected_citations": ["paper"], "required_doc_ids": ["paper"],
        "minimum_distinct_sources": 1, "grader_notes": [], "source_paper_id": "paper",
    }
    benchmark = {
        "schema_version": "research-benchmark-v1", "benchmark_id": "verify-fixture",
        "corpus_hash": corpus_hash, "reserved_doc_ids": ["paper"],
        "questions": [question],
    }
    benchmark_path = dataset / "dev/benchmark.json"
    _write(benchmark_path, benchmark)
    benchmark_file_hash = _file_sha256(benchmark_path)
    _write(dataset / "dev/manifest.json", {
        "schema_version": "research-snapshot-v1", "corpus_hash": corpus_hash,
        "split_status": "previously_used_development", "benchmark_sha256": benchmark_file_hash,
        "selected_question_ids": ["q1"], "selected_target_paper_ids": ["paper"],
        "papers": [{"doc_id": "paper"}],
    })
    splits_path = dataset / "splits.json"
    _write(splits_path, {
        "schema_version": "envoybench-splits-v1", "splits": {
            "dev": {"question_ids": ["q1"]}
        },
    })
    _write(dataset / "manifest.json", {
        "schema_version": "envoybench-data-manifest-v1",
        "frozen_splits_sha256": _file_sha256(splits_path),
        "splits": {
            "dev": {
                "benchmark": "dev/benchmark.json", "corpus": "dev/corpus",
                "manifest": "dev/manifest.json", "benchmark_sha256": benchmark_file_hash,
                "corpus_hash": corpus_hash, "question_count": 1,
                "status": "previously_used_development",
            },
            "test_candidate": {
                "question_count": 1, "corpus_paper_count": 1,
                "corpus_hash": corpus_hash,
                "status": "unreviewed_candidate_not_scored",
                "human_review_status": "pending",
            },
        },
    })

    run_dir = tmp_path / "run"
    manifest = {
        "schema_version": "envoybench-run-v1", "status": "complete",
        "run_id": "fixture-run", "comparison_id": "fixture-comparison",
        "benchmark_id": benchmark["benchmark_id"],
        "benchmark_hash": configuration_hash(benchmark), "corpus_hash": corpus_hash,
        "split": "dev", "split_status": "previously_used_development",
        "full_split": True, "subset_smoke": False, "split_question_count": 1,
        "question_ids": ["q1"], "models": [{"key": "base"}, {"key": "trained"}],
        "created_at_utc": "2026-09-30T00:00:00+00:00", "seed": 42, "max_steps": 2,
    }
    _write(run_dir / "manifest.json", manifest)
    rows = []
    for model_key in ("base", "trained"):
        rows.append({
            "schema_version": "envoybench-run-v1", "run_id": manifest["run_id"],
            "comparison_id": manifest["comparison_id"],
            "benchmark_id": benchmark["benchmark_id"],
            "benchmark_hash": manifest["benchmark_hash"], "corpus_hash": corpus_hash,
            "split": "dev", "split_status": manifest["split_status"],
            "question_id": "q1", "question": question["question"],
            "model_key": model_key, "status": "submitted", "predicted_answer": "blue",
            "predicted_citations": ["paper"],
            "predicted_evidence": [{"doc_id": "paper", "start": 14, "end": 18}],
            "trajectory": [{"step": 1, "action": "print(search('blue'))",
                            "observation": "paper", "done": False},
                           {"step": 2, "action": "SUBMIT: blue",
                            "observation": "Submitted", "done": True}],
            "steps": 2, "duration_seconds": 1.0,
        })
    _write(run_dir / "results.json", rows)
    review, key, _ = prepare_provisional_bundle(benchmark, manifest, rows, corpus)
    review["status"] = "complete"
    review["reviewer_id"] = "test-model"
    review["reviewed_at"] = "2026-09-30"
    for row in review["rows"]:
        row["verdict"] = "pass"
    review_dir = tmp_path / "review"
    _write(review_dir / "review.json", review)
    _write(review_dir / "blind-key.json", key)
    return dataset, run_dir, review_dir


def test_verifier_reports_separate_pass_counts_and_exact_spans(tmp_path: Path) -> None:
    dataset, run_dir, review_dir = _fixture(tmp_path)
    report = verify_run(run_dir=run_dir, dataset=dataset, review_dir=review_dir)
    assert report["status"] == "verified"
    assert report["checks"]["source_spans_exact"] == 2
    assert report["checks"]["source_spans_invalid"] == 0
    assert report["checks"]["results_bound_to_review"] is True
    assert report["models"]["base"]["answerable_pass"] == 1
    assert report["models"]["base"]["unanswerable_pass"] == 0
    assert report["review"]["human_reviewed"] is False


def test_verifier_exits_nonzero_if_result_changes_after_review(
    tmp_path: Path, capsys,
) -> None:
    dataset, run_dir, review_dir = _fixture(tmp_path)
    path = run_dir / "results.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    rows[0]["predicted_answer"] = "red"
    _write(path, rows)
    assert main([
        "--dataset", str(dataset), "--run-dir", str(run_dir),
        "--review-dir", str(review_dir),
    ]) == 1
    assert "results" in capsys.readouterr().err.lower()


def test_verifier_exits_nonzero_if_frozen_source_changes(
    tmp_path: Path, capsys,
) -> None:
    dataset, run_dir, review_dir = _fixture(tmp_path)
    _write(dataset / "dev/corpus/paper.json", {
        "doc_id": "paper", "title": "A study", "text": "The answer is red."
    })
    assert main([
        "--dataset", str(dataset), "--run-dir", str(run_dir),
        "--review-dir", str(review_dir),
    ]) == 1
    assert "corpus" in capsys.readouterr().err.lower()


def test_verifier_rejects_duplicate_model_question_pair(tmp_path: Path, capsys) -> None:
    dataset, run_dir, _ = _fixture(tmp_path)
    path = run_dir / "results.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    _write(path, [*rows, rows[0]])
    assert main(["--dataset", str(dataset), "--run-dir", str(run_dir)]) == 1
    assert "duplicate" in capsys.readouterr().err.lower()
