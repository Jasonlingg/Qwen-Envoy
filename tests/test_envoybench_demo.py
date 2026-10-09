"""The demo must keep recorded examples distinct from scored benchmark runs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from benchmarks.envoybench.demo import build_demo_payload, create_app
from benchmarks.envoybench.export import snapshot_html
from benchmarks.envoybench.run import _file_sha256
from src.eval.artifacts import configuration_hash, content_hash


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def test_default_demo_is_real_recorded_development_not_held_out_score() -> None:
    payload = build_demo_payload()
    assert payload["candidate"]["score"] is None
    assert payload["candidate"]["question_count"] == 40
    assert payload["candidate"]["reference_review_status"] == "pending"
    active = payload["active"]
    assert active["kind"].startswith("recorded_trace")
    assert "model-assisted" in active["score_status"].lower()
    assert active["question_denominator"] == 40
    assert [(model["key"], model["metrics"]["pass"]) for model in active["models"]] == [
        ("base", 9), ("v5", 19)
    ]
    assert len(active["cases"]) == 2
    assert all(case["systems"]["base"]["trajectory"] for case in active["cases"])
    assert all(case["systems"]["v5"]["trajectory"] for case in active["cases"])


def test_saved_unreviewed_run_shows_traces_without_semantic_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "data"
    corpus = dataset / "dev/corpus"
    _write(corpus / "paper_a.json", {
        "doc_id": "paper_a", "title": "A study", "text": "The answer is blue."
    })
    corpus_hash = content_hash(corpus)
    question = {
        "id": "q1", "split": "pilot_evaluation", "question": "What is the answer?",
        "answer": "blue", "expected_answerability": "sufficient",
        "expected_citations": ["paper_a"], "required_doc_ids": ["paper_a"],
        "minimum_distinct_sources": 1, "grader_notes": [], "source_paper_id": "paper_a",
    }
    benchmark = {
        "schema_version": "research-benchmark-v1", "benchmark_id": "envoybench-dev-fixture",
        "corpus_hash": corpus_hash, "reserved_doc_ids": ["paper_a"],
        "questions": [question],
    }
    benchmark_path = dataset / "dev/benchmark.json"
    _write(benchmark_path, benchmark)
    benchmark_file_hash = _file_sha256(benchmark_path)
    _write(dataset / "dev/manifest.json", {
        "schema_version": "research-snapshot-v1", "corpus_hash": corpus_hash,
        "split_status": "previously_used_development", "benchmark_sha256": benchmark_file_hash,
        "selected_question_ids": ["q1"], "selected_target_paper_ids": ["paper_a"],
        "papers": [{"doc_id": "paper_a"}],
    })
    splits_path = dataset / "splits.json"
    _write(splits_path, {"schema_version": "envoybench-splits-v1",
                         "splits": {"dev": {"question_ids": ["q1"]}}})
    _write(dataset / "manifest.json", {
        "schema_version": "envoybench-data-manifest-v1",
        "frozen_splits_sha256": _file_sha256(splits_path),
        "splits": {
            "dev": {"benchmark": "dev/benchmark.json", "corpus": "dev/corpus",
                    "manifest": "dev/manifest.json", "benchmark_sha256": benchmark_file_hash,
                    "corpus_hash": corpus_hash, "question_count": 1,
                    "status": "previously_used_development"},
            "test_candidate": {"question_count": 1, "corpus_paper_count": 1,
                               "corpus_hash": corpus_hash,
                               "status": "unreviewed_candidate_not_scored",
                               "human_review_status": "pending"},
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
    long_observation = "paper_a " + "x" * 2000 + '</script><script>alert("tail")</script>'
    long_reasoning = "Inspect paper_a. " + "r" * 2000 + " Complete reasoning tail."
    token_diagnostic = {
        "schema_version": "sampled-content-logprobs-v1",
        "reported_token_count": 70, "valid_logprob_count": 69,
        "sum_logprob": -2.0, "mean_logprob": -2.0 / 69,
        "prefix": [{"token": "</script><img src=x>", "logprob": -0.2}],
        "prefix_truncated": True,
        "scope": "provider generated content; not aligned to cleaned action or answer correctness",
    }
    for key in ("base", "trained"):
        rows.append({
            "schema_version": "envoybench-run-v1", "run_id": manifest["run_id"],
            "comparison_id": manifest["comparison_id"],
            "benchmark_id": benchmark["benchmark_id"],
            "benchmark_hash": manifest["benchmark_hash"], "corpus_hash": corpus_hash,
            "split": "dev", "split_status": manifest["split_status"],
            "question_id": "q1", "question": question["question"], "model_key": key,
            "status": "submitted", "predicted_answer": "blue",
            "predicted_citations": ["paper_a"],
            "predicted_evidence": [{"doc_id": "paper_a", "start": 14, "end": 18}],
            "trajectory": [{"step": 1, "action": "print(search('blue'))",
                            "observation": long_observation, "reward": 0.25,
                            "output": long_observation, "done": False,
                            "reasoning": long_reasoning,
                            "logprob_diagnostics": token_diagnostic},
                           {"step": 2, "action": "SUBMIT: blue",
                            "observation": "Submitted", "done": True,
                            "logprob_diagnostics": token_diagnostic}],
            "steps": 2, "duration_seconds": 1.0,
        })
    _write(run_dir / "results.json", rows)
    payload = build_demo_payload(run_dir=run_dir, dataset=dataset)
    active = payload["active"]
    assert active["kind"] == "envoybench_run"
    assert active["question_denominator"] == 1
    assert "unreviewed" in active["score_status"]
    assert all(model["metrics"]["pass"] is None for model in active["models"])
    assert active["cases"][0]["systems"]["base"]["evidence"][0]["valid"] is True
    summary_step = active["cases"][0]["systems"]["base"]["trajectory"][0]
    assert summary_step["observation_truncated"] is True
    assert len(summary_step["observation"]) == 1800
    assert summary_step["reasoning_truncated"] is True
    assert summary_step["reasoning_original_chars"] == len(long_reasoning)
    assert len(summary_step["reasoning"]) == 1800
    assert summary_step["logprob_diagnostics"] == token_diagnostic
    assert payload["token_diagnostic"] is None

    # The standalone CLI must export full verified turns, not the live summary's excerpts.
    from benchmarks.envoybench.export import main as export_main

    snapshot_path = tmp_path / "full-studio.html"
    monkeypatch.setattr("sys.argv", ["envoybench-export", "--run-dir", str(run_dir),
                                    "--dataset", str(dataset), "--output", str(snapshot_path)])
    assert export_main() == 0
    snapshot = snapshot_path.read_text()
    encoded = snapshot.split('<script id="envoybench-data" type="application/json">', 1)[1]
    encoded = encoded.split("</script>", 1)[0]
    assert "<" not in encoded
    exported = json.loads(encoded)
    assert exported["active"]["trace_scope"] == "full_saved_trajectories"
    exported_step = exported["active"]["cases"][0]["systems"]["base"]["trajectory"][0]
    assert exported_step["observation"] == long_observation
    assert exported_step["reasoning"] == long_reasoning
    assert exported_step["logprob_diagnostics"] == token_diagnostic
    assert "observation_truncated" not in exported_step
    assert "reasoning_truncated" not in exported_step
    assert "output" not in exported_step and "reward" not in exported_step
    assert exported["active"]["models"] == active["models"]
    assert len(summary_step["observation"]) == 1800
    with pytest.raises(ValueError, match="every case's verified saved trajectory"):
        snapshot_html(payload, verified_trace_store={})

    client = TestClient(create_app(run_dir=run_dir, dataset=dataset))
    full = client.get("/api/demo/trace", params={"question_id": "q1", "model": "base"})
    assert full.status_code == 200
    assert full.json() == {
        "question_id": "q1", "model_key": "base", "status": "submitted",
        "error": None, "steps": 2, "duration_seconds": 1.0,
        "trajectory": rows[0]["trajectory"],
    }
    assert full.json()["trajectory"][0]["observation"] == long_observation
    assert full.json()["trajectory"][0]["reasoning"] == long_reasoning
    assert full.json()["trajectory"][0]["reward"] == 0.25
    assert client.get("/api/demo/trace", params={
        "question_id": "missing", "model": "base",
    }).status_code == 404
    assert client.get("/api/demo/trace", params={
        "question_id": "q1", "model": "missing",
    }).status_code == 404
    # A blind review session and the bundled fixture never expose named full traces.
    blind = TestClient(create_app(
        run_dir=run_dir, dataset=dataset, blind_review_dir=tmp_path / "blind",
    ))
    assert blind.get("/api/demo/trace", params={
        "question_id": "q1", "model": "base",
    }).status_code == 404
    fixture = TestClient(create_app(dataset=dataset))
    assert fixture.get("/api/demo/trace", params={
        "question_id": "q1", "model": "base",
    }).status_code == 404

    # A separately selected paired smoke must not replace or regrade the main run.
    smoke_dir = tmp_path / "smoke"
    smoke_manifest = {**manifest, "full_split": False, "subset_smoke": True}
    _write(smoke_dir / "manifest.json", smoke_manifest)
    _write(smoke_dir / "results.json", rows)
    with_diagnostic = build_demo_payload(
        run_dir=run_dir, dataset=dataset, token_diagnostic_run=smoke_dir,
    )
    assert with_diagnostic["active"] == payload["active"]
    diagnostic = with_diagnostic["token_diagnostic"]
    assert diagnostic["status"] == "complete_paired_smoke"
    assert diagnostic["question_count"] == 1
    assert diagnostic["cases"][0]["systems"]["base"]["trajectory"] == rows[0]["trajectory"]
    assert diagnostic["source_sha256"]["results.json"] == _file_sha256(smoke_dir / "results.json")
    assert all("metrics" not in model for model in diagnostic["models"])
    rendered = snapshot_html(with_diagnostic)
    assert "</script><img src=x>" not in rendered
    assert "\\u003c/script>\\u003cimg src=x>" in rendered
    assert "Alternative-token probabilities were not recorded." in rendered
    assert "not aligned to the cleaned Python action" in rendered
    diagnostic_client = TestClient(create_app(
        run_dir=run_dir, dataset=dataset, token_diagnostic_run=smoke_dir,
    ))
    assert diagnostic_client.get("/api/demo").json()["token_diagnostic"] == diagnostic
    with pytest.raises(ValueError, match="blind human review"):
        create_app(dataset=dataset, token_diagnostic_run=smoke_dir,
                   blind_review_dir=tmp_path / "blind")

    # Complete manifest rows alone are insufficient: every pair must submit.
    _write(smoke_dir / "results.json", [{**rows[0], "status": "error"}, rows[1]])
    with pytest.raises(ValueError, match="submissions from every paired model"):
        build_demo_payload(dataset=dataset, token_diagnostic_run=smoke_dir)
    _write(smoke_dir / "results.json", rows[:1])
    with pytest.raises(ValueError, match="missing paired submissions"):
        build_demo_payload(dataset=dataset, token_diagnostic_run=smoke_dir)
    _write(smoke_dir / "results.json", [{**rows[0], "benchmark_hash": "wrong"}, rows[1]])
    with pytest.raises(ValueError, match="benchmark_hash differs"):
        build_demo_payload(dataset=dataset, token_diagnostic_run=smoke_dir)

    # The endpoint serves only the row verified when the app was created.
    _write(run_dir / "results.json", [{**rows[0], "trajectory": []}, rows[1]])
    assert client.get("/api/demo/trace", params={
        "question_id": "q1", "model": "base",
    }).json()["trajectory"][0]["observation"] == long_observation
