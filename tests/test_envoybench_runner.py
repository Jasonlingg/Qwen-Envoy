"""Offline checks for the frozen benchmark runner and its sandbox boundary."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from benchmarks.envoybench import run as runner
from src.env.document_env import StepRecord
from src.eval.artifacts import content_hash
from src.eval.harness import EvalResult


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


@pytest.fixture
def frozen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    dataset = tmp_path / "data"
    corpus = dataset / "dev/corpus"
    _write(corpus / "paper_a.json", {
        "doc_id": "paper_a", "title": "A study", "text": "Blue is the reported result.",
    })
    corpus_hash = content_hash(corpus)
    questions = [{
        "id": "q1", "split": "pilot_evaluation", "question": "What was reported?",
        "answer": "Blue", "expected_citations": ["paper_a"],
        "expected_answerability": "sufficient", "required_doc_ids": ["paper_a"],
        "minimum_distinct_sources": 1, "grader_notes": [],
        "source_paper_id": "paper_a",
    }, {
        "id": "q2", "split": "pilot_evaluation", "question": "What color was reported?",
        "answer": "Blue", "expected_citations": ["paper_a"],
        "expected_answerability": "sufficient", "required_doc_ids": ["paper_a"],
        "minimum_distinct_sources": 1, "grader_notes": [],
        "source_paper_id": "paper_a",
    }]
    benchmark_path = dataset / "dev/benchmark.json"
    _write(benchmark_path, {
        "schema_version": "research-benchmark-v1", "benchmark_id": "fixture",
        "corpus_hash": corpus_hash, "reserved_doc_ids": ["paper_a"], "questions": questions,
    })
    benchmark_hash = runner._file_sha256(benchmark_path)
    _write(dataset / "dev/manifest.json", {
        "schema_version": "research-snapshot-v1", "corpus_hash": corpus_hash,
        "split_status": "previously_used_development",
        "benchmark_sha256": benchmark_hash,
        "selected_question_ids": ["q1", "q2"],
        "selected_target_paper_ids": ["paper_a"],
        "papers": [{"doc_id": "paper_a"}],
    })
    splits_path = dataset / "splits.json"
    _write(splits_path, {
        "schema_version": "envoybench-splits-v1",
        "splits": {"dev": {"question_ids": ["q1", "q2"]}},
    })
    _write(dataset / "manifest.json", {
        "schema_version": "envoybench-data-manifest-v1",
        "frozen_splits_sha256": runner._file_sha256(splits_path),
        "splits": {"dev": {
            "benchmark": "dev/benchmark.json", "corpus": "dev/corpus",
            "manifest": "dev/manifest.json",
            "status": "previously_used_development",
            "benchmark_sha256": benchmark_hash,
            "corpus_hash": corpus_hash,
            "question_count": 2,
        }},
    })
    models = tmp_path / "models.json"
    _write(models, {
        "schema_version": "envoybench-models-v1",
        "models": [
            {"key": "base", "model_id": "Qwen/Qwen3-8B", "revision": "base-commit",
             "endpoint_env": "TEST_QWEN_ENDPOINT",
             "decoding": {"temperature": 0, "max_tokens": 32, "top_p": 1}},
            {"key": "v5", "model_id": "envoy", "revision": "v5-commit",
             "adapter_id": "test/v5", "adapter_revision": "v5-commit",
             "endpoint_env": "TEST_QWEN_ENDPOINT", "api_key_env": "TEST_MODEL_KEY",
             "decoding": {"temperature": 0, "max_tokens": 32, "top_p": 1}},
        ],
    })
    monkeypatch.setenv("TEST_QWEN_ENDPOINT", "http://localhost:8000/v1")
    monkeypatch.setenv("TEST_MODEL_KEY", "fixture-secret-never-persist")
    return dataset, models, tmp_path / "run"


def test_validate_only_hash_checks_and_never_calls_docker_or_model(
    frozen: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, models, output = frozen
    monkeypatch.setattr(runner, "preflight_sandbox", lambda: pytest.fail("Docker was called"))
    monkeypatch.setattr(runner, "run_eval", lambda **_: pytest.fail("model was called"))
    validated = runner.run(dataset, "dev", models, output, validate_only=True)
    assert validated["question_ids"] == ["q1", "q2"]
    assert validated["split_status"] == "previously_used_development"
    assert validated["verifier_protocol"] == "fail-closed-v2"
    assert not output.exists()

    doc = dataset / "dev/corpus/paper_a.json"
    doc.write_text(doc.read_text().replace("Blue", "Red"))
    with pytest.raises(ValueError, match="corpus contents differ"):
        runner.run(dataset, "dev", models, output, validate_only=True)


def test_changed_benchmark_answer_is_rejected_before_model_calls(
    frozen: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, models, output = frozen
    monkeypatch.setattr(runner, "preflight_sandbox", lambda: pytest.fail("Docker was called"))
    monkeypatch.setattr(runner, "run_eval", lambda **_: pytest.fail("model was called"))
    benchmark_path = dataset / "dev/benchmark.json"
    benchmark = json.loads(benchmark_path.read_text())
    benchmark["questions"][0]["answer"] = "Red"
    _write(benchmark_path, benchmark)
    with pytest.raises(ValueError, match="benchmark bytes differ"):
        runner.run(dataset, "dev", models, output, validate_only=True)
    from benchmarks.envoybench.score import _load_frozen_cli_benchmark

    with pytest.raises(ValueError, match="benchmark bytes differ"):
        _load_frozen_cli_benchmark(benchmark_path)


def test_docker_preflight_rejects_old_image_and_does_not_call_model(
    frozen: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, models, output = frozen
    monkeypatch.setattr(runner, "run_eval", lambda **_: pytest.fail("model was called"))
    def old_image(*args, **kwargs):
        return subprocess.CompletedProcess(
            args, 0,
            stdout=json.dumps([{
                "Config": {"Labels": {}}, "Id": "sha256:old",
                "RepoTags": ["rlm-sandbox:latest"],
            }]),
            stderr="",
        )

    monkeypatch.setattr(runner.subprocess, "run", old_image)
    with pytest.raises(RuntimeError, match="org.envoybench.sandbox=v1"):
        runner.run(dataset, "dev", models, output)
    assert not output.exists()


def test_docker_preflight_can_inspect_exact_tag_via_image_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def docker_run(args, **kwargs):
        if args[:3] == ["docker", "image", "inspect"] and args[-1] == "rlm-sandbox":
            return subprocess.CompletedProcess(args, 1, stdout="", stderr="No such image")
        if args[:3] == ["docker", "image", "ls"]:
            return subprocess.CompletedProcess(
                args, 0, stdout="rlm-sandbox:latest sha256:checked\n", stderr=""
            )
        assert args == ["docker", "image", "inspect", "sha256:checked"]
        return subprocess.CompletedProcess(args, 0, stdout=json.dumps([{
            "Config": {"Labels": {"org.envoybench.sandbox": "v1"}},
            "Id": "sha256:checked", "RepoTags": ["rlm-sandbox:latest"],
            "RepoDigests": ["rlm-sandbox@sha256:registry"],
        }]), stderr="")

    monkeypatch.setattr(runner.subprocess, "run", docker_run)
    assert runner.preflight_sandbox() == {
        "image_id": "sha256:checked", "repo_digests": ["rlm-sandbox@sha256:registry"],
    }


def test_paired_run_records_full_matrix_without_endpoint_or_key(
    frozen: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, models, output = frozen
    monkeypatch.setattr(runner, "preflight_sandbox", lambda: {
        "image_id": "sha256:sandbox", "repo_digests": [],
    })
    invocations = []

    def fake_eval(**kwargs):
        assert kwargs["use_docker"] is True
        assert kwargs["require_evidence"] is True
        assert kwargs["evidence_verifier"] is True
        assert kwargs["escalate_after_verifier_failure"] is True
        key, factory = next(iter(kwargs["policies"].items()))
        policy = factory()
        assert policy.system_prompt == runner.QASPER_SYSTEM_PROMPT
        assert policy.extra_body["top_p"] == 1
        assert policy.timeout == 120
        invocations.append((key, [q["id"] for q in kwargs["questions"]]))
        return [EvalResult(
            question_id=q["id"], question=q["question"], policy_name=key,
            reward=0.1, answer_score=0.1, citation_precision=1, citation_recall=1,
            efficiency_bonus=0, steps=2, predicted_answer="Blue",
            predicted_citations=["paper_a"],
            predicted_evidence=[{"doc_id": "paper_a", "start": 0, "end": 4}],
            duration_seconds=1.5,
            trajectory=[
                StepRecord(step=1, action='print(search("result"))', observation="paper_a",
                           reward=0, done=False,
                           reasoning="Find the source first" if key == "v5" else None),
                StepRecord(step=2, action='SUBMIT: Blue CITATIONS: ["paper_a"]',
                           observation="Submitted", reward=0.1, done=True),
            ],
        ) for q in kwargs["questions"]]

    monkeypatch.setattr(runner, "run_eval", fake_eval)
    manifest = runner.run(dataset, "dev", models, output)
    assert invocations == [("base", ["q1"]), ("base", ["q2"]),
                           ("v5", ["q1"]), ("v5", ["q2"])]
    assert manifest["status"] == "complete"
    assert manifest["verifier_protocol"] == "fail-closed-v2"
    assert manifest["escalate_after_verifier_failure"] is True
    assert manifest["full_split"] is True
    assert manifest["sandbox_image"]["image_id"] == "sha256:sandbox"
    assert manifest["models"][0]["serving_hardware"] == "unreported"
    assert len(manifest["models"]) == 2
    assert manifest["models"][1]["adapter_revision"] == "v5-commit"
    rows = json.loads((output / "results.json").read_text())
    assert {(r["question_id"], r["model_key"]) for r in rows} == {
        ("q1", "base"), ("q2", "base"), ("q1", "v5"), ("q2", "v5"),
    }
    assert all(row["status"] == "submitted" for row in rows)
    assert rows[0]["trajectory"][0]["output"] == "paper_a"
    assert "reasoning" not in rows[0]["trajectory"][0]
    assert next(row for row in rows if row["model_key"] == "v5")["trajectory"][0][
        "reasoning"
    ] == "Find the source first"
    assert rows[0]["predicted_evidence"][0]["end"] == 4
    assert rows[0]["environment_diagnostics_not_benchmark_score"]["legacy_reward"] == 0.1
    serialized = (output / "manifest.json").read_text() + (output / "results.json").read_text()
    assert "fixture-secret-never-persist" not in serialized
    assert "http://localhost:8000/v1" not in serialized
    assert '"answer": "Blue"' not in serialized
    assert not (output / "results.partial.json").exists()


def test_verifier_protocol_versions_are_bound_to_separate_runs(
    frozen: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, models, output = frozen
    monkeypatch.setattr(runner, "preflight_sandbox", lambda: {
        "image_id": "sha256:sandbox", "repo_digests": [],
    })
    observed_escalation: list[bool] = []

    def incomplete_eval(**kwargs):
        observed_escalation.append(kwargs["escalate_after_verifier_failure"])
        return []

    monkeypatch.setattr(runner, "run_eval", incomplete_eval)
    with pytest.raises(ValueError, match="verifier_protocol"):
        runner.run(dataset, "dev", models, output, verifier_protocol="unknown")
    assert not output.exists()

    with pytest.raises(RuntimeError, match="incomplete question matrix"):
        runner.run(dataset, "dev", models, output, verifier_protocol="legacy-v1")
    legacy = json.loads((output / "manifest.json").read_text())
    assert "verifier_protocol" not in legacy
    assert legacy["escalate_after_verifier_failure"] is False

    modern_output = output.with_name("fail-closed-run")
    with pytest.raises(RuntimeError, match="incomplete question matrix"):
        runner.run(dataset, "dev", models, modern_output)
    modern = json.loads((modern_output / "manifest.json").read_text())
    assert modern["verifier_protocol"] == "fail-closed-v2"
    assert modern["escalate_after_verifier_failure"] is True
    assert modern["comparison_id"] != legacy["comparison_id"]
    assert observed_escalation == [False, True]


def test_partial_model_run_never_writes_complete_result(
    frozen: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, models, output = frozen
    monkeypatch.setattr(runner, "preflight_sandbox", lambda: {
        "image_id": "sha256:sandbox", "repo_digests": [],
    })
    monkeypatch.setattr(runner, "run_eval", lambda **_: [])
    with pytest.raises(RuntimeError, match="incomplete question matrix"):
        runner.run(dataset, "dev", models, output)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "incomplete"
    assert not (output / "results.json").exists()


def test_model_config_refuses_inline_key_and_remote_http(
    frozen: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, models, _ = frozen
    data = json.loads(models.read_text())
    data["models"][0]["api_key"] = "oops"
    _write(models, data)
    with pytest.raises(ValueError, match="credentials"):
        runner.load_models(models)
    del data["models"][0]["api_key"]
    monkeypatch.setenv("TEST_QWEN_ENDPOINT", "http://models.example.com/v1")
    _write(models, data)
    with pytest.raises(ValueError, match="HTTPS"):
        runner.load_models(models)
    monkeypatch.setenv("TEST_QWEN_ENDPOINT", "http://localhost:8000/v1")
    data["models"][0]["serving_hourly_usd"] = -1
    _write(models, data)
    with pytest.raises(ValueError, match="serving_hourly_usd"):
        runner.load_models(models)


def test_model_config_records_optional_context_window(frozen):
    _, models, _ = frozen
    assert "serving_context_window_tokens" not in runner.load_models(models)[0]["safe"]
    config = json.loads(models.read_text())
    config["models"][0]["serving_context_window_tokens"] = 8192
    _write(models, config)
    assert runner.load_models(models)[0]["safe"]["serving_context_window_tokens"] == 8192


def test_model_config_records_and_uses_longer_reasoning_timeout(frozen):
    _, models, _ = frozen
    config = json.loads(models.read_text())
    config["models"][0]["request_timeout_seconds"] = 300
    _write(models, config)
    model = runner.load_models(models)[0]
    assert model["safe"]["request_timeout_seconds"] == 300
    assert runner._model_factory(model, 42)().timeout == 300


@pytest.mark.parametrize("value", [True, 0, -1, 601, 120.5, "300"])
def test_model_config_rejects_invalid_reasoning_timeout(frozen, value):
    _, models, _ = frozen
    config = json.loads(models.read_text())
    config["models"][0]["request_timeout_seconds"] = value
    _write(models, config)
    with pytest.raises(ValueError, match="request_timeout_seconds"):
        runner.load_models(models)


@pytest.mark.parametrize("context_window", [True, False, 0, -1, 31, 32, 8192.0, "8192"])
def test_model_config_rejects_invalid_or_too_small_context_window(frozen, context_window):
    _, models, _ = frozen
    config = json.loads(models.read_text())
    # Fixture decoding budget is 32 tokens; serving context must leave room
    # for input as well as generated output, not merely equal max_tokens.
    config["models"][0]["serving_context_window_tokens"] = context_window
    _write(models, config)
    with pytest.raises(ValueError, match="serving_context_window_tokens must exceed max_tokens"):
        runner.load_models(models)


def test_result_row_preserves_partial_token_telemetry_after_endpoint_failure():
    metadata = {"token_usage": {
        "request_count": 2,
        "failed_request_count": 1,
        "usage_observed_request_count": 1,
        "usage_missing_request_count": 1,
        "usage_complete": False,
        "observed_prompt_tokens": 300,
        "observed_completion_tokens": 50,
        "prompt_tokens": None,
        "completion_tokens": None,
        "requests": [
            {"prompt_tokens": 300, "completion_tokens": 50, "failed": False},
            {"prompt_tokens": None, "completion_tokens": None, "failed": True},
        ],
    }}
    result = EvalResult(
        question_id="q1", question="Question", policy_name="base", reward=0,
        answer_score=0, citation_precision=0, citation_recall=0, efficiency_bonus=0,
        steps=1, duration_seconds=3.0, status="error",
        error="RuntimeError: model endpoint returned HTTP 400",
        policy_metadata=metadata,
        trajectory=[StepRecord(
            step=1, action="print('inspect')", observation="source", reward=0, done=False,
        )],
    )
    manifest = {
        "run_id": "r", "comparison_id": "paired", "benchmark_id": "b",
        "benchmark_hash": "benchmark-hash", "corpus_hash": "corpus-hash",
        "split": "test_candidate", "split_status": "unreviewed",
    }
    row = runner._result_row(result, "base", manifest, [])
    assert row["status"] == "error"
    assert row["steps"] == 1
    assert row["policy_metadata"] == metadata
    assert row["policy_metadata"]["token_usage"]["prompt_tokens"] is None


def test_result_row_serializes_generated_reasoning_only_when_present() -> None:
    result = EvalResult(
        question_id="q1", question="Question", policy_name="qwen", reward=0,
        answer_score=0, citation_precision=0, citation_recall=0, efficiency_bonus=0,
        steps=2, trajectory=[
            StepRecord(step=1, action="print('a')", observation="a", reward=0,
                       done=False, reasoning="Inspect source a"),
            StepRecord(step=2, action="SUBMIT: a CITATIONS: []", observation="submitted",
                       reward=0, done=True),
        ],
    )
    manifest = {
        "run_id": "r", "comparison_id": "c", "benchmark_id": "b",
        "benchmark_hash": "h", "corpus_hash": "h", "split": "dev",
        "split_status": "unreviewed",
    }
    row = runner._result_row(result, "qwen", manifest, [])
    assert row["trajectory"][0]["reasoning"] == "Inspect source a"
    assert "reasoning" not in row["trajectory"][1]
