"""Offline protocol checks; no GPU, endpoint, Docker or model calls."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from benchmarks.envoybench import inference_ablation as ablation
from benchmarks.envoybench import run as runner


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    dataset = tmp_path / "dataset"
    benchmark_path = dataset / "dev/benchmark.json"
    benchmark = {"benchmark_id": "fixture-dev", "corpus_hash": "frozen-corpus",
                 "questions": [{"id": f"q{i}", "question": f"Question {i}", "answer": "gold"}
                               for i in range(1, 5)]}
    write(benchmark_path, benchmark)
    write(dataset / "manifest.json", {"frozen": True})
    monkeypatch.setattr(runner, "load_split", lambda data, split: (
        benchmark, {"split_status": "previously_used_development"},
        benchmark_path, dataset / "corpus", {},
    ))
    model = {"key": "qwen_v5", **ablation.PINNED_IDENTITY,
             "decoding": copy.deepcopy(ablation.DECODING),
             "extra_body": copy.deepcopy(ablation.EXTRA_BODY),
             "serving_hardware": "fixture A40", "serving_runtime": "vLLM 0.30.0 fixture",
             "serving_context_window_tokens": 8192, "send_seed": False,
             "endpoint_env": "ABLATION_ENDPOINT", "api_key_env": "ABLATION_KEY"}
    source = tmp_path / "source"
    write(source / "manifest.json", {"schema_version": runner.RUN_SCHEMA, "status": "complete",
                                     "run_id": "original-run", "models": [model]})
    # Deliberately unreadable results; preparation must never select based on outcomes.
    (source / "results.json").write_text("not JSON; do not read outputs")
    models = tmp_path / "models.json"
    write(models, {"schema_version": runner.MODEL_SCHEMA, "models": [model]})
    prepared = tmp_path / "prepared"
    return {"dataset": dataset, "source": source, "models": models, "prepared": prepared,
            "model": model, "root": tmp_path, "benchmark": benchmark}


def prepare(setup: dict) -> dict:
    return ablation.prepare(setup["dataset"], setup["source"], setup["prepared"])


def fake_run(setup: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ABLATION_ENDPOINT", "http://localhost:8000/v1")
    monkeypatch.setenv("ABLATION_KEY", "secret-never-persist")

    def run(dataset, split, models_path, output, **kwargs):
        assert split == "dev" and kwargs == {
            "model_keys": ["qwen_v5"], "question_ids": ["q1", "q2", "q3"],
            "seed": 42, "max_steps": 15,
        }
        plan = ablation._read_plan(setup["prepared"])
        manifest = {name: plan[name] for name in (
            "benchmark_id", "benchmark_hash", "corpus_hash", "split", "question_ids",
            "max_steps", "seed", "benchmark_file_sha256", "dataset_index_sha256",
        )}
        manifest.update({"status": "complete", "run_id": output.name,
                         "comparison_id": "same-protocol", "sandbox_image": {"image_id": "fixed"},
                         "models": [runner.load_models(models_path)[0]["safe"]],
                         "implementation_sha256": runner._implementation_hashes()})
        seconds = 8 if output.name.startswith("on") else 10
        rows = [{"question_id": qid, "run_id": manifest["run_id"], "model_key": "qwen_v5",
                 "comparison_id": "same-protocol", "status": "submitted", "error": None,
                 "predicted_answer": "Blue", "predicted_citations": ["paper"],
                 "predicted_evidence": [{"doc_id": "paper", "start": 0, "end": 4}],
                 "trajectory": [{"action": 'read("paper")'}, {"action": "SUBMIT: Blue"}],
                 "duration_seconds": seconds + 2,
                 "policy_metadata": {"token_usage": {"request_count": 2,
                                                      "failed_request_count": 0,
                     "requests": [{"duration_seconds": seconds / 2, "prompt_tokens": 100,
                                   "completion_tokens": 10, "cached_prompt_tokens": 0},
                                  {"duration_seconds": seconds / 2, "prompt_tokens": 150,
                                   "completion_tokens": 10,
                                   "cached_prompt_tokens": 100 if output.name.startswith("on")
                                   else 0}]}}}
                for qid in plan["question_ids"]]
        write(output / "manifest.json", manifest)
        write(output / "results.json", rows)
        return manifest

    monkeypatch.setattr(runner, "run", run)


def execute(setup: dict, phase: str, **overrides) -> Path:
    output = setup["root"] / phase
    kwargs = {"phase": phase, "server_prefix_caching": phase.split("-")[0],
              "server_process_id": f"process-{phase}", "fresh_server_process": True,
              "warmup_cache_reset_complete": True, **overrides}
    ablation.execute(setup["prepared"], setup["dataset"], setup["source"], setup["models"],
                      output, **kwargs)
    return output


def phases(setup: dict, monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    prepare(setup)
    fake_run(setup, monkeypatch)
    return [execute(setup, phase["phase"]) for phase in ablation.PHASES]


def update_phase(directory: Path, file: str, change) -> None:
    value = json.loads((directory / file).read_text())
    change(value)
    write(directory / file, value)
    if file != "ablation.json":
        sidecar = json.loads((directory / "ablation.json").read_text())
        sidecar["artifact_sha256"][file] = runner._file_sha256(directory / file)
        write(directory / "ablation.json", sidecar)


def test_prepare_is_offline_deterministic_and_ignores_outputs(setup: dict, monkeypatch) -> None:
    monkeypatch.delenv("ABLATION_ENDPOINT", raising=False)
    monkeypatch.delenv("ABLATION_KEY", raising=False)
    monkeypatch.setattr(runner, "load_models", lambda *a: pytest.fail("resolved endpoints"))
    monkeypatch.setattr(runner, "run", lambda *a, **k: pytest.fail("ran inference"))
    plan = prepare(setup)
    assert plan == ablation.build_plan(setup["dataset"], setup["source"])
    assert plan["question_ids"] == ["q1", "q2", "q3"]
    assert [p["prefix_caching"] for p in plan["phases"]] == ["off", "on", "on", "off"]
    assert plan["requests_executed"] == 0
    assert not (setup["prepared"] / "results.json").exists()
    assert "secret-never-persist" not in json.dumps(plan)


def test_matching_phases_report_timing_signal_not_quality(setup: dict, monkeypatch) -> None:
    directories = phases(setup, monkeypatch)
    report = ablation.compare(setup["prepared"], directories)
    assert report["median_paired_request_reduction_fraction"] == pytest.approx(0.2)
    assert report["median_paired_elapsed_reduction_fraction"] == pytest.approx(1 / 6)
    assert report["decision"] == "diagnostic_timing_signal"
    assert report["quality_promotion"] is False
    assert len(report["timing_coverage"]) == 12
    assert all(row["request_timing_complete"] for row in report["timing_coverage"])
    assert next(row for row in report["timing_coverage"] if row["phase"] == "on-1")[
        "token_usage"]["cached_prompt_tokens"] == 100


@pytest.mark.parametrize("field,value", [
    ("adapter_sha256", "other"), ("serving_context_window_tokens", 16384),
    ("serving_hardware", "other GPU"), ("serving_runtime", "other runtime"),
    ("decoding", {"max_tokens": 1024, "temperature": 0.1, "top_p": 1}),
])
def test_execute_refuses_changed_configuration_before_runner(
    setup: dict, monkeypatch, field, value,
):
    prepare(setup)
    monkeypatch.setenv("ABLATION_ENDPOINT", "http://localhost:8000/v1")
    monkeypatch.setenv("ABLATION_KEY", "fixture-secret")
    config = json.loads(setup["models"].read_text())
    config["models"][0][field] = value
    write(setup["models"], config)
    monkeypatch.setattr(runner, "run", lambda *a, **k: pytest.fail("called runner"))
    with pytest.raises(ValueError, match="differs"):
        execute(setup, "off-1")


@pytest.mark.parametrize("overrides", [{"server_prefix_caching": "on"},
    {"fresh_server_process": False}, {"warmup_cache_reset_complete": False}])
def test_server_declarations_are_required(setup: dict, monkeypatch, overrides):
    prepare(setup)
    monkeypatch.setattr(runner, "run", lambda *a, **k: pytest.fail("called runner"))
    with pytest.raises(ValueError, match="differs|required"):
        execute(setup, "off-1", **overrides)


@pytest.mark.parametrize("change", [
    lambda rows: rows[0].update(predicted_answer="Red"),
    lambda rows: rows[0]["trajectory"][0].update(action='search("different")'),
    lambda rows: rows[0].update(status="error", error="failure"),
    lambda rows: rows[0]["trajectory"][0].update(
        observation="\nSTDERR:\nTraceback (most recent call last):\nTypeError: broken\n"),
    lambda rows: rows[0].update(duration_seconds=0),
    lambda rows: rows[0]["policy_metadata"]["token_usage"]["requests"][0].pop("duration_seconds"),
    lambda rows: rows[0]["policy_metadata"]["token_usage"]["requests"][0].pop("prompt_tokens"),
    lambda rows: rows[0]["policy_metadata"]["token_usage"]["requests"][0].update(
        completion_tokens=11),
])
def test_output_failure_and_timing_confounds_withhold_gain(setup: dict, monkeypatch, change):
    directories = phases(setup, monkeypatch)
    update_phase(directories[1], "results.json", change)
    report = ablation.compare(setup["prepared"], directories)
    assert report["timing_gain_claim_eligible"] is False
    assert report["threshold_met"] is False
    assert report["decision"] == "inconclusive_confounded"


def test_compare_rejects_missing_questions_and_reused_process(setup: dict, monkeypatch):
    directories = phases(setup, monkeypatch)
    with pytest.raises(ValueError, match="four"):
        ablation.compare(setup["prepared"], directories[:3])
    update_phase(directories[1], "ablation.json",
                 lambda sidecar: sidecar.update(server_process_id="process-off-1"))
    with pytest.raises(ValueError, match="unique fresh process"):
        ablation.compare(setup["prepared"], directories)
    update_phase(directories[1], "ablation.json",
                 lambda sidecar: sidecar.update(server_process_id="process-on-1"))
    update_phase(directories[1], "results.json", lambda rows: rows.pop())
    with pytest.raises(ValueError, match="matrix"):
        ablation.compare(setup["prepared"], directories)


def test_prepared_implementation_change_is_rejected_before_requests(setup: dict, monkeypatch):
    prepare(setup)
    monkeypatch.setattr(ablation, "_implementation", lambda: {"changed": "hash"})
    monkeypatch.setattr(runner, "run", lambda *a, **k: pytest.fail("called runner"))
    with pytest.raises(ValueError, match="implementation or settings changed"):
        execute(setup, "off-1")


def test_compare_rejects_wrong_phase_order(setup: dict, monkeypatch):
    directories = phases(setup, monkeypatch)
    update_phase(directories[1], "ablation.json",
                 lambda sidecar: sidecar.update(phase_started_at_utc="2000-01-01T00:00:00+00:00"))
    with pytest.raises(ValueError, match="execution order"):
        ablation.compare(setup["prepared"], directories)


def test_cli_requires_explicit_mode():
    with pytest.raises(SystemExit) as raised:
        ablation.main(["--output", "unused"])
    assert raised.value.code == 2
