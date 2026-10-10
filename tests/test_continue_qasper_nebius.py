"""Offline safety checks for the bounded Nebius continuation wrapper."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from scripts import continue_qasper_nebius as continuation


@pytest.fixture
def real_plan(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch,
) -> dict:
    # The completed continuation is pinned to the old source hashes. Bypass only
    # that guard here so the remaining archived-input checks stay exercised.
    monkeypatch.setattr(continuation, "_implementation_variance", lambda _manifest: {
        "file": "benchmarks/envoybench/run.py",
    })
    output = tmp_path_factory.mktemp("continuation") / "extension"
    return continuation.prepare_continuation(output=output)


def _copy_source(tmp_path: Path) -> Path:
    release = tmp_path / "release"
    release.mkdir()
    shutil.copytree(continuation.ORIGINAL, release / "nebius-run")
    shutil.copy2(continuation.RELEASE / "SHA256SUMS", release / "SHA256SUMS")
    return release / "nebius-run"


def _mutate_pinned_json(original: Path, filename: str, edit) -> None:
    path = original / filename
    value = json.loads(path.read_text())
    edit(value)
    path.write_text(json.dumps(value, indent=2) + "\n")
    checksum_path = original.parent / "SHA256SUMS"
    relative = f"nebius-run/{filename}"
    lines = checksum_path.read_text().splitlines()
    lines = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {relative}"
        if line.endswith(f"  {relative}") else line
        for line in lines
    ]
    checksum_path.write_text("\n".join(lines) + "\n")


def test_real_preflight_finds_only_two_questions_and_cumulative_guard(real_plan: dict) -> None:
    ids = real_plan["continuation_question_ids"]
    assert len(ids) == 2
    assert ids[0].startswith("qasper_test_f88f45")
    assert ids[1].startswith("qasper_test_fd556a")
    assert real_plan["question_39_restarted_at_step"] == 1
    assert real_plan["question_39_original_interrupted_after_steps"] == 11
    assert real_plan["implementation_variance"]["file"] == "benchmarks/envoybench/run.py"
    budget = continuation._seed_budget(real_plan)
    assert budget.snapshot()["requests"] == 366
    assert budget.snapshot()["estimated_usd"] == 1.947862
    assert budget.snapshot()["prompt_tokens"] == 1819534
    assert budget.snapshot()["completion_tokens"] == 42776
    assert budget.snapshot()["config"]["max_estimated_usd"] == 25.0
    assert budget.snapshot()["config"]["max_requests"] == 600
    assert budget.halted_reason is None


def test_changed_runtime_blocks_reusing_historical_continuation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="implementation differs"):
        continuation.prepare_continuation(output=tmp_path / "extension")


def test_cli_defaults_to_dry_run_without_dispatch(
    real_plan: dict, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(continuation, "prepare_continuation", lambda **_: real_plan)
    monkeypatch.setattr(
        continuation.runner, "run", lambda *_, **__: pytest.fail("paid runner called"),
    )
    assert continuation.main([]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["mode"] == "dry_run"
    assert printed["total_inference_budget"]["max_estimated_usd"] == 25.0
    assert not Path(real_plan["output"]).exists()


def test_execute_dispatches_exact_subset_with_seeded_budget_and_writes_lineage(
    real_plan: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "new-run"
    plan = {**real_plan, "output": str(output)}
    monkeypatch.setenv("NEBIUS_API_KEY", "test-only-placeholder")
    calls = []

    def fake_run(dataset, split, models, destination, **kwargs):
        calls.append((dataset, split, models, destination, kwargs))
        assert destination == output
        assert kwargs["model_keys"] == [continuation.MODEL_KEY]
        assert kwargs["question_ids"] == plan["continuation_question_ids"]
        assert kwargs["seed"] == 42 and kwargs["max_steps"] == 15
        assert kwargs["verifier_protocol"] == "legacy-v1"
        assert kwargs["budget"].requests == 366
        assert kwargs["budget"].estimated_usd == 1.947862
        assert kwargs["budget"].config["max_estimated_usd"] == 25.0
        destination.mkdir()
        (destination / "manifest.json").write_text(json.dumps({
            "run_id": "new-run-id", "status": "complete",
            "question_ids": kwargs["question_ids"],
            "inference_budget": kwargs["budget"].config,
        }))
        return {"question_ids": kwargs["question_ids"]}

    monkeypatch.setattr(continuation.runner, "run", fake_run)
    result = continuation.execute_continuation(
        plan, env_file=tmp_path / "unused.env"
    )
    assert result["question_ids"] == plan["continuation_question_ids"]
    assert len(calls) == 1
    lineage = json.loads((output / "continuation-lineage.json").read_text())
    assert lineage["original_run_id"] == plan["original_run_id"]
    assert lineage["original_files_sha256"] == plan["original_files_sha256"]
    assert lineage["continuation_run_id"] == "new-run-id"
    assert lineage["continuation_usage"]["requests"] == 366
    assert lineage["question_39_restarted_at_step"] == 1
    assert "test-only-placeholder" not in (output / "continuation-lineage.json").read_text()


def test_execute_needs_secret_before_dispatch(
    real_plan: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = {**real_plan, "output": str(tmp_path / "new-run")}
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    monkeypatch.setattr(
        continuation.runner, "run", lambda *_, **__: pytest.fail("paid runner called"),
    )
    with pytest.raises(ValueError, match="NEBIUS_API_KEY is required"):
        continuation.execute_continuation(plan, env_file=tmp_path / "empty.env")
    assert not Path(plan["output"]).exists()


def test_preflight_rejects_corrupt_original_row_even_with_updated_checksum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _copy_source(tmp_path)
    _mutate_pinned_json(
        original, "results.partial.json",
        lambda rows: rows[37].update({"status": "error"}),
    )
    monkeypatch.setattr(continuation, "_implementation_variance", lambda _manifest: None)
    with pytest.raises(ValueError, match="question 38 was not terminal"):
        continuation.prepare_continuation(original=original, output=tmp_path / "new-run")


def test_preflight_rejects_bad_usage_even_with_updated_checksum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _copy_source(tmp_path)
    _mutate_pinned_json(
        original, "usage-budget.json",
        lambda usage: usage.update({"requests": 0}),
    )
    monkeypatch.setattr(continuation, "_implementation_variance", lambda _manifest: None)
    with pytest.raises(ValueError, match="original usage requests differs"):
        continuation.prepare_continuation(original=original, output=tmp_path / "new-run")


def test_preflight_rejects_output_inside_original(real_plan: dict) -> None:
    with pytest.raises(ValueError, match="inside the original run"):
        continuation.prepare_continuation(output=continuation.ORIGINAL / "extension")
