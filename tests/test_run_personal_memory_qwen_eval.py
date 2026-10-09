"""CPU-only checks for the pinned paired GPU runner's artifact recovery."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts import run_personal_memory_qwen_eval as runner
from scripts.personal_memory_eval import freeze
from src.eval.research_review import prepare_review, review_markdown


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def _model_artifacts(
    path: Path, label: str, recipe: dict, base: Path, adapter: Path, *, complete: bool = True
) -> None:
    policy = "qwen_sft_policy" if label == "v5" else "qwen_base_policy"
    checkpoint = str(adapter) if label == "v5" else None
    manifest = {
        **{key: value for key, value in recipe.items() if key != "model_identity"},
        "run_id": f"test-{label}",
        "run_label": label,
        "comparison_id": runner.configuration_hash(runner._protocol(recipe)),
        "reward_weights": runner.REWARD_WEIGHTS,
        "base_model": str(base),
        "checkpoint_id": checkpoint,
        "decoding": [
            json.dumps(
                {
                    "max_tokens": recipe["max_tokens"],
                    "temperature": recipe["temperature"],
                    "top_p": recipe["top_p"],
                },
                sort_keys=True,
            )
        ],
        "policy_settings": {
            policy: {
                "model": str(base),
                "checkpoint": checkpoint,
                "max_tokens": recipe["max_tokens"],
                "temperature": recipe["temperature"],
                "top_p": recipe["top_p"],
            },
        },
    }
    questions = json.loads(runner.QUESTIONS.read_text())["questions"]
    selected = questions if complete else questions[:-1]
    rows = [
        {
            "question_id": item["id"],
            "question": item["question"],
            "policy": policy,
            "run_label": label,
            "run_id": manifest["run_id"],
            "comparison_id": manifest["comparison_id"],
            "checkpoint_id": checkpoint,
            "status": "error" if label == "base" and index == 0 else "completed",
            "predicted_answer": "",
            "predicted_citations": [],
            "predicted_evidence": [],
            "duration_seconds": 0.1,
            "trajectory": [],
        }
        for index, item in enumerate(selected)
    ]
    _write(path, rows)
    _write(path.with_suffix(".manifest.json"), manifest)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    freeze(snapshot)
    base, adapter = tmp_path / "base", tmp_path / "adapter"
    identity = {
        "base_model": runner.BASE_ID,
        "base_revision": runner.BASE_REVISION,
        "base_path": str(base),
        "adapter_model": runner.V5_ID,
        "adapter_revision": runner.V5_REVISION,
        "adapter_path": str(adapter),
        "adapter_sha256": "test-adapter-sha256",
    }
    monkeypatch.setattr(runner, "_resolve_weights", lambda *_: (base, adapter, identity))
    output = tmp_path / "paired"
    calls = []

    def fake_run(command: list[str], *, env: dict[str, str]) -> None:
        calls.append(command[1])
        if command[1] == "scripts/run_eval.py":
            label = command[command.index("--run-label") + 1]
            path = Path(command[command.index("--output") + 1])
            recipe = json.loads((output / "run-config.json").read_text())
            _model_artifacts(path, label, recipe, base, adapter)
            if label == "base":
                raise subprocess.CalledProcessError(1, command)
        else:
            review_path = Path(command[command.index("--output") + 1])
            review, key, automatic = prepare_review(
                runner.QUESTIONS,
                snapshot / "corpus",
                [output / "base.json", output / "v5.json"],
                42,
            )
            review_path.mkdir()
            _write(review_path / "review.json", review)
            _write(review_path / "blind-key.json", key)
            _write(review_path / "automatic.json", automatic)
            (review_path / "review.md").write_text(review_markdown(review))

    monkeypatch.setattr(runner, "_run", fake_run)
    return snapshot, output, base, adapter, identity, calls


def test_complete_nonzero_base_still_runs_paired_arm_and_resumes_without_writes(setup, monkeypatch):
    snapshot, output, base, adapter, _, calls = setup
    result = runner.run(snapshot, output, base, adapter)
    assert result["questions"] == len(json.loads(runner.QUESTIONS.read_text())["questions"])
    assert calls == [
        "scripts/run_eval.py",
        "scripts/run_eval.py",
        "scripts/review_code_exec_pilot.py",
    ]
    assert json.loads((output / "base.json").read_text())[0]["status"] == "error"
    before = {
        path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()
    }
    monkeypatch.setattr(runner, "_run", lambda *_, **__: pytest.fail("reran GPU or review"))
    runner.run(snapshot, output, base, adapter, resume=True)
    after = {
        path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()
    }
    assert before == after


def test_resume_runs_only_missing_arm_and_downstream_outputs(setup):
    snapshot, output, base, adapter, _, calls = setup
    runner.run(snapshot, output, base, adapter)
    calls.clear()
    (output / "v5.json").unlink()
    (output / "v5.manifest.json").unlink()
    (output / "bm25.json").unlink()
    (output / "bm25.manifest.json").unlink()
    (output / "diagnostics.json").unlink()
    shutil.rmtree(output / "blind-review")
    runner.run(snapshot, output, base, adapter, resume=True)
    assert calls == ["scripts/run_eval.py", "scripts/review_code_exec_pilot.py"]
    assert (output / "diagnostics.json").is_file()


def test_nonzero_with_incomplete_transcript_fails_closed(setup, monkeypatch):
    snapshot, output, base, adapter, _, calls = setup
    original = runner._run

    def incomplete_run(command: list[str], *, env: dict[str, str]) -> None:
        if command[1] == "scripts/run_eval.py":
            label = command[command.index("--run-label") + 1]
            recipe = json.loads((output / "run-config.json").read_text())
            path = Path(command[command.index("--output") + 1])
            _model_artifacts(path, label, recipe, base, adapter, complete=False)
            raise subprocess.CalledProcessError(1, command)
        original(command, env=env)

    monkeypatch.setattr(runner, "_run", incomplete_run)
    with pytest.raises(ValueError, match="incomplete base transcript"):
        runner.run(snapshot, output, base, adapter)
    assert calls == []
    assert not (output / "v5.json").exists()


def test_resume_rejects_changed_identity_and_manifest_without_running(setup, monkeypatch):
    snapshot, output, base, adapter, identity, _ = setup
    runner.run(snapshot, output, base, adapter)
    monkeypatch.setattr(runner, "_run", lambda *_, **__: pytest.fail("reran model"))
    changed = {**identity, "adapter_sha256": "changed"}
    monkeypatch.setattr(runner, "_resolve_weights", lambda *_: (base, adapter, changed))
    with pytest.raises(ValueError, match="model identity"):
        runner.run(snapshot, output, base, adapter, resume=True)
    monkeypatch.setattr(runner, "_resolve_weights", lambda *_: (base, adapter, identity))
    manifest_path = output / "base.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["system_prompt_sha256"] = "0" * 64
    _write(manifest_path, manifest)
    with pytest.raises(ValueError, match="base system_prompt_sha256"):
        runner.run(snapshot, output, base, adapter, resume=True)
