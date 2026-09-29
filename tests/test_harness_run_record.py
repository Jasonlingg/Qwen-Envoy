"""Run metadata stays outside the vault and is not silently overwritten."""

from __future__ import annotations

import json

import pytest

from src.harness.run_record import RunRecord


def test_run_record_keeps_inputs_events_and_artifacts_outside_vault(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    root = tmp_path / "runs"
    record = RunRecord(
        root,
        vault,
        "run-1",
        {
            "corpus_hash": "a" * 64,
            "checkpoint": "Qwen/Qwen3-8B",
            "seed": 42,
        },
    )
    record.append_event("evidence_verified", {"passages": 1})
    record.save_artifact("bundle.json", {"passage_ids": ["p_1"]})

    manifest = json.loads(record.manifest.read_text())
    events = [json.loads(line) for line in record.events.read_text().splitlines()]
    assert manifest["inputs"]["seed"] == 42
    assert [event["state"] for event in events] == ["created", "evidence_verified"]
    assert json.loads((record.directory / "bundle.json").read_text()) == {"passage_ids": ["p_1"]}
    assert list(vault.rglob("*")) == []

    with pytest.raises(FileExistsError):
        record.save_artifact("bundle.json", {})
    with pytest.raises(FileExistsError):
        RunRecord(root, vault, "run-1", {"seed": 42})


def test_run_record_rejects_vault_root_and_unsafe_artifact_path(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    with pytest.raises(ValueError, match="outside the vault"):
        RunRecord(vault / "runs", vault, "run-1", {"seed": 42})
    record = RunRecord(tmp_path / "runs", vault, "run-1", {"seed": 42})
    with pytest.raises(ValueError, match="safe .json basename"):
        record.save_artifact("../escape.json", {})
