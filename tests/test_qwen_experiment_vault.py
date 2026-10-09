"""The real-report demo stays tied to its source and does not invent a decision."""

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from scripts.build_qwen_experiment_vault import (
    DEFAULT_OUTPUT,
    DEFAULT_REPORT,
    build_vault,
    render_notes,
)
from scripts.launch_qwen_experiment_demo import prepare_workspace
from src.product.graph import build_graph
from src.product.impact import build_impact_review
from src.product.memory import freeze_vault
from src.research.agent import load_snapshot


def _metadata(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8").split("---", 2)[1])


def test_generated_thread_is_deterministic_and_source_derived(tmp_path):
    report = json.loads(DEFAULT_REPORT.read_text(encoding="utf-8"))
    digest = hashlib.sha256(DEFAULT_REPORT.read_bytes()).hexdigest()
    vault = tmp_path / "vault"
    notes = build_vault(vault)

    assert set(notes) == {"prior", "result", "decision"}
    assert len(list(vault.rglob("*.md"))) == 3
    assert {path.name for path in notes.values()} == set(render_notes())
    for role, path in notes.items():
        assert path.read_bytes() == (DEFAULT_OUTPUT / path.name).read_bytes()
        metadata = _metadata(path)
        assert metadata["source_repo_path"] == "reports/qasper-scale-sft-2026-09-25.json"
        assert metadata["source_sha256"] == digest
        assert metadata["effective_date"] == report["date"]
        assert metadata["review_status"] == "source_derived_unreviewed"
        assert metadata["kind"] == {"prior": "decision", "result": "result",
                                    "decision": "source"}[role]

    prior = notes["prior"].read_text(encoding="utf-8")
    measured = notes["result"].read_text(encoding="utf-8")
    decision = notes["decision"].read_text(encoding="utf-8")
    assert "retrospective reconstruction" in prior
    assert "**not** a contemporaneous" in prior
    assert "would not invalidate this rule" in prior
    for condition in report["decision_rule"]["promote_only_if"]:
        assert condition in prior
    assert f"[[{notes['prior'].stem}]]" in measured
    for arm, label in (("starting_sft", "starting SFT checkpoint"),
                       ("epoch_1", "after epoch 1"),
                       ("epoch_2", "after epoch 2")):
        assert f"**{report['results'][arm]['semantic']['pass']}/40**" in measured
        assert label in measured
    breakdown = report["semantic_passes_by_answerability"]
    for category in ("sufficient", "insufficient"):
        counts = " / ".join(str(breakdown[arm][category])
                            for arm in ("starting_sft", "epoch_1", "epoch_2"))
        assert counts in measured
    assert "not subgroup denominators" in measured
    assert "not independent human review" in measured
    assert "not a personal-vault product evaluation or a base-Qwen comparison" in measured
    assert f"[[{notes['result'].stem}]]" in decision
    assert report["decision"]["reason"] in decision
    assert "not a new user-approved Obsidian correction" in decision


def test_builder_refuses_nonempty_output_without_replacing_content(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    existing = vault / "private-note.md"
    existing.write_text("do not replace me", encoding="utf-8")

    with pytest.raises(FileExistsError):
        build_vault(vault)

    assert existing.read_text(encoding="utf-8") == "do not replace me"
    assert list(vault.iterdir()) == [existing]


def test_result_links_prior_strategy_with_exact_reviewable_quotes(tmp_path):
    vault = tmp_path / "vault"
    notes = build_vault(vault)
    snapshot = tmp_path / "snapshot"
    freeze_vault(vault, snapshot)
    _, docs = load_snapshot(snapshot)
    graph = build_graph(snapshot)
    paths = {node["source_path"]: doc_id for doc_id, node in graph["nodes"].items()}
    result_path = notes["result"].relative_to(vault).as_posix()
    prior_path = notes["prior"].relative_to(vault).as_posix()
    result_id, prior_id = paths[result_path], paths[prior_path]

    assert {"source": result_id, "target": prior_id, "type": "wikilink"} in graph["edges"]
    review = build_impact_review(snapshot, result_path)
    assert review["status"] == "possible_impact"
    assert review["prior"]["doc_id"] == prior_id
    assert review["prior"]["record_id"] == _metadata(notes["prior"])["record_id"]
    assert review["source"]["doc_id"] == result_id
    assert review["proposal"]["kind"] == "correction"
    assert review["proposal"]["supersedes"] == prior_id
    assert {item["doc_id"] for item in review["evidence"]} == {result_id, prior_id}
    for item in review["evidence"]:
        assert docs[item["doc_id"]]["text"][item["start"]:item["end"]] == item["quote"]
    prior_quote = next(item["quote"] for item in review["evidence"]
                       if item["doc_id"] == prior_id)
    for gate in ("semantic mean exceeds starting_sft",
                 "at least two more paired wins than losses",
                 "at least 28 of 40 semantic passes"):
        assert gate in prior_quote


def test_disposable_demo_starts_from_report_derived_notes(tmp_path):
    vault, state, notes = prepare_workspace(tmp_path)
    assert state.is_dir() and not state.is_relative_to(vault)
    assert notes["result"].read_bytes() == (
        DEFAULT_OUTPUT / notes["result"].name
    ).read_bytes()
    assert all(path.is_relative_to(vault) for path in notes.values())
