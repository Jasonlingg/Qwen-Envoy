"""Obsidian export keeps evidence linked and leaves personal notes alone."""

import hashlib
import json
import re

import pytest

from scripts.export_research_vault import ROOT, export
from src.product.ml_experiment_store import ExperimentStore


def test_export_creates_real_linked_vault_and_refresh_preserves_user_note(tmp_path):
    vault = tmp_path / "vault"
    counts = export(vault, [])
    assert counts == {"atlas": 11, "experiments": 0, "questions": 0, "decisions": 0}
    home = (vault / "Home.md").read_text()
    assert "[[Attachments/README|Qwen Envoy README]]" in home
    assert (vault / "Attachments" / "README.md").read_bytes() == (ROOT / "README.md").read_bytes()
    assert "[[Atlas/reader-rl-pilot|reader-rl-pilot]]" in home
    atlas_note = (vault / "Atlas" / "reader-rl-pilot.md").read_text()
    source_id = re.search(r"\[\[Sources/([0-9a-f]{16})\|Initial pilot\]\]", atlas_note)
    assert source_id
    source_note = (vault / "Sources" / f"{source_id.group(1)}.md").read_text()
    original = ROOT / "docs" / "READER_RL_PILOT_20261025.md"
    attachment = vault / "Attachments" / "docs" / original.name
    assert attachment.read_bytes() == original.read_bytes()
    assert hashlib.sha256(original.read_bytes()).hexdigest() in source_note

    user_note = vault / "My Notes" / "idea.md"
    user_note.parent.mkdir()
    user_note.write_text("My private thought\n")
    assert export(vault, [], refresh=True) == counts
    assert user_note.read_text() == "My private thought\n"
    assert (vault / "Attachments" / "README.md").read_bytes() == (ROOT / "README.md").read_bytes()


def test_export_refuses_unmarked_existing_directory(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "Home.md").write_text("Existing user note\n")
    with pytest.raises(ValueError, match="marker"):
        export(vault, [], refresh=True)
    assert (vault / "Home.md").read_text() == "Existing user note\n"


def test_refresh_updates_saved_records_and_receipt_without_changing_store(tmp_path):
    root, vault = tmp_path / "records", tmp_path / "vault"
    store = ExperimentStore(root)
    saved = store.save_brief({"question": "Can we reduce false refusals?"})
    export(vault, [root])
    paper = vault / "Attachments/docs/READER_RL_PILOT_20261025.md"
    previous_mtime = paper.stat().st_mtime_ns
    store.save_brief({"question": "Can better evidence reduce false refusals?"},
                     saved["experiment_id"], expected_revision=1)
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    export(vault, [root], refresh=True)
    assert paper.stat().st_mtime_ns == previous_mtime
    assert before == {p.name: p.read_bytes() for p in root.iterdir()}
    assert "better evidence" in (vault / "Experiments" / f"{saved['experiment_id']}.md").read_text()
    receipt = json.loads((vault / ".envoy-export.json").read_text())
    assert receipt["repository"] == str(ROOT)
    assert receipt["refreshed_at"]
    assert receipt["counts"]["experiments"] == 1
    assert receipt["atlas_sha256"] == hashlib.sha256(
        (ROOT / "docs/EXPERIMENT_ATLAS.md").read_bytes()).hexdigest()


def test_refresh_rejects_edits_in_generated_notes(tmp_path):
    vault = tmp_path / "vault"
    export(vault, [])
    home = vault / "Home.md"
    home.write_text("My annotation\n")
    with pytest.raises(ValueError, match="edited or added"):
        export(vault, [], refresh=True)
    assert home.read_text() == "My annotation\n"


def test_bad_source_leaves_previous_export_intact(tmp_path, monkeypatch):
    import scripts.export_research_vault as module

    vault = tmp_path / "vault"
    export(vault, [])
    before = (vault / "Home.md").read_bytes()

    def broken_source(*args):
        raise ValueError("Invalid source record")

    monkeypatch.setattr(module, "export_store", broken_source)
    with pytest.raises(ValueError, match="Invalid source"):
        export(vault, [tmp_path / "records"], refresh=True)
    assert (vault / "Home.md").read_bytes() == before


def test_legacy_refresh_backs_up_generated_notes_and_preserves_obsidian_settings(tmp_path):
    vault = tmp_path / "vault"
    export(vault, [])
    (vault / ".envoy-export.json").unlink()
    (vault / "Home.md").write_text("Legacy note with annotations\n")
    config = vault / ".obsidian" / "app.json"
    config.parent.mkdir()
    config.write_text("{}")
    export(vault, [], refresh=True)
    backups = list((vault / ".envoy-backups").glob("*/Home.md"))
    assert len(backups) == 1
    assert backups[0].read_text() == "Legacy note with annotations\n"
    assert config.read_text() == "{}"


def test_refresh_rejects_symlink_to_personal_file(tmp_path):
    vault = tmp_path / "vault"
    export(vault, [])
    personal = tmp_path / "personal.md"
    personal.write_text("Keep me")
    (vault / "Home.md").unlink()
    (vault / "Home.md").symlink_to(personal)
    with pytest.raises(ValueError, match="symlink"):
        export(vault, [], refresh=True)
    assert personal.read_text() == "Keep me"
