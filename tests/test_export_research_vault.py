"""Obsidian export keeps evidence linked and leaves personal notes alone."""

import hashlib
import re

import pytest

from scripts.export_research_vault import ROOT, export


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
