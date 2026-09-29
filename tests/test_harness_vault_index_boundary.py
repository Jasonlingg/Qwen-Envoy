"""Staged drafts must never become source material through a generic vault import."""

from __future__ import annotations

import pytest

from src.research.agent import load_snapshot
from src.research.vault import import_vault


def test_generic_import_excludes_inbox_and_explicit_inbox_import_fails(tmp_path):
    vault = tmp_path / "vault"
    library = vault / "library"
    library.mkdir(parents=True)
    (library / "reviewed.md").write_text("# Reviewed\n\nA source note.\n", encoding="utf-8")
    inbox = vault / "_inbox" / "run-1"
    inbox.mkdir(parents=True)
    (inbox / "draft.md").write_text("# Draft\n\nUnreviewed text.\n", encoding="utf-8")

    snapshot = tmp_path / "snapshot"
    import_vault(vault, ".", snapshot)
    _, docs = load_snapshot(snapshot)
    assert {doc["metadata"]["source_path"] for doc in docs.values()} == {"library/reviewed.md"}

    with pytest.raises(ValueError, match="_inbox"):
        import_vault(vault, "_inbox/run-1", tmp_path / "draft-snapshot")


def test_case_variant_inbox_is_excluded_even_on_case_sensitive_filesystems(tmp_path):
    vault = tmp_path / "vault"
    library = vault / "library"
    library.mkdir(parents=True)
    (library / "reviewed.md").write_text("# Reviewed\n", encoding="utf-8")
    alias = vault / "_INBOX" / "run-1"
    alias.mkdir(parents=True)
    (alias / "draft.md").write_text("# Draft\n", encoding="utf-8")

    snapshot = tmp_path / "snapshot"
    import_vault(vault, ".", snapshot)
    _, docs = load_snapshot(snapshot)
    assert {doc["metadata"]["source_path"] for doc in docs.values()} == {"library/reviewed.md"}
    with pytest.raises(ValueError, match="_inbox"):
        import_vault(vault, "_INBOX/run-1", tmp_path / "other-snapshot")
