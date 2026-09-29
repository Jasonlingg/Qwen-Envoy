"""The vault staging boundary must publish complete batches without path escape."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
from pathlib import Path

import pytest

from src.harness import staging


def _draft(body: str = "# Draft\n") -> str:
    return f"---\nreview_status: agent_authored_draft\n---\n{body}"


def _stage_in_process(vault: str, result_queue: multiprocessing.Queue) -> None:
    try:
        staging.stage_drafts(Path(vault), "same-run", {"Papers/a.md": _draft()})
    except staging.DuplicateRunError:
        result_queue.put("duplicate")
    else:
        result_queue.put("staged")


def test_stages_complete_batch_and_hash_manifest_inside_inbox(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    library = vault / "library"
    library.mkdir()
    existing_note = library / "reviewed.md"
    existing_note.write_text("reviewed", encoding="utf-8")

    paper = _draft("# Paper\n")
    result = staging.stage_drafts(
        vault,
        "run-001",
        {"Papers/Paper One.md": paper, "Weekly/Week 39.md": _draft("# Week\n")},
    )

    assert result.run_dir == vault.resolve() / "_inbox" / "run-001"
    assert set(result.files) == {"Papers/Paper One.md", "Weekly/Week 39.md"}
    assert all(path.is_relative_to(result.run_dir) for path in result.files.values())
    assert result.files["Papers/Paper One.md"].read_text(encoding="utf-8") == paper
    assert existing_note.read_text(encoding="utf-8") == "reviewed"
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "staged-drafts-v1"
    assert manifest["run_id"] == "run-001"
    assert manifest["files"]["Papers/Paper One.md"] == {
        "sha256": hashlib.sha256(paper.encode()).hexdigest(),
        "bytes": len(paper.encode()),
    }
    assert (
        result.hashes["Papers/Paper One.md"] == manifest["files"]["Papers/Paper One.md"]["sha256"]
    )
    assert sorted(path.name for path in (vault / "_inbox").iterdir()) == [
        ".stage.lock",
        "run-001",
    ]


@pytest.mark.parametrize(
    ("run_id", "name"),
    [
        ("../library", "Papers/a.md"),
        ("bad/id", "Papers/a.md"),
        (".hidden", "Papers/a.md"),
        ("valid", "../library/a.md"),
        ("valid", "/library/a.md"),
        ("valid", "Papers/../../library/a.md"),
        ("valid", "Papers//a.md"),
        ("valid", "Papers/./a.md"),
        ("valid", "Papers\\a.md"),
        ("valid", "Papers/a.txt"),
        ("valid", "manifest.json/a.md"),
    ],
)
def test_rejects_unsafe_names_without_creating_inbox(
    tmp_path: Path, run_id: str, name: str
) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    with pytest.raises(staging.StagingError):
        staging.stage_drafts(vault, run_id, {name: _draft()})
    assert not (vault / "_inbox").exists()


def test_rejects_case_insensitive_directory_collision(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    with pytest.raises(staging.StagingError, match="collide"):
        staging.stage_drafts(vault, "run-001", {"Papers/a.md": _draft(), "papers/b.md": _draft()})
    assert not (vault / "_inbox").exists()


@pytest.mark.parametrize(
    "content",
    [
        "# No frontmatter\n",
        "---\nreview_status: agent_authored_draft\n",
        "---\nreview_status: human_reviewed\n---\n# False approval\n",
        "---\nreview_status: agent_authored_draft\nhuman_reviewed: false\n---\n",
        "---\nreview_status: agent_authored_draft\napproved: true\n---\n",
        "---\nreview_status: agent_authored_draft\napproval_status: user-approved\n---\n",
        "---\nreview_status: agent_authored_draft\nreview_status: agent_authored_draft\n---\n",
        "---\nreview_status: [agent_authored_draft\n---\n",
        "---\nother_status: agent_authored_draft\n---\n",
    ],
)
def test_rejects_missing_or_fake_draft_review_status(tmp_path: Path, content: str) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    with pytest.raises(staging.StagingError):
        staging.stage_drafts(vault, "run-001", {"Papers/a.md": content})
    assert not (vault / "_inbox").exists()


def test_rejects_nonexistent_vault_and_inbox_symlink(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(staging.StagingError, match="existing real directory"):
        staging.stage_drafts(missing, "run-001", {"a.md": _draft()})
    assert not missing.exists()

    vault = tmp_path / "vault"
    outside = tmp_path / "outside"
    vault.mkdir()
    outside.mkdir()
    (vault / "_inbox").symlink_to(outside, target_is_directory=True)
    with pytest.raises(staging.StagingError, match="unsafe directory"):
        staging.stage_drafts(vault, "run-001", {"a.md": _draft()})
    assert list(outside.iterdir()) == []


def test_rejects_symlinked_vault_and_existing_run_target(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(vault, target_is_directory=True)
    with pytest.raises(staging.StagingError):
        staging.stage_drafts(alias, "run-001", {"a.md": _draft()})

    outside = tmp_path / "outside"
    outside.mkdir()
    inbox = vault / "_inbox"
    inbox.mkdir()
    (inbox / "run-001").symlink_to(outside, target_is_directory=True)
    with pytest.raises(staging.DuplicateRunError):
        staging.stage_drafts(vault, "run-001", {"a.md": _draft()})
    assert list(outside.iterdir()) == []


def test_failed_write_removes_unpublished_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    original = staging._write_bytes_at
    calls = 0

    def fail_second_write(parent_fd: int, name: str, data: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("disk full")
        original(parent_fd, name, data)

    monkeypatch.setattr(staging, "_write_bytes_at", fail_second_write)
    with pytest.raises(OSError, match="disk full"):
        staging.stage_drafts(vault, "run-001", {"a.md": _draft("a"), "b.md": _draft("b")})
    assert not (vault / "_inbox" / "run-001").exists()
    assert sorted(path.name for path in (vault / "_inbox").iterdir()) == [".stage.lock"]


def test_atomic_publish_does_not_replace_externally_created_empty_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    original = staging._publish_no_replace

    def insert_destination_before_publish(inbox_fd: int, temp_name: str, run_id: str) -> None:
        staging.os.mkdir(run_id, dir_fd=inbox_fd)
        original(inbox_fd, temp_name, run_id)

    monkeypatch.setattr(staging, "_publish_no_replace", insert_destination_before_publish)
    with pytest.raises(staging.DuplicateRunError):
        staging.stage_drafts(vault, "run-001", {"a.md": _draft()})
    inbox = vault / "_inbox"
    assert (inbox / "run-001").is_dir()
    assert list((inbox / "run-001").iterdir()) == []
    assert sorted(path.name for path in inbox.iterdir()) == [".stage.lock", "run-001"]


def test_post_publish_fsync_failure_reports_visible_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    original_publish = staging._publish_no_replace
    original_fsync = staging.os.fsync
    published = False

    def publish_then_mark(inbox_fd: int, temp_name: str, run_id: str) -> None:
        nonlocal published
        original_publish(inbox_fd, temp_name, run_id)
        published = True

    def fail_inbox_fsync(fd: int) -> None:
        if published:
            raise OSError("simulated fsync failure")
        original_fsync(fd)

    monkeypatch.setattr(staging, "_publish_no_replace", publish_then_mark)
    monkeypatch.setattr(staging.os, "fsync", fail_inbox_fsync)
    with pytest.raises(staging.PublishedButUnconfirmedError) as raised:
        staging.stage_drafts(vault, "run-001", {"a.md": _draft()})
    run_dir = vault.resolve() / "_inbox" / "run-001"
    assert raised.value.run_dir == run_dir
    assert raised.value.manifest_path == run_dir / "manifest.json"
    assert (run_dir / "a.md").read_text(encoding="utf-8") == _draft()
    assert (run_dir / "manifest.json").is_file()
    assert sorted(path.name for path in (vault / "_inbox").iterdir()) == [
        ".stage.lock",
        "run-001",
    ]


def test_duplicate_run_is_rejected_across_processes(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    context = multiprocessing.get_context("fork")
    queue = context.Queue()
    processes = [
        context.Process(target=_stage_in_process, args=(str(vault), queue)) for _ in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0
    assert sorted([queue.get(timeout=2), queue.get(timeout=2)]) == ["duplicate", "staged"]
    assert (vault / "_inbox" / "same-run" / "Papers" / "a.md").read_text() == _draft()
