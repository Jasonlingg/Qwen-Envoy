"""Offline publication checks for reviewed weekly paper notes and digest."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.weekly_digest import main
from src.product.graph import build_graph
from src.product.memory import freeze_vault
from src.product.weekly_digest import plan_weekly_digest, write_weekly_digest
from src.research.agent import load_snapshot
from src.research.papers import build_snapshot


@pytest.fixture
def inputs(tmp_path):
    ids = ["2609.09516v1", "2503.09517v2", "2503.09518v1"]

    def download(url):
        version = url.rsplit("/", 1)[-1]
        if "/html/" in url:
            return ("<article><h2>Methods</h2>" + "".join(
                f"<p>Fixed corpus experiment {number} for {version}.</p>"
                for number in range(6)
            ) + "</article>").encode()
        submitted, updated = {
            "2609.09516v1": ("2026/09/22", "2026/09/22"),
            "2503.09517v2": ("2025/03/12", "2026/09/24"),
            "2503.09518v1": ("2025/03/12", "2025/03/12"),
        }[version]
        return (f'<meta property="og:url" content="https://arxiv.org/abs/{version}">'
                f'<meta name="citation_title" content="Study of {version}">'
                f'<meta name="citation_abstract" content="We study search in {version}.">'
                f'<meta name="citation_date" content="{submitted}">'
                f'<meta name="citation_online_date" content="{updated}">').encode()

    snapshot = tmp_path / "snapshot"
    manifest = build_snapshot(ids, snapshot, downloader=download, delay=0)
    _, docs = load_snapshot(snapshot)
    selection = {
        "schema_version": "envoy-weekly-selection-v1",
        "week": "2026-W39",
        "topic": "Evidence tools",
        "corpus_hash": manifest["corpus_hash"],
        "review_status": "human_reviewed",
        "papers": [],
    }
    for item in manifest["papers"][:2]:
        doc = docs[item["doc_id"]]
        quote = doc["text"][:len("We study search")]
        selection["papers"].append({
            "doc_id": doc["doc_id"],
            "arxiv_id": item["arxiv_id"],
            "source_url": item["source_url"],
            "why_it_matters": "It suggests a retrieval baseline to compare with our worker.",
            "evidence": [{"start": 0, "end": len(quote), "quote": quote}],
            "limitations": ["This synthetic excerpt proves no downstream quality."],
        })
    selection_file = tmp_path / "selection.json"
    selection_file.write_text(json.dumps(selection), encoding="utf-8")
    vault = tmp_path / "vault"
    vault.mkdir()
    return snapshot, selection_file, selection, vault


def _save(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_preview_writes_nothing_then_confirm_creates_linked_notes(inputs, tmp_path, capsys):
    snapshot, selection_file, selection, vault = inputs
    args = ["--snapshot", str(snapshot), "--selection", str(selection_file),
            "--vault", str(vault)]

    assert main(args) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["status"] == "ready"
    assert preview["corpus_hash"] == selection["corpus_hash"]
    assert list(vault.rglob("*")) == []

    assert main([*args, "--confirm"]) == 0
    saved = json.loads(capsys.readouterr().out)
    assert saved["status"] == "written"
    assert len(saved["files"]) == 3
    digest = vault / "AI Research" / "Weekly" / "2026-W39.md"
    assert digest.is_file()
    assert "bounded, human-selected update" in digest.read_text()
    assert "New this week" in digest.read_text()
    assert "Revised this week" in digest.read_text()
    assert "Pinned version: 2026-09-24" in digest.read_text()
    assert "Submitted: 2025-03-12 · Updated: 2026-09-24" in digest.read_text()
    assert "semantic support" in next((vault / "AI Research" / "Papers").rglob("*.md")).read_text()

    vault_snapshot = tmp_path / "vault-snapshot"
    freeze_vault(vault, vault_snapshot)
    graph = build_graph(vault_snapshot)
    assert len(graph["nodes"]) == 3
    assert len([edge for edge in graph["edges"] if edge["type"] == "wikilink"]) == 4
    assert graph["unresolved"] == []

    original = digest.read_bytes()
    assert main([*args, "--confirm"]) == 1
    assert "already exists" in capsys.readouterr().err
    assert digest.read_bytes() == original


@pytest.mark.parametrize(("change", "message"), [
    (lambda s: s.update(corpus_hash="0" * 64), "corpus hashes differ"),
    (lambda s: s.update(review_status="unreviewed"), "review_status"),
    (lambda s: s.update(week="2026-W54"), "valid ISO week"),
    (lambda s: s["papers"].append(deepcopy(s["papers"][0])), "duplicate selected paper"),
    (lambda s: s["papers"][0].update(arxiv_id="2503.99999v1"), "source identity"),
    (lambda s: s["papers"][0].update(source_url="https://example.com/no"),
     "source identity"),
    (lambda s: s["papers"][0]["evidence"][0].update(quote="invented"),
     "invalid exact evidence"),
    (lambda s: s["papers"][0]["evidence"][0].update(start=True),
     "invalid exact evidence"),
    (lambda s: s["papers"][0]["evidence"][0].update(doc_id="another-paper"),
     "evidence source identity"),
    (lambda s: s["papers"][0].update(limitations=[]), "limitations"),
    (lambda s: s["papers"][0].update(role="context"), "must predate"),
    (lambda s: s["papers"][0].update(role="unreviewed"), "paper role"),
])
def test_invalid_selections_fail_before_any_write(inputs, change, message):
    snapshot, selection_file, selection, vault = inputs
    change(selection)
    _save(selection_file, selection)
    with pytest.raises(ValueError, match=message):
        plan_weekly_digest(snapshot, selection_file, vault)
    assert list(vault.rglob("*")) == []


def test_rejects_out_of_vault_output_root_and_symlink(inputs, tmp_path):
    snapshot, selection_file, _, vault = inputs
    for bad_root in ("../outside", "/tmp/outside", "AI Research/../../outside"):
        with pytest.raises(ValueError, match="output root"):
            plan_weekly_digest(snapshot, selection_file, vault, output_root=bad_root)
    (vault / "AI Research").symlink_to(tmp_path / "outside", target_is_directory=True)
    with pytest.raises(ValueError, match="escapes the selected vault"):
        plan_weekly_digest(snapshot, selection_file, vault)


def test_snapshot_tampering_and_manifest_identity_are_rejected(inputs):
    snapshot, selection_file, _, vault = inputs
    manifest_file = snapshot / "manifest.json"
    manifest = json.loads(manifest_file.read_text())
    manifest["papers"][0]["source_url"] = "https://example.com/substitute"
    _save(manifest_file, manifest)
    with pytest.raises(ValueError, match="snapshot source identity mismatch"):
        plan_weekly_digest(snapshot, selection_file, vault)
    manifest["papers"][0]["source_url"] = (
        f"https://arxiv.org/abs/{manifest['papers'][0]['arxiv_id']}"
    )
    _save(manifest_file, manifest)
    corpus_file = next((snapshot / "corpus").glob("*.json"))
    corpus_file.write_text(corpus_file.read_text() + " ")
    with pytest.raises(ValueError, match="hash mismatch"):
        plan_weekly_digest(snapshot, selection_file, vault)
    assert list(vault.rglob("*")) == []


def test_library_writer_requires_explicit_confirmation(inputs):
    snapshot, selection_file, _, vault = inputs
    plan = plan_weekly_digest(snapshot, selection_file, vault)
    with pytest.raises(ValueError, match="--confirm"):
        write_weekly_digest(plan)
    assert list(vault.rglob("*")) == []


def test_agent_authored_draft_can_preview_but_cannot_write(inputs, capsys):
    snapshot, selection_file, selection, vault = inputs
    selection["review_status"] = "agent_authored_draft"
    _save(selection_file, selection)
    plan = plan_weekly_digest(snapshot, selection_file, vault)
    assert plan.review_status == "agent_authored_draft"
    assert "Agent-authored draft for human review" in plan.files[-1][1]
    assert "human-selected update" not in plan.files[-1][1]
    with pytest.raises(ValueError, match="human_reviewed selection"):
        write_weekly_digest(plan, confirm=True)
    args = ["--snapshot", str(snapshot), "--selection", str(selection_file),
            "--vault", str(vault)]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "draft_preview_only"
    assert main([*args, "--confirm"]) == 1
    assert "human_reviewed selection" in capsys.readouterr().err
    assert list(vault.rglob("*")) == []


def test_show_content_previews_draft_markdown_without_writing(inputs, capsys):
    snapshot, selection_file, selection, vault = inputs
    selection["review_status"] = "agent_authored_draft"
    _save(selection_file, selection)
    args = ["--snapshot", str(snapshot), "--selection", str(selection_file),
            "--vault", str(vault), "--show-content"]
    assert main(args) == 0
    output = capsys.readouterr().out
    assert '"status": "draft_preview_only"' in output
    assert "===== AI Research/Weekly/2026-W39.md =====" in output
    assert "# AI research · 2026-W39" in output
    assert "Agent-authored draft for human review" in output
    assert "Submitted: 2025-03-12 · Updated: 2026-09-24" in output
    assert "```text\nWe study search\n```" in output
    assert list(vault.rglob("*")) == []


def test_digest_links_to_all_source_spans_without_repeating_quotes(inputs):
    snapshot, selection_file, selection, vault = inputs
    _, docs = load_snapshot(snapshot)
    first = selection["papers"][0]
    passage = docs[first["doc_id"]]["text"][20:45]
    first["evidence"].append({"start": 20, "end": 45, "quote": passage})
    first["limitations"].append("A second limitation needs its own line.")
    _save(selection_file, selection)

    plan = plan_weekly_digest(snapshot, selection_file, vault)
    paper_note, digest = plan.files[0][1], plan.files[-1][1]
    assert "**Offsets 20–45**" in paper_note
    assert f"```text\n{passage}\n```" in paper_note
    assert f"```text\n{passage}\n```" not in digest
    assert "#Reviewed evidence|2 exact source excerpts in the paper note" in digest
    assert "- A second limitation needs its own line." in digest


def test_analysis_scope_is_visible_and_bounded(inputs):
    snapshot, selection_file, selection, vault = inputs
    selection["analysis_note"] = "Selection was based on abstracts only."
    _save(selection_file, selection)
    plan = plan_weekly_digest(snapshot, selection_file, vault)
    assert "**Analysis scope:** Selection was based on abstracts only." in plan.files[0][1]
    assert "**Analysis scope:** Selection was based on abstracts only." in plan.files[-1][1]

    selection["analysis_note"] = "x" * 501
    _save(selection_file, selection)
    with pytest.raises(ValueError, match="analysis_note"):
        plan_weekly_digest(snapshot, selection_file, vault)


def test_existing_week_blocks_paper_note_creation(inputs):
    snapshot, selection_file, _, vault = inputs
    weekly = vault / "AI Research" / "Weekly" / "2026-W39.md"
    weekly.parent.mkdir(parents=True)
    weekly.write_text("Existing approved digest\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        plan_weekly_digest(snapshot, selection_file, vault)
    assert weekly.read_text() == "Existing approved digest\n"
    assert not (vault / "AI Research" / "Papers").exists()


def test_older_paper_requires_explicit_context_and_in_week_anchor(inputs, tmp_path):
    snapshot, selection_file, selection, vault = inputs
    _, docs = load_snapshot(snapshot)
    old = next(item for item in json.loads((snapshot / "manifest.json").read_text())["papers"]
               if item["arxiv_id"] == "2503.09518v1")
    doc = docs[old["doc_id"]]
    old_row = {
        "doc_id": old["doc_id"], "arxiv_id": old["arxiv_id"],
        "source_url": old["source_url"],
        "why_it_matters": "Background for this week's comparison.",
        "evidence": [{"start": 0, "end": 15, "quote": doc["text"][:15]}],
        "limitations": ["Older source; not a new result this week."],
    }
    selection["papers"].append(old_row)
    _save(selection_file, selection)
    with pytest.raises(ValueError, match="neither new nor revised"):
        plan_weekly_digest(snapshot, selection_file, vault)

    old_row.update(role="context")
    _save(selection_file, selection)
    with pytest.raises(ValueError, match="context_reason"):
        plan_weekly_digest(snapshot, selection_file, vault)

    old_row["context_reason"] = "It defines the earlier approach that the revised paper tests."
    _save(selection_file, selection)
    plan = plan_weekly_digest(snapshot, selection_file, vault)
    assert len(plan.files) == 4
    write_weekly_digest(plan, confirm=True)
    digest = vault / "AI Research" / "Weekly" / "2026-W39.md"
    assert "Background context; not counted as this week's new research" in digest.read_text()
    assert "Why included as context" in digest.read_text()
    assert "Submitted: 2025-03-12 · Updated: 2025-03-12" in digest.read_text()

    another_vault = tmp_path / "context-only-vault"
    another_vault.mkdir()
    selection["papers"] = [old_row]
    _save(selection_file, selection)
    with pytest.raises(ValueError, match="at least one new or revised"):
        plan_weekly_digest(snapshot, selection_file, another_vault)
    assert list(another_vault.rglob("*")) == []

    selection["papers"] = [{**selection["papers"][0],
                            "doc_id": "arxiv_2609_09516v1",
                            "arxiv_id": "2609.09516v1",
                            "source_url": "https://arxiv.org/abs/2609.09516v1",
                            "role": "context", "context_reason": "Still in week"}]
    _save(selection_file, selection)
    with pytest.raises(ValueError, match="must predate"):
        plan_weekly_digest(snapshot, selection_file, another_vault)


def test_later_revision_cannot_be_backdated_to_submission_week(inputs):
    snapshot, selection_file, selection, vault = inputs
    selection["week"] = "2025-W11"
    selection["papers"] = [selection["papers"][1]]  # submitted in W11, pinned v2 updated later
    _save(selection_file, selection)
    with pytest.raises(ValueError, match="pinned version is neither new nor revised"):
        plan_weekly_digest(snapshot, selection_file, vault)
    assert list(vault.rglob("*")) == []


def test_related_vault_note_links_and_next_measurement_are_visible_without_editing_source(
    inputs, tmp_path,
):
    snapshot, selection_file, selection, vault = inputs
    existing = vault / "Learning" / "Research" / "Worker Decision.md"
    existing.parent.mkdir(parents=True)
    existing.write_text("# Worker decision\n\nI will compare retrieval before training.\n",
                        encoding="utf-8")
    before = existing.read_bytes()
    selection["papers"][0]["related_notes"] = [{
        "path": "Learning/Research/Worker Decision.md",
        "reason": "This decision sets the project baseline before the new paper.",
    }]
    selection["next_measurement"] = (
        "Compare BM25, base Qwen, and v5 on the same held-out question IDs using a "
        "supported-answer rubric; record latency and cost."
    )
    _save(selection_file, selection)

    plan = plan_weekly_digest(snapshot, selection_file, vault)
    paper_note = plan.files[0][1]
    digest_note = plan.files[-1][1]
    link = "[[Learning/Research/Worker Decision.md|Worker decision]]"
    assert link in paper_note and link in digest_note
    assert "sets the project baseline" in paper_note
    assert "sets the project baseline" in digest_note
    assert "## Next measurement to consider" in digest_note
    assert "Proposal, not a finding: Compare BM25, base Qwen, and v5" in digest_note
    write_weekly_digest(plan, confirm=True)
    assert existing.read_bytes() == before

    vault_snapshot = tmp_path / "with-related-snapshot"
    freeze_vault(vault, vault_snapshot)
    graph = build_graph(vault_snapshot)
    by_path = {node["source_path"]: doc_id for doc_id, node in graph["nodes"].items()}
    target = by_path["Learning/Research/Worker Decision.md"]
    incoming = [edge for edge in graph["edges"]
                if edge["target"] == target and edge["type"] == "wikilink"]
    assert {graph["nodes"][edge["source"]]["kind"] for edge in incoming} == {
        "paper", "weekly_digest"}
    assert graph["unresolved"] == []


def test_related_note_rejects_unsafe_missing_duplicate_and_unbounded_paths(inputs, tmp_path):
    snapshot, selection_file, selection, vault = inputs
    note = vault / "Learning" / "Decision.md"
    note.parent.mkdir(parents=True)
    note.write_text("# Decision\n", encoding="utf-8")
    outside = tmp_path / "outside.md"
    outside.write_text("# Outside\n", encoding="utf-8")
    (vault / "Learning" / "Link.md").symlink_to(outside)
    (vault / "Linked Folder").symlink_to(outside.parent, target_is_directory=True)
    before = note.read_bytes()

    def assert_rejected(related, message):
        selection["papers"][0]["related_notes"] = related
        _save(selection_file, selection)
        with pytest.raises(ValueError, match=message):
            plan_weekly_digest(snapshot, selection_file, vault)
        assert note.read_bytes() == before
        assert not (vault / "AI Research").exists()

    reason = "A relevant prior decision."
    for path in ("../outside.md", "/tmp/outside.md", "Learning/../Decision.md",
                 "Learning/Decision.txt", "Learning/Decision.md#Heading"):
        assert_rejected([{"path": path, "reason": reason}], "safe vault-relative")
    assert_rejected([{"path": "Learning/Missing.md", "reason": reason}], "does not exist")
    assert_rejected([{"path": "Learning/Link.md", "reason": reason}], "symlink")
    assert_rejected([{"path": "Linked Folder/outside.md", "reason": reason}], "symlink")
    valid = {"path": "Learning/Decision.md", "reason": reason}
    assert_rejected([valid, valid], "duplicate related note")
    assert_rejected([valid] * 6, "at most five")
    assert_rejected([{"path": valid["path"], "reason": "x" * 241}],
                    "related note reason")
    assert_rejected([{"path": "x" * 241 + ".md", "reason": reason}],
                    "related note path")


def test_next_measurement_is_optional_but_bounded(inputs):
    snapshot, selection_file, selection, vault = inputs
    assert "Next measurement" not in plan_weekly_digest(
        snapshot, selection_file, vault,
    ).files[-1][1]
    selection["next_measurement"] = "x" * 601
    _save(selection_file, selection)
    with pytest.raises(ValueError, match="next_measurement"):
        plan_weekly_digest(snapshot, selection_file, vault)
    assert list(vault.rglob("*")) == []
