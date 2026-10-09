"""The working thread index must never mistake a proposal for accepted memory."""

import json

import pytest

from src.harness.staging import stage_drafts
from src.product.learning_thread import (
    approve_revision,
    create_thread,
    discard_pending_proposal,
    get_thread,
    list_threads,
    propose_revision,
    record_staged_draft,
    render_proposal_markdown,
)


def test_proposal_then_explicit_acceptance_preserves_prior_view(tmp_path):
    store = tmp_path / "state"
    initial = create_thread(
        store, question="Does multi-step search help here?",
        current_view="I expect Qwen to outperform lexical retrieval.",
        effective_date="2026-09-01", evidence_refs=["paper:p1#span-7"],
        recorded_at="2026-09-01T10:00:00-04:00",
    )
    thread_id = initial["thread_id"]
    original_revision = initial["current_approved_view"]
    proposal = propose_revision(
        store, thread_id, revised_view="Lexical retrieval is the better current baseline.",
        reason="A paired pilot favored lexical retrieval on supported answers.",
        effective_date="2026-09-28", evidence_refs=["snapshot:abc:passage-4"],
        observation_refs=["eval:paired-pilot-12"],
        assessments=[
            {"reference": "snapshot:abc:passage-4", "relation": "context"},
            {"reference": "eval:paired-pilot-12", "relation": "challenges"},
        ],
        proposed_at="2026-09-28T12:00:00Z",
    )

    before = get_thread(store, thread_id)
    assert before["current_approved_view"] == original_revision
    assert len(before["timeline"]) == 1
    assert before["pending_proposals"] == [proposal]
    assert before["vault_promotion_status"] == "not_promoted_by_thread_store"
    draft = render_proposal_markdown(store, thread_id, proposal["proposal_id"])
    assert "review_status: agent_authored_draft" in draft
    assert "snapshot:abc:passage-4" in draft
    assert "eval:paired-pilot-12" in draft
    assert "challenges: `eval:paired-pilot-12`" in draft
    assert "approved_by:" not in draft
    vault = tmp_path / "vault"
    vault.mkdir()
    staged = stage_drafts(vault, "learning-thread-fixture", {"thread-draft.md": draft})
    assert staged.run_dir == vault / "_inbox" / "learning-thread-fixture"
    assert staged.files["thread-draft.md"].read_text() == draft
    assert not (vault / "library").exists()

    with pytest.raises(ValueError, match="explicit user confirmation"):
        approve_revision(store, thread_id, proposal["proposal_id"], approved_by="user")
    assert get_thread(store, thread_id) == before

    accepted = approve_revision(
        store, thread_id, proposal["proposal_id"], approved_by="user",
        confirmed=True, approved_at="2026-09-29T09:00:00-04:00",
    )
    after = get_thread(store, thread_id)
    assert after["current_approved_view"] == accepted
    assert after["timeline"][0] == original_revision
    assert after["timeline"][1]["supersedes_revision_id"] == original_revision["revision_id"]
    assert after["timeline"][1]["observation_refs"] == ["eval:paired-pilot-12"]
    assert after["timeline"][1]["assessments"][1]["relation"] == "challenges"
    assert after["pending_proposals"] == []
    assert accepted["recorded_at"] == "2026-09-29T13:00:00+00:00"
    assert list(tmp_path.rglob("*.md")) == [staged.files["thread-draft.md"]]
    with pytest.raises(ValueError, match="not pending"):
        approve_revision(store, thread_id, proposal["proposal_id"],
                         approved_by="user", confirmed=True)


def test_stale_proposal_cannot_replace_newer_accepted_view(tmp_path):
    store = tmp_path / "state"
    thread_id = create_thread(
        store, question="Should I deploy this retrieval approach?",
        current_view="Maybe", effective_date="2026-09-01",
    )["thread_id"]
    first = propose_revision(
        store, thread_id, revised_view="Wait for more evidence", reason="Small sample",
        effective_date="2026-09-20",
    )
    stale = propose_revision(
        store, thread_id, revised_view="Deploy now", reason="Promising pilot",
        effective_date="2026-09-21",
    )
    approve_revision(store, thread_id, first["proposal_id"],
                     approved_by="user", confirmed=True)
    with pytest.raises(ValueError, match="stale"):
        approve_revision(store, thread_id, stale["proposal_id"],
                         approved_by="user", confirmed=True)
    with pytest.raises(ValueError, match="stale"):
        render_proposal_markdown(store, thread_id, stale["proposal_id"])
    assert get_thread(store, thread_id)["current_approved_view"]["view"] == (
        "Wait for more evidence"
    )
    assert get_thread(store, thread_id)["pending_proposals"] == []
    assert list_threads(store)[0]["pending_proposal_count"] == 0
    persisted = json.loads((store / f"{thread_id}.json").read_text())
    stale_record = next(item for item in persisted["proposals"]
                        if item["proposal_id"] == stale["proposal_id"])
    assert stale_record["status"] == "stale"
    assert stale_record["superseded_by_revision_id"] == persisted["revisions"][-1]["revision_id"]
    followup = propose_revision(
        store, thread_id, revised_view="Run a larger test next", reason="Open question",
        effective_date="2026-09-22",
    )
    assert get_thread(store, thread_id)["pending_proposals"] == [followup]


def test_rejects_bad_dates_refs_and_thread_paths(tmp_path):
    store = tmp_path / "state"
    with pytest.raises(ValueError, match="ISO date"):
        create_thread(store, question="Q", current_view="V", effective_date="tomorrow")
    with pytest.raises(ValueError, match="duplicate"):
        create_thread(store, question="Q", current_view="V", effective_date="2026-09-01",
                      evidence_refs=["same", "same"])
    with pytest.raises(ValueError, match="invalid thread_id"):
        get_thread(store, "../../vault/library")
    thread_id = create_thread(store, question="Q", current_view="V",
                              effective_date="2026-09-01")["thread_id"]
    with pytest.raises(ValueError, match="valid relation"):
        propose_revision(store, thread_id, revised_view="New", reason="Why",
                         effective_date="2026-09-02", evidence_refs=["source:1"],
                         assessments=[{"reference": "source:1", "relation": "proven"}])


def test_list_threads_scans_only_safe_thread_files(tmp_path):
    store = tmp_path / "state"
    later = create_thread(store, question="Z question", current_view="Z view",
                          effective_date="2026-09-01")
    earlier = create_thread(store, question="A question", current_view="A view",
                            effective_date="2026-09-02")
    propose_revision(store, earlier["thread_id"], revised_view="Maybe A",
                     reason="New reading", effective_date="2026-09-03")
    (store / "other.json").write_text("not a thread")
    (store / ("thread_" + "f" * 32 + ".json")).symlink_to(store / "other.json")

    summaries = list_threads(store)
    assert [row["thread_id"] for row in summaries] == [
        earlier["thread_id"], later["thread_id"],
    ]
    assert summaries[0]["pending_proposal_count"] == 1
    assert summaries[0]["revision_count"] == 1
    assert summaries[0]["current_approved_view"]["view"] == "A view"


def test_staged_hash_binding_and_failed_stage_rollback(tmp_path):
    store = tmp_path / "state"
    thread_id = create_thread(
        store, question="Will the approach work?", current_view="Unknown",
        effective_date="2026-09-01",
    )["thread_id"]
    proposal = propose_revision(
        store, thread_id, revised_view="Promising but unproven", reason="A paper suggests it",
        effective_date="2026-09-02", evidence_refs=["source:1"],
    )
    proposal_id = proposal["proposal_id"]
    assert proposal["staged_sha256"] is None
    with pytest.raises(ValueError, match="64 lowercase hex"):
        record_staged_draft(store, thread_id, proposal_id, "ABC")
    bound = record_staged_draft(store, thread_id, proposal_id, "a" * 64)
    assert bound["staged_sha256"] == "a" * 64
    assert get_thread(store, thread_id)["pending_proposals"][0]["staged_sha256"] == "a" * 64
    assert record_staged_draft(store, thread_id, proposal_id, "a" * 64) == bound
    with pytest.raises(ValueError, match="different staged draft"):
        record_staged_draft(store, thread_id, proposal_id, "b" * 64)

    discarded = discard_pending_proposal(store, thread_id, proposal_id)
    assert discarded["status"] == "discarded"
    assert get_thread(store, thread_id)["pending_proposals"] == []
    assert list_threads(store)[0]["pending_proposal_count"] == 0
    with pytest.raises(ValueError, match="not pending"):
        approve_revision(store, thread_id, proposal_id, approved_by="user", confirmed=True)
    with pytest.raises(ValueError, match="pending"):
        record_staged_draft(store, thread_id, proposal_id, "a" * 64)
    persisted = json.loads((store / f"{thread_id}.json").read_text())
    assert persisted["proposals"][0]["status"] == "discarded"
