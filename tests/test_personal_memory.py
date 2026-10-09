"""Exercise the local product loop without a GPU, account, or model call."""

import json
from pathlib import Path

import pytest

from scripts.personal_memory import main
from src.product.memory import (
    approve_note,
    capture_note,
    freeze_vault,
    inspect_evidence,
    review_code_exec_transcript,
)
from src.research.agent import load_snapshot


def test_capture_review_approve_and_later_recall(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    original = capture_note(
        vault, title="First study attempt", kind="attempt", effective_date="2026-09-01",
        text="I thought spaced practice meant repeating the same problem immediately.",
    )
    original_bytes = original.read_bytes()
    first = tmp_path / "snapshot-1"
    freeze_vault(vault, first)

    review = inspect_evidence(first, "spaced practice repeating problem")
    assert review["retriever"] == "lexical_paragraph_baseline"
    assert review["status"] == "evidence_found"
    assert review["evidence"][0]["quote"]
    assert review["evidence"][0]["record"]["effective_date"] == "2026-09-01"
    old_doc_id = review["evidence"][0]["doc_id"]
    approved = approve_note(
        vault, first, review, title="Revised study lesson",
        text="I now plan to revisit a problem after a delay and check what I retained.",
        kind="correction", effective_date="2026-09-20", evidence_ids=["E1"],
        supersedes=old_doc_id,
    )
    assert original.read_bytes() == original_bytes
    assert "review_status: user_approved" in approved.read_text()
    assert f"supersedes_doc_id: {old_doc_id}" in approved.read_text()
    assert "exact_spans_verified_semantic_support_not_automated" in approved.read_text()

    second = tmp_path / "snapshot-2"
    freeze_vault(vault, second)
    recall = inspect_evidence(second, "revisit problem after a delay retained")
    assert recall["status"] == "evidence_found"
    assert any(item["source_path"] == approved.relative_to(vault).as_posix()
               for item in recall["evidence"])


def test_no_evidence_is_explicit_and_cannot_be_approved(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    capture_note(vault, title="One note", text="This concerns homework algebra.")
    snapshot = tmp_path / "snapshot"
    freeze_vault(vault, snapshot)
    review = inspect_evidence(snapshot, "xylophone quasars")
    assert review["status"] == "no_evidence"
    assert review["evidence"] == []
    assert "selected snapshot" in review["uncertainty"]
    with pytest.raises(ValueError, match="select one or more"):
        approve_note(vault, snapshot, review, title="Unsupported", text="A claim",
                     kind="lesson", evidence_ids=[])
    assert len(list(vault.rglob("*.md"))) == 1


def test_focus_plan_query_returns_the_revised_plan_body_without_homework(tmp_path):
    vault = Path(__file__).resolve().parents[1] / "data/product_memory/sample_vault"
    snapshot = tmp_path / "snapshot"
    freeze_vault(vault, snapshot)

    review = inspect_evidence(snapshot, "phone-drawer focus plan changes")
    assert review["status"] == "evidence_found"
    assert [item["source_path"] for item in review["evidence"]] == [
        "Learning/Life/2026-06-10-focus-plan.md",
        "Learning/Life/2026-07-04-focus-correction.md",
        "Learning/Life/2026-07-18-focus-decision.md",
    ]
    decision = review["evidence"][2]
    assert decision["evidence_id"] == "E3"
    assert decision["record"]["kind"] == "decision"
    assert "write the first concrete action on a card" in decision["quote"]
    assert "silence notifications for 30 minutes" in decision["quote"]
    _, documents = load_snapshot(snapshot)
    assert all(
        documents[item["doc_id"]]["text"][item["start"]:item["end"]] == item["quote"]
        for item in review["evidence"]
    )


def test_linked_demo_navigation_is_not_cited_as_factual_evidence(tmp_path):
    vault = Path(__file__).resolve().parents[1] / "data/product_memory/linked_demo_vault"
    snapshot = tmp_path / "snapshot"
    freeze_vault(vault, snapshot)

    review = inspect_evidence(snapshot, "focus plan June July measure next")
    correction = next(item for item in review["evidence"]
                      if item["source_path"].endswith("2026-07-04-focus-correction.md"))
    assert "I revisited the June phone-drawer idea" in correction["quote"]
    assert "## Linked history" not in correction["quote"]
    assert all(not item["quote"].lstrip().startswith("## Linked history")
               for item in review["evidence"])


def test_approval_rechecks_hash_and_quote_before_writing(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    capture_note(vault, title="Source", text="A fixed factual passage about the project.")
    snapshot = tmp_path / "snapshot"
    freeze_vault(vault, snapshot)
    review = inspect_evidence(snapshot, "factual passage")
    review["evidence"][0]["quote"] = "forged passage"
    with pytest.raises(ValueError, match="does not match"):
        approve_note(vault, snapshot, review, title="Forged", text="A claim",
                     kind="lesson", evidence_ids=["E1"])
    assert len(list(vault.rglob("*.md"))) == 1
    review["corpus_hash"] = "0" * 64
    with pytest.raises(ValueError, match="hashes differ"):
        approve_note(vault, snapshot, review, title="Forged", text="A claim",
                     kind="lesson", evidence_ids=["E1"])


def test_code_execution_review_preserves_abstention_without_fake_quote(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    capture_note(vault, title="One note", text="The answer is inside this note.")
    snapshot = tmp_path / "snapshot"
    freeze_vault(vault, snapshot)
    transcript = tmp_path / "run.json"
    transcript.write_text(json.dumps({
        "question_id": "q1", "question": "What is missing?", "status": "completed",
        "policy_name": "qwen_sft_policy", "predicted_answer": "Unanswerable",
        "predicted_evidence": [], "answer": "GOLD_SHOULD_NOT_LEAK",
    }))
    packet = review_code_exec_transcript(snapshot, transcript)
    assert packet["status"] == "no_evidence"
    assert packet["evidence"] == []
    assert "GOLD_SHOULD_NOT_LEAK" not in json.dumps(packet)


def test_cli_requires_explicit_approval_flag(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    capture_note(vault, title="Source", text="A useful project finding.")
    snapshot = tmp_path / "snapshot"
    freeze_vault(vault, snapshot)
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(inspect_evidence(snapshot, "project finding")))
    before = sorted(vault.rglob("*.md"))
    with pytest.raises(ValueError, match="--confirm"):
        main(["approve", "--vault", str(vault), "--snapshot", str(snapshot),
              "--review", str(review_path), "--title", "A lesson", "--kind", "lesson",
              "--evidence", "E1", "--text", "My edited lesson"])
    assert sorted(vault.rglob("*.md")) == before
