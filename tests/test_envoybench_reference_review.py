"""Studio reference review must preserve frozen sources and require human action."""

from __future__ import annotations

import json

import pytest

from benchmarks.envoybench.reference_review import (
    finalize_reference_review,
    load_reference_review,
    save_reference_decision,
)
from benchmarks.envoybench.score import _validated_references, reference_review_template
from src.eval.artifacts import content_hash


def _fixtures(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "paper.json").write_text(json.dumps({"doc_id": "paper", "text": "Evidence."}))
    benchmark = {
        "schema_version": "research-benchmark-v1",
        "benchmark_id": "studio-test",
        "corpus_hash": content_hash(corpus),
        "reserved_doc_ids": ["paper"],
        "questions": [
            {
                "id": question_id,
                "split": "pilot_evaluation",
                "question": f"Question {question_id}?",
                "expected_answerability": "sufficient",
                "required_doc_ids": ["paper"],
                "minimum_distinct_sources": 1,
                "grader_notes": [],
                "answer": "Evidence.",
                "gold_evidence": [{"doc_id": "paper", "start": 0, "end": 9}],
            }
            for question_id in ("q1", "q2")
        ],
    }
    return corpus, benchmark, tmp_path / "new" / "review.json"


def test_review_requires_individual_decisions_and_explicit_finalization(tmp_path):
    corpus, benchmark, path = _fixtures(tmp_path)
    initial = load_reference_review(path, benchmark, corpus)
    assert not path.exists()
    assert initial["decided_count"] == 0
    assert initial["total_count"] == 2
    assert initial["review"] == reference_review_template(benchmark)

    first = save_reference_decision(
        path, benchmark, corpus, "q1", "reference_valid", "", "human-1",
        expected_sha256=initial["sha256"],
    )
    assert path.exists()
    assert first["decided_count"] == 1
    assert first["review"]["status"] == "incomplete"
    assert first["review"]["reviewed_at"] == ""
    assert first["review"]["questions"][1]["decision"] is None
    with pytest.raises(ValueError, match="q2 needs reference_valid or unscorable"):
        finalize_reference_review(path, benchmark, corpus, "human-1")

    second = save_reference_decision(
        path, benchmark, corpus, "q2", "unscorable", "Paper excerpt is ambiguous.",
        "human-1", expected_sha256=first["sha256"],
    )
    assert second["review"]["status"] == "incomplete"
    completed = finalize_reference_review(
        path, benchmark, corpus, "human-1", expected_sha256=second["sha256"]
    )
    assert completed["review"]["status"] == "complete"
    assert completed["review"]["reviewed_at"]
    assert _validated_references(benchmark, completed["review"]) == (
        {"q1"}, [{"question_id": "q2", "reason": "Paper excerpt is ambiguous."}]
    )
    with pytest.raises(ValueError, match="locked"):
        save_reference_decision(path, benchmark, corpus, "q1", "unscorable", "Changed", "human-1")


def test_review_rejects_stale_edits_and_changed_sources(tmp_path):
    corpus, benchmark, path = _fixtures(tmp_path)
    initial = load_reference_review(path, benchmark, corpus)
    first = save_reference_decision(
        path, benchmark, corpus, "q1", "reference_valid", "", "human-1",
        expected_sha256=initial["sha256"],
    )
    before = path.read_bytes()
    with pytest.raises(ValueError, match="changed since it was loaded"):
        save_reference_decision(
            path, benchmark, corpus, "q2", "reference_valid", "", "human-1",
            expected_sha256=initial["sha256"],
        )
    with pytest.raises(ValueError, match="existing reviewer"):
        save_reference_decision(path, benchmark, corpus, "q2", "reference_valid", "", "human-2")
    with pytest.raises(ValueError, match="unscorable decisions need a reason"):
        save_reference_decision(path, benchmark, corpus, "q2", "unscorable", "  ", "human-1")
    assert path.read_bytes() == before
    assert load_reference_review(path, benchmark, corpus)["sha256"] == first["sha256"]

    tampered = json.loads(path.read_text())
    tampered["questions"][0]["reference_answer"] = "Invented answer"
    path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="differs from frozen benchmark"):
        load_reference_review(path, benchmark, corpus)

    path.write_bytes(before)
    (corpus / "paper.json").write_text(json.dumps({"doc_id": "paper", "text": "Changed."}))
    with pytest.raises(ValueError, match="corpus hash differs"):
        load_reference_review(path, benchmark, corpus)


def test_failed_atomic_replace_keeps_previous_review(tmp_path, monkeypatch):
    corpus, benchmark, path = _fixtures(tmp_path)
    first = save_reference_decision(path, benchmark, corpus, "q1", "reference_valid", "", "human-1")
    before = path.read_bytes()

    def fail_replace(*_args):
        raise OSError("disk error")

    monkeypatch.setattr("benchmarks.envoybench.reference_review.os.replace", fail_replace)
    with pytest.raises(OSError, match="disk error"):
        save_reference_decision(
            path, benchmark, corpus, "q2", "reference_valid", "", "human-1",
            expected_sha256=first["sha256"],
        )
    assert path.read_bytes() == before
    assert list(path.parent.glob("*.tmp")) == []
