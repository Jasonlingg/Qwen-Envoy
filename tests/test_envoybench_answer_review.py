"""The answer editor must preserve blinding and frozen review content."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from benchmarks.envoybench.answer_review import (
    finalize_answer_review,
    load_answer_review,
    save_answer_verdict,
)
from benchmarks.envoybench.score import (
    _review_content,
    prepare_review_bundle,
    reference_review_template,
    score_completed_review,
)
from src.eval.artifacts import configuration_hash, content_hash


def _prepared(tmp_path: Path) -> tuple[Path, dict, Path, dict]:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    paper = "Treatment increased recall by four points, but cost rose."
    (corpus / "paper_a.json").write_text(json.dumps({
        "doc_id": "paper_a", "title": "Synthetic paper", "text": paper,
        "sections": [{"section": "Results", "start": 0, "end": len(paper)}],
    }), encoding="utf-8")
    benchmark = {
        "schema_version": "research-benchmark-v1",
        "benchmark_id": "answer-review-test",
        "corpus_hash": content_hash(corpus),
        "reserved_doc_ids": ["paper_a"],
        "questions": [{
            "id": "q1", "split": "pilot_evaluation",
            "question": "What happened to recall and cost?",
            "answer": "Recall rose by four points and cost rose.",
            "expected_answerability": "sufficient",
            "required_doc_ids": ["paper_a"],
            "expected_citations": ["paper_a"],
            "minimum_distinct_sources": 1,
            "grader_notes": ["Check both outcomes."],
        }],
    }
    benchmark_hash = configuration_hash(benchmark)
    manifest = {
        "schema_version": "envoybench-run-v1", "status": "complete",
        "run_id": "synthetic-run", "comparison_id": "synthetic-pair",
        "split": "test_candidate", "split_status": "unreviewed",
        "benchmark_id": benchmark["benchmark_id"],
        "benchmark_hash": benchmark_hash,
        "corpus_hash": benchmark["corpus_hash"],
        "full_split": True, "subset_smoke": False,
        "split_question_count": 1, "question_ids": ["q1"],
        "models": [{"key": "base"}, {"key": "trained"}],
    }

    def result(model: str, answer: str) -> dict:
        return {
            "schema_version": "envoybench-run-v1",
            "run_id": manifest["run_id"],
            "comparison_id": manifest["comparison_id"],
            "split": manifest["split"],
            "split_status": manifest["split_status"],
            "question_id": "q1", "model_key": model,
            "question": benchmark["questions"][0]["question"],
            "benchmark_id": benchmark["benchmark_id"],
            "benchmark_hash": benchmark_hash,
            "corpus_hash": benchmark["corpus_hash"],
            "status": "submitted", "predicted_answer": answer,
            "predicted_citations": ["paper_a"],
            "predicted_evidence": [{
                "doc_id": "paper_a", "start": 0, "end": len(paper), "quote": paper,
            }],
            "trajectory": [], "duration_seconds": 0.1,
        }

    references = reference_review_template(benchmark)
    references["status"] = "complete"
    references["reviewer_id"] = "source-reviewer"
    references["reviewed_at"] = "2026-09-30"
    references["questions"][0]["decision"] = "reference_valid"
    review, key, _ = prepare_review_bundle(
        benchmark, manifest,
        [result("base", "It got cheaper."),
         result("trained", "Recall rose by four points, and cost rose.")],
        corpus, references,
    )
    review_dir = tmp_path / "review"
    review_dir.mkdir()
    review_path = review_dir / "review.json"
    review_path.write_text(json.dumps(review, indent=2) + "\n", encoding="utf-8")
    (review_dir / "blind-key.json").write_text(json.dumps(key), encoding="utf-8")
    return review_path, benchmark, corpus, key


def test_editor_never_reads_key_and_completed_sheet_scores(tmp_path, monkeypatch):
    review_path, benchmark, corpus, key = _prepared(tmp_path)
    original_read_bytes = Path.read_bytes

    def guard_read_bytes(path: Path) -> bytes:
        if path.name == "blind-key.json":
            raise AssertionError("blind key opened during answer review")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guard_read_bytes)
    initial = load_answer_review(review_path, benchmark, corpus)
    assert initial["decided_count"] == 0
    assert initial["total_count"] == 2
    assert initial["reviewer_kind"] is None
    assert all("model_key" not in row and "system" not in row for row in initial["rows"])
    with pytest.raises(ValueError, match="expected_sha256 revision is required"):
        save_answer_verdict(
            review_path, benchmark, corpus, initial["rows"][0]["blind_id"],
            "pass", "", "", "reviewer-1",
        )
    row_hashes = {
        row["blind_id"]: configuration_hash(_review_content(row))
        for row in initial["rows"]
    }

    first_id, second_id = [row["blind_id"] for row in initial["rows"]]
    first = save_answer_verdict(
        review_path, benchmark, corpus, first_id, "pass", "", "", "reviewer-1",
        expected_sha256=initial["revision"],
    )
    assert first["status"] == "incomplete"
    assert first["reviewer_kind"] is None
    assert first["decided_count"] == 1
    assert first["revision"] != initial["revision"]
    assert row_hashes == {
        row["blind_id"]: configuration_hash(_review_content(row))
        for row in first["rows"]
    }
    with pytest.raises(ValueError, match="changed since it was loaded"):
        save_answer_verdict(
            review_path, benchmark, corpus, second_id, "fail", "Wrong",
            "paper_a:0-4", "reviewer-1", expected_sha256=initial["revision"],
        )
    with pytest.raises(ValueError, match="every blind answer needs a verdict"):
        finalize_answer_review(
            review_path, benchmark, corpus, "reviewer-1",
            expected_sha256=first["revision"],
        )
    with pytest.raises(ValueError, match="reason and relevant source passage"):
        save_answer_verdict(
            review_path, benchmark, corpus, second_id, "fail", "", "",
            "reviewer-1", expected_sha256=first["revision"],
        )

    second = save_answer_verdict(
        review_path, benchmark, corpus, second_id, "fail",
        "The cost claim conflicts with the paper.", "paper_a:0-56", "reviewer-1",
        expected_sha256=first["revision"],
    )
    with pytest.raises(ValueError, match="reviewer_id differs"):
        finalize_answer_review(
            review_path, benchmark, corpus, "different-reviewer",
            expected_sha256=second["revision"],
        )
    complete = finalize_answer_review(
        review_path, benchmark, corpus, "reviewer-1",
        expected_sha256=second["revision"],
    )
    assert complete["status"] == "complete"
    assert complete["reviewer_kind"] == "human"
    assert complete["decided_count"] == complete["total_count"] == 2
    assert row_hashes == {
        row["blind_id"]: configuration_hash(_review_content(row))
        for row in complete["rows"]
    }
    assert score_completed_review(
        json.loads(review_path.read_text(encoding="utf-8")), key
    )["human"] is not None
    with pytest.raises(ValueError, match="locked"):
        save_answer_verdict(
            review_path, benchmark, corpus, first_id, "fail", "Wrong",
            "paper_a:0-56", "reviewer-1", expected_sha256=complete["revision"],
        )


def test_editor_rejects_changed_corpus_question_and_symlink(tmp_path):
    review_path, benchmark, corpus, _ = _prepared(tmp_path)
    initial = load_answer_review(review_path, benchmark, corpus)
    changed = copy.deepcopy(benchmark)
    changed["questions"][0]["question"] = "Changed question"
    with pytest.raises(ValueError, match="benchmark_hash mismatch"):
        load_answer_review(review_path, changed, corpus)

    paper_path = corpus / "paper_a.json"
    paper_path.write_text(paper_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="corpus hash differs"):
        load_answer_review(review_path, benchmark, corpus)
    paper_path.write_text(paper_path.read_text(encoding="utf-8")[:-1], encoding="utf-8")

    raw = json.loads(review_path.read_text(encoding="utf-8"))
    raw["rows"][0]["question"] = "Tampered"
    review_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from frozen benchmark"):
        load_answer_review(review_path, benchmark, corpus)

    review_path.unlink()
    review_path.symlink_to("blind-key.json")
    with pytest.raises(ValueError, match="regular review.json"):
        load_answer_review(review_path, benchmark, corpus)
    assert initial["revision"]


def test_keyed_scorer_catches_answer_tampering_after_locked_verdicts(tmp_path):
    review_path, benchmark, corpus, key = _prepared(tmp_path)
    sheet = load_answer_review(review_path, benchmark, corpus)
    for row in sheet["rows"]:
        sheet = save_answer_verdict(
            review_path, benchmark, corpus, row["blind_id"], "pass", "", "",
            "reviewer-1", expected_sha256=sheet["revision"],
        )
    raw = json.loads(review_path.read_text(encoding="utf-8"))
    raw["rows"][0]["answer"] = "Tampered anonymous answer"
    review_path.write_text(json.dumps(raw), encoding="utf-8")
    current = load_answer_review(review_path, benchmark, corpus)
    finalized = finalize_answer_review(
        review_path, benchmark, corpus, "reviewer-1",
        expected_sha256=current["revision"],
    )
    with pytest.raises(ValueError, match="content changed"):
        score_completed_review(finalized, key)
