"""Small CPU checks for the authored personal-memory development fixture."""

from __future__ import annotations

import json

import pytest

from scripts.personal_memory_eval import QUESTIONS, _step_diagnostics, baseline, diagnose, freeze


def test_sample_freeze_has_locked_sources_and_anchors(tmp_path):
    snapshot = tmp_path / "snapshot"
    result = freeze(snapshot)
    assert result["documents"] == 9
    assert result["questions"] == 10
    assert (
        result["corpus_hash"] == "a5470628a40fe6114271f3b2923ba025967bdc2552157543d2a7224440e43c47"
    )
    assert "human_review_pending" in result["status"]
    assert (snapshot / "source-check-v2.md").is_file()
    assert freeze(snapshot)["corpus_hash"] == result["corpus_hash"]


def test_cpu_baseline_has_bounded_exact_spans_and_no_generated_answer(tmp_path):
    snapshot = tmp_path / "snapshot"
    freeze(snapshot)
    output = tmp_path / "bm25.json"
    baseline(snapshot, output)
    rows = json.loads(output.read_text())
    assert len(rows) == 10
    for row in rows:
        assert row["status"] == "retrieval_only"
        assert row["predicted_answer"] == ""
        assert len(row["predicted_evidence"]) <= 5
        for span in row["predicted_evidence"]:
            doc = json.loads((snapshot / "corpus" / f"{span['doc_id']}.json").read_text())
            assert 0 <= span["start"] < span["end"] <= len(doc["text"])
    manifest = json.loads(output.with_suffix(".manifest.json").read_text())
    assert manifest["retrieval"]["uses_reference_answers_or_source_anchors"] is False


def test_diagnostics_separate_tool_behavior_from_semantic_review(tmp_path):
    snapshot = tmp_path / "snapshot"
    freeze(snapshot)
    baseline_path = tmp_path / "bm25.json"
    baseline(snapshot, baseline_path)
    output = tmp_path / "diagnostics.json"
    result = diagnose(
        questions=QUESTIONS,
        corpus=snapshot / "corpus",
        runs=[baseline_path],
        output=output,
    )
    assert result["semantic_support"] == "pending"
    report = json.loads(output.read_text())
    summary = report["systems"]["bm25_passage_baseline"]["summary"]
    assert summary["submitted_answer_count"] == 0
    assert summary["document_tool_action_steps"] == 0
    assert summary["required_note_recall"] == 1.0
    assert "pending independent human review" in report["semantic_support"]


def test_diagnostics_reject_stale_run_manifest(tmp_path):
    snapshot = tmp_path / "snapshot"
    freeze(snapshot)
    baseline_path = tmp_path / "bm25.json"
    baseline(snapshot, baseline_path)
    manifest_path = baseline_path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["questions_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="question file hash differs"):
        diagnose(
            questions=QUESTIONS,
            corpus=snapshot / "corpus",
            runs=[baseline_path],
            output=tmp_path / "diagnostics.json",
        )


def test_tool_diagnostic_counts_success_error_and_repeat():
    row = {
        "trajectory": [
            {"action": "print(search('focus'))", "observation": "[{'doc_id': 'one'}]"},
            {"action": "print(read('one'))", "observation": "example text"},
            {"action": "print(read('one'))", "observation": "example text"},
            {"action": "print(extract('one', '['))", "observation": "Traceback: bad regex"},
            {"action": "SUBMIT: answer CITATIONS: []", "observation": "Submitted"},
        ]
    }
    assert _step_diagnostics(row) == {
        "document_tool_action_steps": 4,
        "successful_document_tool_action_steps": 3,
        "execution_error_steps": 1,
        "repeated_action_steps": 1,
    }
