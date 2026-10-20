"""Review blinding and missing-case retention are evidence boundaries."""

import json

from src.eval.native_pilot_packet import package, save


def test_missing_cases_remain_and_operator_metadata_does_not_enter_review(tmp_path):
    source = tmp_path / "raw"
    source.mkdir()
    questions = {"common_instruction": "same input", "questions": [
        {"id": f"MLP{i:02}", "question": "hypothetical question", "input": {}}
        for i in range(1, 6)]}
    save(source / "MLP01-B" / "result.json", {
        "question_id": "MLP01", "arm": "B", "status": "completed",
        "model_turns_attempted": 1, "answer": "Verbatim answer text.",
        "elapsed_seconds": 42, "token_usage": {"total": {"inputTokens": 100}},
        "items": [{"type": "webSearch", "id": "tool", "results": [
            {"url": "https://example.org/paper", "text": "source body retained only locally"}]}]})
    out = tmp_path / "packet"
    package(source, out, {}, questions, seed=73)
    report = json.loads((out / "collection-report.json").read_text())
    assert len(report["outcomes"]) == 10 and report["generation_turns_attempted"] == 1
    assert sum(r["status"] == "unattempted" for r in report["outcomes"]) == 9
    worksheet = json.loads((out / "review/worksheet.json").read_text())
    assert len(worksheet["pairs"]) == 5
    for pair in worksheet["pairs"]:
        assert pair["pairwise_preference"] is None
        assert pair["reviewer_kind"] is None
        for row in pair["answers"]:
            assert row["initial_verdict"] == "review_unknown"
            assert row["four_dimension_scores_or_null"] is None
    review_text = "\n".join(p.read_text() for p in (out / "review").rglob("*") if p.is_file())
    assert '"arm"' not in review_text and '"token_usage"' not in review_text
    assert "source body retained only locally" not in review_text
    assert "Verbatim answer text." in review_text
    receipt = json.loads((out / "operator/MLP01-B.json").read_text())
    assert receipt["tool_receipts"][0]["source_urls"] == ["https://example.org/paper"]
    assert receipt["tool_receipts"][0]["item_sha256"]
