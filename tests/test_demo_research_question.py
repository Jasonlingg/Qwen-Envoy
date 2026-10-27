"""The question-first walkthrough survives a real MCP restart without network."""

import asyncio

from scripts import demo_research_question
from tests.test_ml_experiment_evidence import sources


def test_real_mcp_question_walkthrough_with_fixture_paper(tmp_path, monkeypatch):
    library, _, _, _ = sources(tmp_path)
    monkeypatch.setattr(demo_research_question, "ArxivLibrary", lambda _: library)
    monkeypatch.setattr(demo_research_question, "PAPER", "2609.12345v2")
    report = asyncio.run(demo_research_question.run(tmp_path / "demo"))
    question = report["question"]
    assert len(report["mcp_calls"]) == 14
    assert not any(call["is_error"] for call in report["mcp_calls"])
    assert report["assessment"] == "inconclusive"
    assert report["model_requests"] == report["gpu_seconds"] == 0
    assert question["considered_papers"][0]["receipt"]["quote"]
    assert question["paper_to_result_links"][0]["receipt"]["result_data_kind"] == "recorded"
    assert "latency benefit remains unknown" in question["current_view"]["view"][
        "statement"]
