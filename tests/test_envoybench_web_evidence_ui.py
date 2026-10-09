"""The web-evidence pilot is a read-only, separate Studio report."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from benchmarks.envoybench.demo import create_app


def _report() -> dict:
    return {
        "schema_version": "web-evidence-report-v1",
        "status": "complete",
        "scope_note": "Assistant-checked pilot examples, not independent human labels.",
        "provenance": {
            "case_manifest_sha256": "case-sha",
            "capture_manifest_sha256": "capture-sha",
            "html_parser_version": "html-v1",
            "pdf_parser_version": "pdf-v1",
            "retriever": {"kind": "literal", "version": "v1", "top_k": 3},
            "model_checkpoint": None,
            "seed": None,
            "network_during_scoring": False,
        },
        "aggregate": {
            "cases_total": 1,
            "case_status_counts": {"parsed": 1},
            "facts_total": 1,
            "fact_status_counts": {"pass": 1},
            "parsed_fact_denominator": 1,
            "retained_fact_count": 1,
            "retention_rate_on_parsed": 1.0,
            "retrieval_hit_denominator": 1,
            "retrieval_hit_count": 1,
            "retrieval_hit_rate_on_retained": 1.0,
            "end_to_end_pass_rate": 1.0,
            "fetch_seconds_total": 0.1,
            "parse_seconds_total": 0.2,
            "search_seconds_total": 0.3,
            "model_api_cost_usd": 0,
            "infrastructure_cost_usd": None,
        },
        "cases": [
            {
                "id": "study-1",
                "format": "html",
                "requested_url": "https://example.org/study",
                "final_url": "https://example.org/study",
                "title": "A study",
                "status": "parsed",
                "raw_sha256": "raw-sha",
                "fetch_seconds": 0.1,
                "parse_seconds": 0.2,
                "search_seconds": 0.3,
                "facts": [
                    {
                        "id": "fact-1",
                        "query": "population",
                        "expected_text": "The study enrolled 42 participants.",
                        "status": "pass",
                        "expected_occurrences": 1,
                        "expected_spans": [
                            {"start": 18, "end": 53, "quote": "The study enrolled 42 participants."}
                        ],
                        "search_results": [
                            {
                                "start": 18,
                                "end": 53,
                                "quote": "The study enrolled 42 participants.",
                                "section": "Methods",
                                "score": 1.0,
                                "contains_expected": True,
                            }
                        ],
                    }
                ],
            }
        ],
    }


def test_web_evidence_report_is_read_only_and_kept_out_of_blind_review(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(_report()), encoding="utf-8")
    before = path.read_bytes()
    client = TestClient(
        create_app(web_evidence_report_path=path, paper_state_dir=tmp_path / "paper")
    )

    page = client.get("/web-evidence")
    assert page.status_code == 200
    assert "From a web page to evidence" in page.text
    assert "assistant-checked" in page.text
    assert "literal" in page.text.lower()
    assert "innerHTML" not in page.text
    for route in ("/", "/papers", "/model"):
        assert 'href="/web-evidence"' in client.get(route).text
    response = client.get("/api/web-evidence")
    assert response.status_code == 200
    assert response.json() == _report()
    assert client.post("/api/web-evidence", json={"status": "changed"}).status_code == 405
    assert path.read_bytes() == before

    blind = TestClient(
        create_app(
            blind_review_dir=tmp_path / "blind",
            web_evidence_report_path=path,
            paper_state_dir=tmp_path / "paper-blind",
        )
    )
    assert blind.get("/web-evidence").status_code == 404
    assert blind.get("/api/web-evidence").status_code == 404
    assert 'href="/web-evidence" data-blind-hidden data-live-only hidden' in (
        blind.get("/").text
    )
    assert blind.app.state.web_evidence_report is None


def test_web_evidence_page_explains_missing_report(tmp_path):
    client = TestClient(create_app(paper_state_dir=tmp_path / "paper"))
    assert client.get("/web-evidence").status_code == 200
    assert client.get("/api/web-evidence").status_code == 404
    assert "--web-evidence-report PATH" in client.get("/web-evidence").text


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"status": "running"}, "complete"),
        ({"schema_version": "unknown"}, "schema"),
        ({"cases": {}}, "cases"),
    ],
)
def test_web_evidence_rejects_nonfinal_or_invalid_reports(tmp_path, change, reason):
    report = _report()
    report.update(change)
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match=reason):
        create_app(web_evidence_report_path=path, paper_state_dir=tmp_path / "paper")
