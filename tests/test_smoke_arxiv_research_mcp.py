import asyncio

import pytest

pytest.importorskip("mcp")

from scripts import smoke_arxiv_research_mcp as smoke  # noqa: E402
from tests.arxiv_fakes import article_page, make_library  # noqa: E402


class Discovery:
    def search(self, query, since=None, until=None, limit=10):
        return {"source": "arxiv" if since else "semantic_scholar", "papers": []}

    def citations(self, arxiv_id, direction="references", limit=20):
        return {"arxiv_id": arxiv_id, "direction": direction, "papers": []}


def test_smoke_uses_mcp_and_checks_exact_passages_without_models(tmp_path, monkeypatch):
    html = article_page().replace(b"cache eviction policy", b"executable Python code actions JSON")
    library, _ = make_library(tmp_path, {"2402.01030v2": {"html": html}})
    original, configurations = smoke.create_server, []

    def server(library, discovery, **kwargs):
        configurations.append(kwargs)
        return original(library, discovery, **kwargs)

    monkeypatch.setattr(smoke, "create_server", server)
    report = asyncio.run(smoke.run_smoke(library, Discovery(), paper_ids=["2402.01030"]))
    assert configurations == [{"investigator": None, "reasoner": None}]
    assert report["model_budget"] == {
        "requests": 0, "input_tokens": 0, "output_tokens": 0, "usd": 0}
    assert report["actual_model_requests"] == report["actual_input_tokens"] == 0
    assert report["actual_output_tokens"] == report["actual_model_cost_usd"] == 0
    assert report["unconfigured_model_check"] is True
    assert report["paid_smoke"]["status"] == "skipped"
    assert report["provenance_checks"]["exact"] == report["provenance_checks"]["checked"] > 0
    assert all(p["retrieved_quote_exact"] for p in report["review_passages"])
    assert report["semantic_support"] == "not_reviewed"
    assert report["claim_level_review"]["status"] == "not_performed"
    assert len(report["tools"]) <= smoke.MAX_TOOL_CALLS


def test_smoke_has_no_paid_option():
    with pytest.raises(SystemExit):
        smoke.parse_args(["--paid"])


def test_smoke_accepts_separate_detail_output(tmp_path):
    args = smoke.parse_args(["--library-root", str(tmp_path / "library"),
                             "--output", str(tmp_path / "summary.json"),
                             "--detail-output", str(tmp_path / "detail.json")])
    assert args.detail_output == tmp_path / "detail.json"
    assert args.output != args.detail_output


def test_fetch_budget_and_pdf_bound(tmp_path, monkeypatch):
    seen = []

    def http(url, headers, *, timeout, max_bytes):
        seen.append((url, timeout, max_bytes))
        return b"response"

    monkeypatch.setattr(smoke, "http_get", http)
    monkeypatch.setattr(smoke, "MAX_FETCH_CALLS", 1)
    library, discovery, state = smoke.build_sources(tmp_path)
    assert library._fetch_pdf("https://arxiv.org/pdf/2402.01030") == b"response"
    assert seen == [("https://arxiv.org/pdf/2402.01030", smoke.HTTP_TIMEOUT, 5_000_000)]
    with pytest.raises(smoke.SourceUnavailable, match="budget exhausted"):
        discovery.http("https://api.semanticscholar.org/x", {})
    assert len(state["fetches"]) == 1


def test_source_tool_failures_remain_in_report(tmp_path):
    library, _ = make_library(tmp_path, {})
    report = asyncio.run(smoke.run_smoke(library, Discovery(), paper_ids=["2402.01030"]))
    failed = next(row for row in report["tools"] if row["tool"] == "open_paper")
    assert failed["result"]["status"] == "source_unavailable"
    assert report["actual_model_requests"] == 0


def test_public_report_hashes_source_text_and_bounds_quotes():
    quote = "one two three four five six seven eight nine ten eleven twelve"
    raw = {"tools": [{"result": {"abstract": quote, "evidence": [{"quote": quote}],
                                 "text": quote}}],
           "review_passages": [{"arxiv_id": "2402.01030v1", "start": 12, "quote": quote}] * 3}
    public = smoke.compact_report(raw, "raw-hash")
    assert public["raw_detail_sha256"] == "raw-hash"
    assert "abstract" not in public["tools"][0]["result"]
    assert "quote" not in public["tools"][0]["result"]["evidence"][0]
    assert "text" not in public["tools"][0]["result"]
    excerpts = [row["excerpt"] for row in public["review_passages"] if "excerpt" in row]
    assert len(excerpts) == 1 and len(excerpts[0].split()) == 8
    assert raw["tools"][0]["result"]["abstract"] == quote
    assert smoke.compact_report(raw, "raw-hash", raw_detail_path="out/new-detail.json")[
        "raw_detail_path"] == "out/new-detail.json"
