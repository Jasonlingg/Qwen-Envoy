"""Frozen web-to-evidence pilot checks without live websites or model calls."""

import io
import json

import pytest

from benchmarks.envoybench.web_evidence import _median_and_p95, capture, score
from src.research.web_sources import FetchedPage

PAGE = b"""<!doctype html><html><head><title>Evidence Memo | Example</title></head>
<body><nav>navigation noise</nav><article><h1>Evidence Memo</h1>
<p>Widget protocol requires one signed acknowledgement.</p>
<p>The follow-up test measured 12 milliseconds.</p>
</article></body></html>"""


def _write_cases(path, cases, *, review=None):
    path.write_text(
        json.dumps(
            {
                "schema_version": "web-evidence-cases-v1",
                "review": review,
                "cases": cases,
            }
        )
    )


def _html_case(case_id="memo", url="https://example.org/memo"):
    return {
        "id": case_id,
        "url": url,
        "format": "html",
        "stratum": "article",
        "expected_title": "Evidence Memo",
        "review_status": "assistant_cross_checked",
        "facts": [
            {
                "id": "retained_and_found",
                "query": "widget protocol signed acknowledgement",
                "expected_text": "Widget protocol requires one signed acknowledgement.",
                "review_status": "assistant_cross_checked",
            },
            {
                "id": "retained_not_found",
                "query": "totally unrelated lexeme",
                "expected_text": "The follow-up test measured 12 milliseconds.",
            },
            {
                "id": "missing_from_parse",
                "query": "navigation noise",
                "expected_text": "navigation noise",
            },
        ],
    }


def test_latency_summary_uses_nearest_rank_p95():
    assert _median_and_p95([]) == (None, None)
    assert _median_and_p95([0.4, 0.1, 0.3, 0.2]) == (0.25, 0.4)


def test_capture_and_offline_score_distinguish_parser_retrieval_and_title(tmp_path, monkeypatch):
    cases_path = tmp_path / "cases.json"
    _write_cases(
        cases_path,
        [_html_case()],
        review={"status": "assistant_cross_checked", "independent_human_review": False},
    )
    capture_dir = tmp_path / "capture"
    seen = []

    def fake_fetch(url):
        seen.append(url)
        return FetchedPage(PAGE, "https://example.org/final", "text/html", "utf-8")

    manifest = capture(cases_path, capture_dir, fetch_html=fake_fetch)
    assert seen == ["https://example.org/memo"]
    assert manifest["cases"][0]["status"] == "captured"
    assert manifest["cases"][0]["raw_sha256"]
    assert (capture_dir / "raw" / "000-memo.html").read_bytes() == PAGE

    def disallow_network(_url):
        pytest.fail("score must not fetch the network")

    monkeypatch.setattr("src.research.web_sources.fetch_web_page", disallow_network)
    monkeypatch.setattr("src.research.web_sources.fetch_public_pdf", disallow_network)
    report_path = tmp_path / "report.json"
    report = score(cases_path, capture_dir, report_path)
    assert report == json.loads(report_path.read_text())
    assert report["status"] == "complete"
    assert report["review"]["independent_human_review"] is False
    assert report["provenance"]["network_during_scoring"] is False
    row = report["cases"][0]
    assert row["status"] == "parsed"
    assert row["stratum"] == "article"
    assert row["review_status"] == "assistant_cross_checked"
    assert row["title_diagnostic"]["match"] is False
    assert row["title"] == "Evidence Memo | Example"
    facts = {fact["id"]: fact for fact in row["facts"]}
    assert facts["retained_and_found"]["status"] == "pass"
    assert facts["retained_not_found"]["status"] == "retrieval_miss"
    assert facts["missing_from_parse"]["status"] == "fact_missing"
    assert facts["retained_and_found"]["search_seconds"] >= 0
    assert facts["retained_not_found"]["search_seconds"] >= 0
    assert facts["missing_from_parse"]["search_seconds"] is None
    passed = facts["retained_and_found"]
    span = passed["expected_spans"][0]
    hit = next(hit for hit in passed["search_results"] if hit["contains_expected"])
    assert hit["matched_span"] == {"start": span["start"], "end": span["end"]}
    assert hit["quote"][
        span["start"] - hit["start"] : span["end"] - hit["start"]
    ] == span["quote"]
    agg = report["aggregate"]
    assert agg["facts_total"] == 3
    assert agg["fact_status_counts"] == {
        "pass": 1,
        "retrieval_miss": 1,
        "fact_missing": 1,
    }
    assert agg["retention_rate_on_parsed"] == pytest.approx(2 / 3)
    assert agg["retrieval_hit_rate_on_retained"] == 0.5
    assert agg["html_title_match_count"] == 0
    assert agg["html_title_denominator"] == 1
    assert agg["fetch_latency_denominator"] == 1
    assert agg["search_latency_denominator"] == 2
    assert agg["fetch_seconds_median"] == agg["fetch_seconds_p95"]
    assert agg["search_seconds_p95"] >= agg["search_seconds_median"] >= 0
    assert agg["latency_percentile_method"] == "median_exact_p95_nearest_rank"
    assert agg["model_api_cost_usd"] == 0.0
    assert agg["infrastructure_cost_usd"] is None


def test_fetch_and_parse_failures_have_distinct_denominators(tmp_path):
    cases_path = tmp_path / "cases.json"
    cases = [
        _html_case("unavailable", "https://example.org/unavailable"),
        _html_case("empty", "https://example.org/empty"),
    ]
    _write_cases(cases_path, cases)

    def fake_fetch(url):
        if url.endswith("unavailable"):
            raise OSError("offline test outage")
        return FetchedPage(b"<html><title>Empty</title></html>", url)

    capture_dir = tmp_path / "capture"
    manifest = capture(cases_path, capture_dir, fetch_html=fake_fetch)
    assert [row["status"] for row in manifest["cases"]] == ["fetch_error", "captured"]
    report = score(cases_path, capture_dir, tmp_path / "report.json")
    assert [row["status"] for row in report["cases"]] == ["fetch_error", "parse_error"]
    assert {fact["status"] for fact in report["cases"][0]["facts"]} == {"fetch_error"}
    assert {fact["status"] for fact in report["cases"][1]["facts"]} == {"parse_error"}
    assert report["aggregate"]["parsed_fact_denominator"] == 0
    assert report["aggregate"]["retention_rate_on_parsed"] is None
    assert report["aggregate"]["facts_total"] == 6
    assert report["aggregate"]["end_to_end_pass_rate"] == 0
    assert report["aggregate"]["search_latency_denominator"] == 0
    assert report["aggregate"]["search_seconds_median"] is None
    assert report["aggregate"]["search_seconds_p95"] is None


def test_tampered_raw_or_changed_case_file_refuses_score(tmp_path):
    cases_path = tmp_path / "cases.json"
    _write_cases(cases_path, [_html_case()])
    capture_dir = tmp_path / "capture"
    capture(cases_path, capture_dir, fetch_html=lambda url: FetchedPage(PAGE, url))
    raw_path = capture_dir / "raw" / "000-memo.html"
    raw_path.write_bytes(PAGE + b"tampered")
    report_path = tmp_path / "report.json"
    with pytest.raises(ValueError, match="hash mismatch"):
        score(cases_path, capture_dir, report_path)
    assert not report_path.exists()
    raw_path.write_bytes(PAGE)
    cases = [_html_case()]
    cases[0]["facts"][0]["query"] = "different query"
    _write_cases(cases_path, cases)
    with pytest.raises(ValueError, match="different case manifest hash"):
        score(cases_path, capture_dir, report_path)
    assert not report_path.exists()


def test_existing_capture_and_report_are_never_overwritten(tmp_path):
    cases_path = tmp_path / "cases.json"
    _write_cases(cases_path, [_html_case()])
    capture_dir = tmp_path / "capture"
    capture(cases_path, capture_dir, fetch_html=lambda url: FetchedPage(PAGE, url))
    with pytest.raises(FileExistsError):
        capture(cases_path, capture_dir, fetch_html=lambda url: pytest.fail("must not fetch"))
    report_path = tmp_path / "report.json"
    first = score(cases_path, capture_dir, report_path)
    with pytest.raises(FileExistsError):
        score(cases_path, capture_dir, report_path)
    assert json.loads(report_path.read_text()) == first


def _pdf_bytes() -> bytes:
    pypdf = pytest.importorskip("pypdf")
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = pypdf.PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})}
    )
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 20 200 Td (Evidence is on page one.) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    result = io.BytesIO()
    writer.write(result)
    return result.getvalue()


def test_pdf_capture_preserves_page_evidence(tmp_path):
    raw = _pdf_bytes()
    cases_path = tmp_path / "cases.json"
    _write_cases(
        cases_path,
        [
            {
                "id": "paper",
                "url": "https://example.org/paper.pdf",
                "format": "pdf",
                "facts": [
                    {
                        "id": "first_page",
                        "query": "evidence page one",
                        "expected_text": "Evidence is on page one.",
                    }
                ],
            }
        ],
    )
    capture_dir = tmp_path / "capture"
    capture(
        cases_path,
        capture_dir,
        fetch_pdf=lambda url: FetchedPage(raw, url, "application/pdf"),
    )
    report = score(cases_path, capture_dir, tmp_path / "report.json")
    row = report["cases"][0]
    assert row["status"] == "parsed"
    assert row["parser"]["kind"] == "pdf_text_no_ocr"
    assert row["facts"][0]["status"] == "pass"
    assert row["facts"][0]["search_results"][0]["section"] == "Page 1"


def test_malformed_ids_and_private_urls_rejected_before_network(tmp_path):
    cases_path = tmp_path / "cases.json"
    bad = _html_case("../escape", "https://example.org/memo")
    _write_cases(cases_path, [bad])
    with pytest.raises(ValueError, match="lowercase slugs"):
        capture(cases_path, tmp_path / "capture", fetch_html=lambda url: pytest.fail("network"))
    assert not (tmp_path / "capture").exists()
    bad = _html_case("safe", "file:///etc/passwd")
    _write_cases(cases_path, [bad])
    with pytest.raises(ValueError, match="HTTP"):
        capture(cases_path, tmp_path / "capture", fetch_html=lambda url: pytest.fail("network"))
    assert not (tmp_path / "capture").exists()
