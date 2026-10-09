"""A small, offline-replayable web-to-evidence extraction and retrieval pilot.

Capture is the only network phase. Score re-parses saved response bytes and tests
literal evidence retention plus lexical top-three retrieval. Neither measure is
a semantic judgement that a passage supports a claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import tempfile
import time
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.research import file_sources, web_sources
from src.research.tools_runtime import ResearchTools

CASE_SCHEMA = "web-evidence-cases-v1"
CAPTURE_SCHEMA = "web-evidence-capture-v1"
REPORT_SCHEMA = "web-evidence-report-v1"
_ID = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")
_CONTENT_TYPES = {
    "html": {"text/html", "application/xhtml+xml"},
    "pdf": {"application/pdf", "application/x-pdf", "application/octet-stream"},
}
_LIMITS = {"html": web_sources.MAX_HTML_BYTES, "pdf": web_sources.MAX_PDF_BYTES}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(data: dict) -> bytes:
    return (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _load_cases(path: Path) -> tuple[list[dict], str, Any]:
    """Validate only the benchmark contract; optional editorial metadata is allowed."""
    raw = path.read_bytes()
    data = json.loads(raw)
    if not isinstance(data, dict) or data.get("schema_version") != CASE_SCHEMA:
        raise ValueError(f"Case file must use {CASE_SCHEMA}")
    cases = data.get("cases")
    if not isinstance(cases, list) or not 1 <= len(cases) <= 100:
        raise ValueError("Case file must contain 1..100 cases")
    seen_cases: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Each case must be an object")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not _ID.fullmatch(case_id):
            raise ValueError("Case IDs must be lowercase slugs of at most 64 characters")
        if case_id in seen_cases:
            raise ValueError(f"Duplicate case ID: {case_id}")
        seen_cases.add(case_id)
        if case.get("format") not in _CONTENT_TYPES:
            raise ValueError(f"Case {case_id} format must be html or pdf")
        if "expected_title" in case:
            expected_title = case["expected_title"]
            if (
                case["format"] != "html"
                or not isinstance(expected_title, str)
                or not expected_title.strip()
                or len(expected_title) > 2000
            ):
                raise ValueError(f"Case {case_id} expected_title must be nonempty HTML title")
        web_sources.canonicalize_web_url(case.get("url"))
        facts = case.get("facts")
        if not isinstance(facts, list) or not 1 <= len(facts) <= 20:
            raise ValueError(f"Case {case_id} must have 1..20 facts")
        seen_facts: set[str] = set()
        for fact in facts:
            if not isinstance(fact, dict):
                raise ValueError(f"Case {case_id} has a non-object fact")
            fact_id = fact.get("id")
            if not isinstance(fact_id, str) or not _ID.fullmatch(fact_id):
                raise ValueError(f"Case {case_id} fact IDs must be lowercase slugs")
            if fact_id in seen_facts:
                raise ValueError(f"Case {case_id} has duplicate fact ID: {fact_id}")
            seen_facts.add(fact_id)
            for field in ("query", "expected_text"):
                value = fact.get(field)
                if not isinstance(value, str) or not value.strip() or len(value) > 2000:
                    raise ValueError(f"Case {case_id}/{fact_id} needs a nonempty {field}")
    return cases, _sha256(raw), data.get("review")


def _source_name(index: int, case: dict) -> str:
    return f"{index:03d}-{case['id']}.{case['format']}"


def capture(
    cases_path: Path,
    output: Path,
    *,
    fetch_html: Callable | None = None,
    fetch_pdf: Callable | None = None,
) -> dict:
    """Fetch bounded public responses once and freeze their exact bytes.

    Fetch failures are data points. A new directory is required for each attempt,
    so an old response cannot silently be replaced by a changed website.
    """
    cases, case_hash, _ = _load_cases(Path(cases_path))
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    raw_dir = output / "raw"
    raw_dir.mkdir()
    html_fetcher = fetch_html or web_sources.fetch_web_page
    pdf_fetcher = fetch_pdf or web_sources.fetch_public_pdf
    rows = []
    for index, case in enumerate(cases):
        requested = web_sources.canonicalize_web_url(case["url"])
        row: dict[str, Any] = {
            "id": case["id"],
            "format": case["format"],
            "requested_url": requested,
        }
        start = time.perf_counter()
        try:
            fetcher = html_fetcher if case["format"] == "html" else pdf_fetcher
            page = fetcher(requested)
            if not isinstance(page, web_sources.FetchedPage) or not isinstance(page.body, bytes):
                raise TypeError("Fetcher must return FetchedPage with bytes body")
            if not page.body or len(page.body) > _LIMITS[case["format"]]:
                raise ValueError("Source is empty or exceeds its size limit")
            if page.content_type not in _CONTENT_TYPES[case["format"]]:
                raise ValueError("Source response has an unexpected content type")
            if case["format"] == "pdf" and not page.body.startswith(b"%PDF-"):
                raise ValueError("Source response is not a PDF")
            final_url = web_sources.canonicalize_web_url(page.final_url)
            if requested.startswith("https://") and not final_url.startswith("https://"):
                raise ValueError("HTTPS source cannot redirect to HTTP")
            raw_file = _source_name(index, case)
            (raw_dir / raw_file).write_bytes(page.body)
            row.update(
                status="captured",
                final_url=final_url,
                content_type=page.content_type,
                charset=page.charset,
                raw_file=raw_file,
                raw_sha256=_sha256(page.body),
                raw_bytes=len(page.body),
            )
        except Exception as exc:
            row.update(status="fetch_error", error=f"{type(exc).__name__}: {exc}")
        row["fetch_seconds"] = round(time.perf_counter() - start, 6)
        rows.append(row)
    manifest = {
        "schema_version": CAPTURE_SCHEMA,
        "captured_at_utc": _utc_now(),
        "case_manifest_sha256": case_hash,
        "capture_limits_bytes": _LIMITS,
        "cases": rows,
    }
    (output / "manifest.json").write_bytes(_json_bytes(manifest))
    return manifest


def _read_capture(cases: list[dict], case_hash: str, capture_dir: Path) -> tuple[list[dict], str]:
    manifest_raw = (capture_dir / "manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != CAPTURE_SCHEMA:
        raise ValueError("Capture manifest has the wrong schema")
    if manifest.get("case_manifest_sha256") != case_hash:
        raise ValueError("Capture was made from a different case manifest hash")
    rows = manifest.get("cases")
    if not isinstance(rows, list) or len(rows) != len(cases):
        raise ValueError("Capture case count differs from case manifest")
    for index, (case, row) in enumerate(zip(cases, rows)):
        if not isinstance(row, dict) or any(
            row.get(key) != value
            for key, value in (
                ("id", case["id"]),
                ("format", case["format"]),
                ("requested_url", web_sources.canonicalize_web_url(case["url"])),
            )
        ):
            raise ValueError("Capture case ID, URL, or format differs from case manifest")
        if row.get("status") == "captured":
            if row.get("raw_file") != _source_name(index, case):
                raise ValueError(f"Capture raw filename mismatch for {case['id']}")
            if row.get("content_type") not in _CONTENT_TYPES[case["format"]]:
                raise ValueError(f"Capture content type mismatch for {case['id']}")
            if not isinstance(row.get("charset"), str):
                raise ValueError(f"Capture charset missing for {case['id']}")
            web_sources.canonicalize_web_url(row.get("final_url"))
        elif row.get("status") != "fetch_error" or not isinstance(row.get("error"), str):
            raise ValueError(f"Capture status invalid for {case['id']}")
        fetch_seconds = row.get("fetch_seconds")
        if not isinstance(fetch_seconds, (float, int)) or fetch_seconds < 0:
            raise ValueError(f"Capture fetch timing invalid for {case['id']}")
    return rows, _sha256(manifest_raw)


def _read_raw(capture_dir: Path, row: dict, case: dict) -> bytes:
    raw_path = capture_dir / "raw" / row["raw_file"]
    if raw_path.is_symlink() or not raw_path.is_file():
        raise ValueError(f"Capture raw file missing or unsafe for {case['id']}")
    if raw_path.stat().st_size > _LIMITS[case["format"]]:
        raise ValueError(f"Capture raw file exceeds size limit for {case['id']}")
    raw = raw_path.read_bytes()
    if len(raw) != row.get("raw_bytes") or _sha256(raw) != row.get("raw_sha256"):
        raise ValueError(f"Capture raw hash mismatch for {case['id']}")
    return raw


def _parse_document(case: dict, row: dict, raw: bytes) -> tuple[dict, dict]:
    if case["format"] == "html":
        title, text, sections, decode_loss = web_sources.extract_html_text(raw, row["charset"])
        parser = {
            "kind": "html_visible_blocks",
            "version": web_sources.PARSER_VERSION,
            "decode_loss": decode_loss,
        }
    else:
        text, sections, coverage, empty_pages, pypdf_version = file_sources._pdf_document(raw)
        title = case["id"]
        parser = {
            "kind": coverage,
            "version": file_sources.PARSER_VERSION,
            "pypdf_version": pypdf_version,
            "empty_pages": empty_pages,
        }
    doc = {
        "doc_id": f"web_evidence_{case['id']}",
        "title": title,
        "text": text,
        "sections": sections,
        "metadata": {
            "source_url": row["final_url"],
            "coverage": parser["kind"],
            "submitted": "unknown",
            "source_kind": "web_page" if case["format"] == "html" else "fetched_pdf",
        },
    }
    return doc, parser


def _literal_spans(text: str, expected: str) -> tuple[int, list[dict]]:
    """Count all exact occurrences while keeping the report bounded."""
    count, spans, cursor = 0, [], 0
    while (start := text.find(expected, cursor)) != -1:
        count += 1
        if len(spans) < 20:
            spans.append({"start": start, "end": start + len(expected), "quote": expected})
        cursor = start + 1
    return count, spans


def _normalized_title(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def _median_and_p95(values: list[float]) -> tuple[float | None, float | None]:
    """Use an exact median and nearest-rank p95; tiny pilots do not merit interpolation."""
    if not values:
        return None, None
    ordered = sorted(values)
    return round(statistics.median(ordered), 6), round(
        ordered[math.ceil(0.95 * len(ordered)) - 1], 6
    )


def score(cases_path: Path, capture_dir: Path, output: Path) -> dict:
    """Evaluate frozen bytes with no fetcher, URL opener, or model call."""
    cases, case_hash, review = _load_cases(Path(cases_path))
    capture_dir, output = Path(capture_dir), Path(output)
    if output.exists():
        raise FileExistsError(output)
    capture_rows, capture_hash = _read_capture(cases, case_hash, capture_dir)
    results: list[dict] = []
    parsed: dict[str, dict] = {}
    # Parse every successful capture before querying one frozen corpus.
    with tempfile.TemporaryDirectory(prefix="envoy-web-evidence-") as temp_name:
        corpus = Path(temp_name)
        for case, capture_row in zip(cases, capture_rows):
            result: dict[str, Any] = {
                "id": case["id"],
                "format": case["format"],
                "requested_url": capture_row["requested_url"],
                "final_url": capture_row.get("final_url"),
                "raw_sha256": capture_row.get("raw_sha256"),
                "raw_bytes": capture_row.get("raw_bytes", 0),
                "fetch_seconds": capture_row["fetch_seconds"],
                "parse_seconds": 0.0,
                "search_seconds": 0.0,
                "facts": [],
            }
            for field in ("stratum", "review_status"):
                if field in case:
                    result[field] = case[field]
            if "expected_title" in case:
                result["title_diagnostic"] = {
                    "expected_title": case["expected_title"],
                    "extracted_title": None,
                    "match": None,
                    "normalization": "unicode_nfkc_whitespace_casefold_v1",
                }
            if capture_row["status"] == "fetch_error":
                result["status"] = "fetch_error"
                result["error"] = capture_row["error"]
            else:
                raw = _read_raw(capture_dir, capture_row, case)
                start = time.perf_counter()
                try:
                    doc, parser = _parse_document(case, capture_row, raw)
                except Exception as exc:
                    result["status"] = "parse_error"
                    result["error"] = f"{type(exc).__name__}: {exc}"
                else:
                    result["status"] = "parsed"
                    result["parsed_chars"] = len(doc["text"])
                    result["parsed_sha256"] = _sha256(doc["text"].encode("utf-8"))
                    result["parser"] = parser
                    result["title"] = doc["title"]
                    if "expected_title" in case:
                        result["title_diagnostic"]["extracted_title"] = doc["title"]
                        result["title_diagnostic"]["match"] = (
                            _normalized_title(doc["title"])
                            == _normalized_title(case["expected_title"])
                        )
                    parsed[case["id"]] = doc
                    (corpus / f"{doc['doc_id']}.json").write_bytes(_json_bytes(doc))
                result["parse_seconds"] = round(time.perf_counter() - start, 6)
            results.append(result)

        tools = ResearchTools(corpus) if parsed else None
        for case, result in zip(cases, results):
            for fact in case["facts"]:
                fact_result: dict[str, Any] = {
                    "id": fact["id"],
                    "query": fact["query"],
                    "expected_text": fact["expected_text"],
                    "status": result["status"],
                    "expected_occurrences": 0,
                    "expected_spans": [],
                    "search_results": [],
                    "search_seconds": None,
                }
                if "review_status" in fact:
                    fact_result["review_status"] = fact["review_status"]
                if result["status"] == "parsed":
                    doc = parsed[case["id"]]
                    count, spans = _literal_spans(doc["text"], fact["expected_text"])
                    fact_result["expected_occurrences"] = count
                    fact_result["expected_spans"] = spans
                    if count == 0:
                        fact_result["status"] = "fact_missing"
                    else:
                        start = time.perf_counter()
                        assert tools is not None
                        hits = tools.search_paper(doc["doc_id"], fact["query"], top_k=3)
                        elapsed = time.perf_counter() - start
                        fact_result["search_seconds"] = round(elapsed, 6)
                        result["search_seconds"] += elapsed
                        for hit in hits:
                            quote = hit["quote"]
                            if doc["text"][hit["start"] : hit["end"]] != quote:
                                raise ValueError(
                                    "Retriever returned a passage without exact offsets"
                                )
                            local_start = quote.find(fact["expected_text"])
                            fact_result["search_results"].append(
                                {
                                    "start": hit["start"],
                                    "end": hit["end"],
                                    "quote": quote,
                                    "section": hit["section"],
                                    "score": hit["score"],
                                    "contains_expected": local_start >= 0,
                                    "matched_span": (
                                        {
                                            "start": hit["start"] + local_start,
                                            "end": (
                                                hit["start"]
                                                + local_start
                                                + len(fact["expected_text"])
                                            ),
                                        }
                                        if local_start >= 0
                                        else None
                                    ),
                                }
                            )
                        fact_result["status"] = (
                            "pass"
                            if any(
                                hit["contains_expected"]
                                for hit in fact_result["search_results"]
                            )
                            else "retrieval_miss"
                        )
                result["facts"].append(fact_result)
            result["search_seconds"] = round(result["search_seconds"], 6)

    fact_statuses = Counter(fact["status"] for row in results for fact in row["facts"])
    case_statuses = Counter(row["status"] for row in results)
    parsed_facts = sum(
        len(case["facts"])
        for case, row in zip(cases, results)
        if row["status"] == "parsed"
    )
    retained = fact_statuses["pass"] + fact_statuses["retrieval_miss"]
    total_facts = sum(len(case["facts"]) for case in cases)
    title_results = [
        row["title_diagnostic"]["match"]
        for row in results
        if row["status"] == "parsed" and "title_diagnostic" in row
    ]
    fetch_times = [row["fetch_seconds"] for row in results]
    search_times = [
        fact["search_seconds"]
        for row in results
        for fact in row["facts"]
        if fact["search_seconds"] is not None
    ]
    fetch_median, fetch_p95 = _median_and_p95(fetch_times)
    search_median, search_p95 = _median_and_p95(search_times)
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "complete",
        "scored_at_utc": _utc_now(),
        "review": review,
        "provenance": {
            "case_manifest_sha256": case_hash,
            "capture_manifest_sha256": capture_hash,
            "html_parser_version": web_sources.PARSER_VERSION,
            "pdf_parser_version": file_sources.PARSER_VERSION,
            "retriever": {"kind": "lexical_paragraph", "version": "v1", "top_k": 3},
            "model_checkpoint": None,
            "seed": None,
            "network_during_scoring": False,
        },
        "scope_note": (
            "Literal retained text and lexical passage retrieval only. "
            "No semantic support, source truth, model quality, or answer quality judgement."
        ),
        "aggregate": {
            "cases_total": len(cases),
            "case_status_counts": dict(case_statuses),
            "facts_total": total_facts,
            "fact_status_counts": dict(fact_statuses),
            "parsed_fact_denominator": parsed_facts,
            "retained_fact_count": retained,
            "retention_rate_on_parsed": retained / parsed_facts if parsed_facts else None,
            "retrieval_hit_denominator": retained,
            "retrieval_hit_count": fact_statuses["pass"],
            "retrieval_hit_rate_on_retained": (
                fact_statuses["pass"] / retained if retained else None
            ),
            "end_to_end_pass_rate": fact_statuses["pass"] / total_facts,
            "html_title_match_count": sum(title_results),
            "html_title_denominator": len(title_results),
            "html_title_match_rate": (
                sum(title_results) / len(title_results) if title_results else None
            ),
            "raw_bytes_total": sum(row["raw_bytes"] for row in results),
            "parsed_chars_total": sum(row.get("parsed_chars", 0) for row in results),
            "fetch_seconds_total": round(sum(row["fetch_seconds"] for row in results), 6),
            "parse_seconds_total": round(sum(row["parse_seconds"] for row in results), 6),
            "search_seconds_total": round(sum(row["search_seconds"] for row in results), 6),
            "fetch_latency_denominator": len(fetch_times),
            "fetch_seconds_median": fetch_median,
            "fetch_seconds_p95": fetch_p95,
            "search_latency_denominator": len(search_times),
            "search_seconds_median": search_median,
            "search_seconds_p95": search_p95,
            "latency_percentile_method": "median_exact_p95_nearest_rank",
            "model_api_cost_usd": 0.0,
            "infrastructure_cost_usd": None,
        },
        "cases": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as file:
        file.write(_json_bytes(report))
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    capture_parser = subcommands.add_parser("capture", help="Freeze bounded public web responses")
    capture_parser.add_argument("--cases", type=Path, required=True)
    capture_parser.add_argument("--output", type=Path, required=True)
    score_parser = subcommands.add_parser("score", help="Score only the frozen response bytes")
    score_parser.add_argument("--cases", type=Path, required=True)
    score_parser.add_argument("--capture", type=Path, required=True)
    score_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "capture":
        result = capture(args.cases, args.output)
        counts = Counter(row["status"] for row in result["cases"])
        print(json.dumps({"output": str(args.output), "case_status_counts": counts}))
    else:
        result = score(args.cases, args.capture, args.output)
        print(json.dumps({"output": str(args.output), "aggregate": result["aggregate"]}))


if __name__ == "__main__":
    main()
