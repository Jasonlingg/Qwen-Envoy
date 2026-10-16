"""Free live MCP smoke: source requests only; no reader or reasoner is configured."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from mcp import Client

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.arxiv_research_mcp import create_server  # noqa: E402
from src.product.arxiv_library import ArxivLibrary  # noqa: E402
from src.product.net import MAX_REDIRECTS, SourceUnavailable, http_get  # noqa: E402
from src.product.paper_discovery import PaperDiscovery  # noqa: E402

PAPERS = ("2402.01030", "2510.03215", "2511.20639")
MAX_TOOL_CALLS, MAX_FETCH_CALLS = 20, 12
SOURCE_DEADLINE, HTTP_TIMEOUT, TOOL_TIMEOUT = 180, 15, 120
PAID_SKIP_REASON = (
    "Paid smoke skipped: reliable pre-request tokenization and a provider-enforced "
    "all-in <=$0.50 limit are unavailable on this model path. Reader and reasoner "
    "are unconfigured; request, input-token, and output-token budgets are zero."
)


def build_sources(root: Path):
    started, lock = time.monotonic(), threading.Lock()
    state = {"fetches": [], "max_fetch_calls": MAX_FETCH_CALLS,
             "per_hop_timeout_seconds": HTTP_TIMEOUT, "redirects_per_fetch_max": MAX_REDIRECTS}
    def fetch(url, headers=None, *, max_bytes=2_000_000):
        source = ("semantic_scholar" if urlsplit(url).hostname == "api.semanticscholar.org"
                  else "arxiv")
        with lock:
            if (len(state["fetches"]) >= MAX_FETCH_CALLS
                    or time.monotonic() - started > SOURCE_DEADLINE):
                raise SourceUnavailable(source, "free smoke source budget exhausted")
            row = {"url": url, "max_bytes": max_bytes}
            state["fetches"].append(row)
        begin = time.monotonic()
        try:
            body = http_get(url, headers, timeout=HTTP_TIMEOUT, max_bytes=max_bytes)
            row.update(status="ok", response_bytes=len(body))
            return body
        except Exception as exc:
            row.update(status="error", error=f"{type(exc).__name__}: {exc}"[:500])
            raise
        finally:
            row["elapsed_seconds"] = round(time.monotonic() - begin, 3)
    library = ArxivLibrary(root, fetch=lambda url: fetch(url),
                           fetch_pdf=lambda url: fetch(url, max_bytes=5_000_000))
    return library, PaperDiscovery(root / "discovery-cache", http=fetch), state


async def run_smoke(library, discovery, *, paper_ids=PAPERS) -> dict:
    server = create_server(library, discovery, investigator=None, reasoner=None)
    started = time.monotonic()
    report = {
        "kind": "free_live_mcp_smoke", "started_at": datetime.now(timezone.utc).isoformat(),
        "paid_smoke": {"status": "skipped", "reason": PAID_SKIP_REASON},
        "model_budget": {"requests": 0, "input_tokens": 0, "output_tokens": 0, "usd": 0},
        "actual_model_requests": 0, "actual_input_tokens": 0, "actual_output_tokens": 0,
        "actual_model_cost_usd": 0, "max_tool_calls": MAX_TOOL_CALLS,
        "start_new_tools_deadline_seconds": SOURCE_DEADLINE,
        "per_tool_timeout_seconds": TOOL_TIMEOUT, "tools": [], "papers": [],
        "review_passages": [], "semantic_support": "not_reviewed",
        "claim_level_review": {"status": "not_performed", "reviewer": None},
        "warning": "Exact quotes prove provenance, not semantic support or model improvement.",
    }
    async with Client(server, read_timeout_seconds=TOOL_TIMEOUT) as client:
        async def call(name, arguments):
            if (len(report["tools"]) >= MAX_TOOL_CALLS
                    or time.monotonic() - started > SOURCE_DEADLINE):
                return {"status": "smoke_budget_exhausted"}
            begin = time.monotonic()
            row = {"tool": name, "arguments": arguments}
            report["tools"].append(row)
            try:
                result = await client.call_tool(name, arguments, read_timeout_seconds=TOOL_TIMEOUT)
                payload = result.structured_content or {}
                row.update(is_error=result.is_error, result=payload)
                if result.is_error:
                    row["error"] = str(result.content)[:500]
                return payload
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"[:500]
                return {"status": "tool_error", "error": row["error"]}
            finally:
                row["elapsed_seconds"] = round(time.monotonic() - begin, 3)
        await call("search_papers", {"query": "CodeAct executable code actions", "limit": 3})
        await call("get_citations", {"arxiv_id": PAPERS[0], "limit": 3})
        await call("search_papers", {"query": "all:CodeAct", "since": "2024-01-01",
                                     "until": "2026-10-16", "limit": 3})
        unconfigured = await call("investigate", {
            "question": "What evidence supports code actions?", "arxiv_ids": [PAPERS[0]]})
        report["unconfigured_model_check"] = (
            unconfigured.get("status") == "model_not_configured"
            and unconfigured.get("model_requests_attempted") == 0)
        for identifier in paper_ids[:3]:
            opened = await call("open_paper", {"arxiv_id": identifier})
            if opened.get("status") != "ok":
                continue
            report["papers"].append(opened)
            version = opened["arxiv_id"]
            query = ("executable Python code actions JSON 20%" if identifier == PAPERS[0]
                     else "agent memory reasoning retrieval results")
            found = await call("find_evidence", {
                "query": query, "arxiv_ids": [version], "top_k": 3})
            document = library.document(version)
            spans = found.get("evidence") or [{"start": 0, "end": min(900, len(document["text"]))}]
            for span in spans[:3]:
                start, end = span["start"], min(span["end"], span["start"] + 900)
                passage = await call("get_passage", {"arxiv_id": version, "start": start,
                                                       "length": end - start})
                exact = (passage.get("text") == document["text"][start:end]
                         and passage.get("start") == start and passage.get("end") == end)
                report["review_passages"].append({
                    "arxiv_id": version, "title": opened["title"], "coverage": opened["coverage"],
                    "source_url": opened["source_url"], "start": start, "end": end,
                    "quote": passage.get("text"), "exact_source_match": exact,
                    "retrieved_quote_exact": (span.get("quote") == document["text"][
                        span["start"]:span["end"]]) if "quote" in span else None,
                    "semantic_support": "not_reviewed", "section": span.get("section"),
                })
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    report["provenance_checks"] = {
        "checked": len(report["review_passages"]),
        "exact": sum(p["exact_source_match"] for p in report["review_passages"]),
    }
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library-root", type=Path, default=ROOT / "out/arxiv-mcp-live-library")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports/arxiv-mcp-live-20261016.json")
    return parser.parse_args(argv)


def compact_report(report, raw_sha256):
    """Keep hashes for full source text and at most eight quoted words per paper."""
    def scrub(value):
        if isinstance(value, list):
            return [scrub(item) for item in value]
        if not isinstance(value, dict):
            return value
        result = {}
        for key, item in value.items():
            if key in {"abstract", "quote", "text"} and isinstance(item, str):
                result[key + "_sha256"] = hashlib.sha256(item.encode()).hexdigest()
                result[key + "_chars"] = len(item)
            else:
                result[key] = scrub(item)
        return result
    compact, seen = scrub(report), set()
    for original, row in zip(report["review_passages"], compact["review_passages"]):
        if original["arxiv_id"] not in seen and isinstance(original.get("quote"), str):
            words = list(re.finditer(r"\S+", original["quote"]))[:8]
            if words:
                row["excerpt"] = original["quote"][:words[-1].end()]
                row["excerpt_start"], row["excerpt_end"] = (
                    original["start"], original["start"] + words[-1].end())
                seen.add(original["arxiv_id"])
    compact["raw_detail_path"] = "out/arxiv-mcp-live-detail.json"
    compact["raw_detail_sha256"] = raw_sha256
    return compact


def main() -> None:
    args = parse_args()
    library, discovery, network = build_sources(args.library_root)
    report = asyncio.run(run_smoke(library, discovery))
    report["network"] = network
    detail = ROOT / "out/arxiv-mcp-live-detail.json"
    detail.parent.mkdir(parents=True, exist_ok=True)
    detail.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    public = compact_report(report, hashlib.sha256(detail.read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(public, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({"report": str(args.output), "tools": len(report["tools"]),
                      "papers": len(report["papers"]), **report["provenance_checks"],
                      "actual_model_requests": 0, "paid_smoke": "skipped"}))


if __name__ == "__main__":
    main()
