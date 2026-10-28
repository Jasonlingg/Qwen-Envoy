"""One real MCP research question, with optional one-request Nebius explanation.

Use a fresh output directory. The default path uses public-paper source tools and
local records only. --live-nebius explicitly opts into a capped paid request.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import Client

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiment_mode_extended_mcp import create_server  # noqa: E402
from src.product.arxiv_library import ArxivLibrary  # noqa: E402
from src.product.experiment_discovery import ExperimentDiscovery  # noqa: E402
from src.product.experiment_nemotron import (  # noqa: E402
    MAX_OUTPUT_TOKENS,
    MODEL,
    public_paper_packet,
)
from src.product.ml_experiment_evidence import EvidenceResolver  # noqa: E402
from src.product.ml_experiment_store import ExperimentStore  # noqa: E402
from src.product.paper_discovery import PaperDiscovery  # noqa: E402
from src.research.explainer import EXPLAINER_PROMPT  # noqa: E402

PAPER = "2402.01030v4"
QUESTION = "Should our document-research agent use executable Python actions?"
PRICE_INPUT_PER_M = 0.06
PRICE_OUTPUT_PER_M = 0.24
MAX_SMOKE_USD = 0.50


async def run(output: Path, library_root: Path, *, live_nebius=False):
    output = output.expanduser().absolute()
    output.mkdir(parents=True, exist_ok=False)
    library = ArxivLibrary(library_root.expanduser())
    store = ExperimentStore(output / "records", evidence_resolver=EvidenceResolver(
        library_root=library.root))
    discovery = ExperimentDiscovery(PaperDiscovery(library.root / "discovery-cache"))
    key = None
    if live_nebius:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=False)
        key = os.environ.get("NEBIUS_API_KEY")
        if not key:
            raise ValueError("--live-nebius needs NEBIUS_API_KEY")
    calls = []

    async def call(client, name, args):
        response = await client.call_tool(name, args)
        calls.append({"tool": name, "is_error": bool(response.is_error)})
        if response.is_error:
            raise ValueError(f"MCP {name} failed: {str(response.content)[:300]}")
        return response.structured_content

    async with asyncio.timeout(180):
        async with Client(create_server(store, library, discovery, nemotron_key=key)) as client:
            opened = await call(client, "open_paper", {"arxiv_id": PAPER})
            if opened["status"] != "ok":
                source = opened.get("source", "public source")
                detail = opened.get("detail", opened["status"])
                raise ValueError(f"Public paper was unavailable ({source}): {detail}")
            found = await call(client, "find_evidence", {"query":
                "executable Python code actions revise observations multi-turn",
                "arxiv_ids": [PAPER], "top_k": 3})
            if not found["evidence"]:
                raise ValueError("No relevant paper passage found")
            span = found["evidence"][0]
            excerpt = await call(client, "get_passage", {"arxiv_id": PAPER,
                "start": span["start"], "length": min(900, span["end"] - span["start"])})
            if not excerpt["text"].strip():
                raise ValueError("Selected paper passage was empty")
            saved = await call(client, "save_experiment_brief", {
                "brief": {"question": QUESTION, "unknowns": [
                    "The paper's reported gains may not transfer to our document task."]}})
            experiment_id = saved["experiment_id"]
            note = {"note_id": "codeact_public_paper", "question": QUESTION,
                "findings": [{"claim": "CodeAct proposes executable Python actions that can "
                              "adapt after observations; applicability here remains untested.",
                              "reference_ids": ["codeact_span"]}],
                "references": [{"kind": "paper_span", "reference_id": "codeact_span",
                                "arxiv_id": PAPER, "start": excerpt["start"],
                                "end": excerpt["end"], "quote": excerpt["text"]}],
                "uncertainties": ["Measure execution errors, answer support, latency and "
                                  "cost against the same tasks before changing the harness."],
                "proposed_test": "Compare code actions and fixed tool calls on matched tasks.",
                "origin": "host_research", "data_kind": "recorded"}
            recorded = await call(client, "record_experiment_research", {
                "experiment_id": experiment_id, "expected_revision": 1, "note": note})
            related = await call(client, "find_related_experiments", {
                "question": "Python code actions for document agents"})
            resumed = await call(client, "resume_experiment", {
                "experiment_id": experiment_id})
            if related["matches"][0]["experiment_id"] != experiment_id:
                raise ValueError("Related-work lookup did not return this record")
            packet = public_paper_packet(recorded)
            # A UTF-8 byte count upper-bounds ordinary text token count. Tenfold
            # pricing headroom protects this one smoke request against modest changes.
            input_bytes = len(json.dumps(packet, ensure_ascii=False).encode()) + len(
                EXPLAINER_PROMPT.encode())
            tenfold_upper_usd = 10 * (input_bytes * PRICE_INPUT_PER_M
                                      + MAX_OUTPUT_TOKENS * PRICE_OUTPUT_PER_M) / 1_000_000
            if tenfold_upper_usd > MAX_SMOKE_USD:
                raise ValueError("Nebius smoke would exceed its $0.50 budget")
            explanation = None
            explanation_error = None
            if live_nebius:
                try:
                    explanation = await call(client, "explain_experiment_research", {
                        "experiment_id": experiment_id, "note_id": "codeact_public_paper"})
                except ValueError as exc:
                    explanation_error = str(exc)[:500]
    report = {"experiment_id": experiment_id, "paper": PAPER,
        "paper_source_url": packet["evidence"][0]["source_url"],
        "quote": excerpt["text"], "research_note_sha256": packet["research_note_sha256"],
        "related_match_kind": related["match_kind"],
        "related_experiment_id": related["matches"][0]["experiment_id"],
        "resume": resumed, "mcp_calls": calls, "model": MODEL if live_nebius else None,
        "model_requests": int(live_nebius), "max_output_tokens": MAX_OUTPUT_TOKENS,
        "preflight_tenfold_price_upper_usd": round(tenfold_upper_usd, 6),
        "published_rates_usd_per_million_tokens": {
            "input": PRICE_INPUT_PER_M, "output": PRICE_OUTPUT_PER_M},
        "explanation": explanation,
        "explanation_error": explanation_error,
        "semantic_support": "not_reviewed",
        "warning": "Exact quotes verify provenance, not semantic support. "
                   "The paper does not establish a gain for this project's workload."}
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "mcp_calls": len(calls),
                      "model_requests": report["model_requests"],
                      "experiment_id": experiment_id}))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library-root", type=Path,
                        default=Path("~/.envoy/arxiv-library"))
    parser.add_argument("--live-nebius", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(args.output, args.library_root, live_nebius=args.live_nebius))


if __name__ == "__main__":
    main()
