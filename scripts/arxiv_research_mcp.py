"""MCP server: research live arXiv papers and return verified exact quotes.

Search, citations, opening papers, evidence search, and passages need no model.
`investigate` runs one configured reader per paper in a Docker sandbox; an
optional reasoner may answer, but only by quoting the verified evidence.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from mcp.server import MCPServer
from mcp.types import ToolAnnotations

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.product.arxiv_library import ArxivLibrary  # noqa: E402
from src.product.paper_discovery import PaperDiscovery  # noqa: E402

from src.product.arxiv_mcp_config import (  # noqa: E402
    DEFAULT_NEMOTRON,
    build_reader,
    build_reasoner,
)
from src.product.arxiv_service import investigate as run_investigation  # noqa: E402
from src.product.evidence_search import find_evidence as search_evidence  # noqa: E402
from src.product.net import SourceUnavailable  # noqa: E402
from src.product.parallel_readers import WARNING  # noqa: E402

LOCAL = ToolAnnotations(read_only_hint=True, open_world_hint=False)
REMOTE = ToolAnnotations(read_only_hint=True, open_world_hint=True)
COVERAGE_NOTE = ("HTML/PDF text extraction omits figures and tables and may distort "
                 "equations; abstract_only means only the abstract was available.")


def _unavailable(exc: SourceUnavailable) -> dict[str, Any]:
    return {"status": "source_unavailable", "source": exc.source,
            "detail": exc.detail, "warning": WARNING}


def create_server(library, discovery, *, investigator=None, reasoner=None) -> MCPServer:
    server = MCPServer("Envoy arXiv Research", instructions=(
        "Research arXiv papers. search_papers and get_citations find papers; open_paper "
        "fetches one; find_evidence and get_passage return exact text with offsets; "
        "investigate dispatches a reader per paper and returns verified quotes. Quotes prove "
        "provenance, not support: reason over them before answering."))

    @server.tool(annotations=REMOTE)
    def search_papers(query: str, since: str | None = None, until: str | None = None,
                      limit: int = 10) -> dict[str, Any]:
        """Find papers by relevance (Semantic Scholar), or by date when since/until is set."""
        try:
            return {"status": "ok", **discovery.search(query, since, until, limit),
                    "warning": WARNING}
        except SourceUnavailable as exc:
            return _unavailable(exc)

    @server.tool(annotations=REMOTE)
    def get_citations(arxiv_id: str, direction: str = "references",
                      limit: int = 20) -> dict[str, Any]:
        """List a paper's references or the papers citing it (direction=cited_by)."""
        try:
            return {"status": "ok", **discovery.citations(arxiv_id, direction, limit),
                    "warning": WARNING}
        except SourceUnavailable as exc:
            return _unavailable(exc)

    @server.tool(annotations=REMOTE)
    def open_paper(arxiv_id: str) -> dict[str, Any]:
        """Fetch and pin one arXiv paper; returns coverage and a section outline."""
        try:
            return {"status": "ok", **library.open(arxiv_id), "coverage_note": COVERAGE_NOTE,
                    "warning": WARNING}
        except SourceUnavailable as exc:
            return _unavailable(exc)

    @server.tool(annotations=LOCAL)
    def find_evidence(query: str, arxiv_ids: list[str], top_k: int = 8) -> dict[str, Any]:
        """Rank exact passages across opened papers (model-free)."""
        return search_evidence(library, query, arxiv_ids, top_k)

    @server.tool(annotations=LOCAL)
    def get_passage(arxiv_id: str, start: int = 0, length: int = 4_000) -> dict[str, Any]:
        """Return a bounded exact excerpt of an opened paper (length 1..8000)."""
        if type(start) is not int or type(length) is not int:
            raise ValueError("start and length must be integers")
        if start < 0 or not 1 <= length <= 8_000:
            raise ValueError("start must be nonnegative and length must be 1..8000")
        document = library.document(arxiv_id)
        if start > len(document["text"]):
            raise ValueError("start exceeds paper length")
        end = min(start + length, len(document["text"]))
        return {"arxiv_id": document["metadata"]["arxiv_id"], "start": start, "end": end,
                "total_chars": len(document["text"]), "text": document["text"][start:end],
                "warning": WARNING}

    @server.tool(annotations=REMOTE)
    def investigate(question: str, arxiv_ids: list[str],
                    synthesize: bool = False) -> dict[str, Any]:
        """Run one reader per paper (max 5) and return merged verified quotes."""
        return run_investigation(library, investigator, reasoner, question, arxiv_ids, synthesize)

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library-root", type=Path, default=Path("~/.envoy/arxiv-library"))
    parser.add_argument("--reader", choices=["none", "qwen", "nemotron"], default="none",
                        help="Explicit model opt-in; default none makes no model calls")
    parser.add_argument("--qwen-endpoint")
    parser.add_argument("--qwen-model")
    parser.add_argument("--qwen-checkpoint")
    parser.add_argument("--nemotron-model", default=DEFAULT_NEMOTRON)
    parser.add_argument("--reasoner", choices=["none", "nemotron"], default="none")
    parser.add_argument("--harness", choices=["v1", "v2"], default="v2")
    args = parser.parse_args()
    reader = args.reader
    root = args.library_root.expanduser()
    try:
        investigator = build_reader(reader, qwen_endpoint=args.qwen_endpoint,
                                    qwen_model=args.qwen_model,
                                    qwen_checkpoint=args.qwen_checkpoint,
                                    nemotron_model=args.nemotron_model,
                                    harness_version=args.harness)
        reasoner = build_reasoner(args.reasoner, args.nemotron_model)
    except ValueError as exc:
        parser.error(str(exc))
    create_server(ArxivLibrary(root), PaperDiscovery(root / "discovery-cache"),
                  investigator=investigator, reasoner=reasoner).run(transport="stdio")


if __name__ == "__main__":
    main()
