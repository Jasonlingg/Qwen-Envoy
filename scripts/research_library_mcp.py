"""Read-only MCP access to one frozen public-paper research snapshot.

Search and source opening work without a model. An explicitly configured Qwen
endpoint adds a bounded executable-code investigation; model-authored Python
still runs only in the Qwen investigator's Docker sandbox. This server neither
uses the older JSON-action research agent nor writes Obsidian notes.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import threading
from pathlib import Path
from typing import Any
from uuid import uuid4

from mcp.server import MCPServer
from mcp.types import ToolAnnotations

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env.tools import SEARCH_PROTOCOL_VERSION, TOOL_PREAMBLE  # noqa: E402
from src.policies.code_execution import DEFAULT_MAX_TOKENS  # noqa: E402
from src.policies.openai_compatible import OpenAICompatiblePolicy  # noqa: E402
from src.product.packet_verifier import verify_investigation_packet  # noqa: E402
from src.product.qwen_investigator import (  # noqa: E402
    PUBLIC_PAPER_SYSTEM_PROMPT,
    QwenInvestigator,
)
from src.research.agent import load_snapshot  # noqa: E402
from src.research.tools_runtime import ResearchTools  # noqa: E402

MAX_REVIEWS = 100
MAX_SOURCE_CHARS = 8_000
MAX_QUESTION_CHARS = 4_000
READ_ONLY_LOCAL = ToolAnnotations(read_only_hint=True, open_world_hint=False)
READ_ONLY_MODEL = ToolAnnotations(read_only_hint=True, open_world_hint=True)
PAPER_PROMPT_HASH = hashlib.sha256(PUBLIC_PAPER_SYSTEM_PROMPT.encode()).hexdigest()
TOOL_PREAMBLE_HASH = hashlib.sha256(TOOL_PREAMBLE.encode()).hexdigest()


def _paper_snapshot(snapshot: Path) -> tuple[dict, dict]:
    """Verify corpus bytes and reject a vault import masquerading as papers."""
    manifest, docs = load_snapshot(snapshot)
    papers = manifest.get("papers")
    if (manifest.get("status") != "complete"
            or manifest.get("schema_version") != "research-snapshot-v1"
            or not isinstance(papers, list) or not papers
            or len(papers) != len(docs)
            or any(not isinstance(item, dict) for item in papers)
            or {item.get("doc_id") for item in papers} != set(docs)):
        raise ValueError("expected a complete frozen public-paper research snapshot")
    if any(
        not isinstance(docs[item["doc_id"]].get("metadata"), dict)
        or item.get("arxiv_id") != docs[item["doc_id"]]["metadata"].get("arxiv_id")
        or not item.get("arxiv_id")
        for item in papers
    ):
        raise ValueError("research manifest disagrees with a pinned paper ID")
    return manifest, docs


def _verified_investigation(packet: dict, manifest: dict, docs: dict) -> dict:
    """Return compact model output with every quoted span rebuilt from the corpus."""
    return verify_investigation_packet(
        packet, manifest, docs, allowed_prompt_sha256s=frozenset({PAPER_PROMPT_HASH}))


def create_server(snapshot: Path, investigator: Any | None = None) -> MCPServer:
    """Bind read-only operations to an immutable paper snapshot for this process."""
    snapshot = snapshot.expanduser().resolve(strict=True)
    if not snapshot.is_dir():
        raise ValueError("snapshot must be an existing directory")
    manifest, _ = _paper_snapshot(snapshot)
    corpus_hash = manifest["corpus_hash"]
    tools = ResearchTools(snapshot / "corpus")
    reviews: dict[str, frozenset[str]] = {}
    lock = threading.RLock()
    server = MCPServer(
        "Envoy Research Library",
        instructions=(
            "Read-only access to one frozen collection of original public papers. "
            "Lexical search works without a model; Qwen investigation runs only if "
            "the server was explicitly configured with an endpoint. Model answers "
            "are proposals. Exact excerpts establish provenance, not semantic support."
        ),
    )

    def current_docs() -> dict:
        current, docs = _paper_snapshot(snapshot)
        if current["corpus_hash"] != corpus_hash:
            raise ValueError("research snapshot changed since this MCP server started")
        return docs

    def remember(doc_ids: set[str]) -> str:
        review_id = uuid4().hex
        with lock:
            reviews[review_id] = frozenset(doc_ids)
            if len(reviews) > MAX_REVIEWS:
                reviews.pop(next(iter(reviews)))
        return review_id

    @server.tool(annotations=READ_ONLY_LOCAL)
    def search_research_papers(query: str, top_k: int = 5) -> dict[str, Any]:
        """Find exact passages in pinned public papers with lexical search."""
        if not isinstance(query, str) or not query.strip() or len(query) > MAX_QUESTION_CHARS:
            raise ValueError("query must contain 1..4000 characters")
        if type(top_k) is not int or not 1 <= top_k <= 5:
            raise ValueError("top_k must be 1..5")
        current_docs()
        hits = tools.search_papers(query, top_k=top_k)
        return {
            "status": "evidence_found" if hits else "no_evidence",
            "retriever": "lexical_paper_search",
            "corpus_hash": corpus_hash,
            "review_id": remember({hit["doc_id"] for hit in hits}),
            "evidence": hits,
            "warning": "Search relevance and exact excerpts do not establish claim support.",
        }

    @server.tool(annotations=READ_ONLY_MODEL)
    def investigate_research_question(question: str) -> dict[str, Any]:
        """Ask configured Qwen to search and inspect papers with sandboxed Python."""
        if (not isinstance(question, str) or not question.strip()
                or len(question) > MAX_QUESTION_CHARS):
            raise ValueError("question must contain 1..4000 characters")
        current_docs()
        if investigator is None:
            return {
                "status": "model_not_configured",
                "source_domain": "public_papers",
                "corpus_hash": corpus_hash,
                "candidate_answer": None,
                "evidence": [],
                "model_requests_attempted": 0,
                "warning": (
                    "No Qwen endpoint was configured; use paper search or "
                    "configure one explicitly."
                ),
            }
        packet = investigator.investigate(question.strip(), snapshot)
        docs = current_docs()
        result = _verified_investigation(packet, manifest, docs)
        result["review_id"] = remember({item["doc_id"] for item in result["evidence"]})
        return result

    @server.tool(annotations=READ_ONLY_LOCAL)
    def get_research_source(review_id: str, doc_id: str,
                            start: int = 0, length: int = 4_000) -> dict[str, Any]:
        """Open a bounded excerpt of a paper cited by this server's prior review."""
        if type(start) is not int or type(length) is not int:
            raise ValueError("start and length must be integers")
        if start < 0 or not 1 <= length <= MAX_SOURCE_CHARS:
            raise ValueError("start must be nonnegative and length must be 1..8000")
        with lock:
            allowed = reviews.get(review_id)
        if allowed is None or doc_id not in allowed:
            raise ValueError("paper was not returned in this review; search again")
        docs = current_docs()
        doc = docs[doc_id]
        if start > len(doc["text"]):
            raise ValueError("start exceeds paper length")
        end = min(start + length, len(doc["text"]))
        return {
            "review_id": review_id,
            "corpus_hash": corpus_hash,
            "doc_id": doc_id,
            "title": doc["title"],
            "arxiv_id": doc["metadata"]["arxiv_id"],
            "source_url": doc["metadata"].get("source_url"),
            "coverage": doc["metadata"].get("coverage"),
            "start": start, "end": end,
            "total_chars": len(doc["text"]),
            "text": doc["text"][start:end],
        }

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path,
                        help="Complete frozen public-paper snapshot")
    parser.add_argument("--qwen-endpoint", help="Explicit OpenAI-compatible Qwen endpoint")
    parser.add_argument("--qwen-model", help="Served model identifier")
    parser.add_argument("--qwen-checkpoint", help="Exact checkpoint identifier for the trace")
    args = parser.parse_args()
    supplied = (args.qwen_endpoint, args.qwen_model, args.qwen_checkpoint)
    if any(supplied) and not all(supplied):
        parser.error("Qwen requires --qwen-endpoint, --qwen-model, and --qwen-checkpoint")
    investigator = None
    if all(supplied):
        investigator = QwenInvestigator(
            policy_factory=lambda: OpenAICompatiblePolicy(
                endpoint=args.qwen_endpoint,
                model=args.qwen_model,
                api_key=os.environ.get("ENVOY_MODEL_API_KEY"),
                system_prompt=PUBLIC_PAPER_SYSTEM_PROMPT,
                max_tokens=DEFAULT_MAX_TOKENS,
                temperature=0.0,
                extra_body={"chat_template_kwargs": {"enable_thinking": False},
                            "top_p": 1.0, "seed": 42},
            ),
            model_identity={
                "checkpoint": args.qwen_checkpoint,
                "served_model": args.qwen_model,
                "endpoint": args.qwen_endpoint,
                "temperature": 0.0,
                "top_p": 1.0,
                "seed": 42,
                "max_tokens": DEFAULT_MAX_TOKENS,
                "thinking": False,
            },
            source_domain="public_papers",
        )
    create_server(args.snapshot, investigator=investigator).run(transport="stdio")


if __name__ == "__main__":
    main()
