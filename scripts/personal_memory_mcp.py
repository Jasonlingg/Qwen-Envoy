"""Read-only stdio MCP access to one frozen personal-vault snapshot.

The only model-callable operations search the selected snapshot and open a
source returned by that search. No tool accepts a path or modifies the vault.
"""

from __future__ import annotations

import argparse
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

from src.product.memory import inspect_evidence  # noqa: E402
from src.research.agent import load_snapshot  # noqa: E402

MAX_REVIEWS = 100
MAX_SOURCE_CHARS = 8_000
READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)


def create_server(snapshot: Path) -> MCPServer:
    """Bind two read-only tools to a validated snapshot for this process's life."""
    snapshot = snapshot.expanduser().resolve(strict=True)
    if not snapshot.is_dir():
        raise ValueError("snapshot must be an existing directory")
    manifest, _ = load_snapshot(snapshot)
    if manifest.get("status") != "complete":
        raise ValueError("snapshot import is incomplete; fix it before serving")
    corpus_hash = manifest["corpus_hash"]
    reviews: dict[str, frozenset[str]] = {}
    lock = threading.RLock()
    server = MCPServer(
        "Envoy Personal Memory",
        instructions=(
            "Read-only access to one frozen Obsidian snapshot. Search returns exact "
            "quotations and a review ID; source opening requires that review ID. "
            "Quotation validity establishes provenance, not semantic support."
        ),
    )

    @server.tool(annotations=READ_ONLY)
    def search_memory(query: str, top_k: int = 5) -> dict[str, Any]:
        """Find up to five exact note passages in the fixed snapshot; no model call."""
        if len(query) > 4_000:
            raise ValueError("query exceeds 4000 characters")
        with lock:
            review = inspect_evidence(snapshot, query, top_k)
            if review["corpus_hash"] != corpus_hash:
                raise ValueError("snapshot changed since this MCP server started")
            review_id = uuid4().hex
            reviews[review_id] = frozenset(
                item["doc_id"] for item in review["evidence"]
            )
            if len(reviews) > MAX_REVIEWS:
                reviews.pop(next(iter(reviews)))
            return {"review_id": review_id, **review}

    @server.tool(annotations=READ_ONLY)
    def get_memory_source(review_id: str, doc_id: str,
                          start: int = 0, length: int = 4_000) -> dict[str, Any]:
        """Open a bounded slice of a note cited in this process's search review."""
        if start < 0 or not 1 <= length <= MAX_SOURCE_CHARS:
            raise ValueError("start must be nonnegative and length must be 1..8000")
        with lock:
            allowed = reviews.get(review_id)
            if allowed is None:
                raise ValueError("unknown or expired review ID; search again")
            if doc_id not in allowed:
                raise ValueError("source was not returned in this review")
            current_manifest, docs = load_snapshot(snapshot)
            if current_manifest["corpus_hash"] != corpus_hash:
                raise ValueError("snapshot changed since this MCP server started")
            doc = docs[doc_id]
            full_text = doc["text"]
            if start > len(full_text):
                raise ValueError("start exceeds source length")
            end = min(start + length, len(full_text))
            return {
                "review_id": review_id,
                "corpus_hash": corpus_hash,
                "doc_id": doc_id,
                "title": doc["title"],
                "source_path": doc["metadata"].get("source_path"),
                "start": start,
                "end": end,
                "total_chars": len(full_text),
                "text": full_text[start:end],
            }

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True,
                        help="Existing complete snapshot to serve until restart")
    args = parser.parse_args()
    create_server(args.snapshot).run(transport="stdio")


if __name__ == "__main__":
    main()
