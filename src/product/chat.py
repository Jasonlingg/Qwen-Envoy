"""Source-bound evidence review for the research-library development runner.

The historical chat application is archived in Git. This compatibility helper
checks quotation provenance; it does not judge semantic support.
"""

import re
from pathlib import Path

from src.product.memory import REVIEW_VERSION, _record_context
from src.research.agent import load_snapshot

_EVIDENCE_ID = re.compile(r"E[1-5]\Z")
_MAX_QUOTE_CHARS = 1_200


def _verified_review(snapshot: Path, packet: dict, query: str) -> dict:
    """Replace model-supplied metadata with frozen-source values or reject it."""
    manifest, docs = load_snapshot(snapshot)
    if not isinstance(packet, dict) or packet.get("corpus_hash") != manifest["corpus_hash"]:
        raise ValueError("investigator packet does not match the selected snapshot")
    items = packet.get("evidence")
    if not isinstance(items, list) or len(items) > 5:
        raise ValueError("investigator packet must have at most five source spans")
    if packet.get("status") == "evidence_found" and not items:
        raise ValueError("investigator reported evidence without source spans")
    if packet.get("status") == "no_evidence" and items:
        raise ValueError("investigator reported no evidence with source spans")
    evidence = []
    used: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("evidence span must be an object")
        evidence_id = item.get("evidence_id")
        doc = docs.get(item.get("doc_id"))
        start, end, quote = item.get("start"), item.get("end"), item.get("quote")
        if (not isinstance(evidence_id, str)
                or not _EVIDENCE_ID.fullmatch(evidence_id) or evidence_id in used):
            raise ValueError("evidence IDs must be distinct E1–E5 references")
        if (doc is None or type(start) is not int or type(end) is not int
                or not 0 <= start < end <= len(doc["text"])
                or not isinstance(quote, str) or doc["text"][start:end] != quote):
            raise ValueError(f"evidence {evidence_id} is not an exact snapshot span")
        used.add(evidence_id)
        truncated = len(quote) > _MAX_QUOTE_CHARS
        if truncated:
            quote = quote[:_MAX_QUOTE_CHARS]
            end = start + len(quote)
        evidence.append({
            "evidence_id": evidence_id,
            "doc_id": item["doc_id"],
            "title": doc["title"],
            "start": start,
            "end": end,
            "quote": quote,
            "source_path": doc["metadata"].get("source_path"),
            "source_kind": doc["metadata"].get("source_kind"),
            "record": _record_context(doc),
            "truncated_for_model": truncated,
        })
    return {
        "schema_version": REVIEW_VERSION,
        "retriever": packet.get("retriever", "qwen_investigator"),
        "status": "evidence_found" if evidence else "no_evidence",
        "query": query,
        "corpus_hash": manifest["corpus_hash"],
        "evidence": evidence,
        "warning": "Exact quotations prove provenance, not semantic support.",
        "uncertainty": packet.get("uncertainty") if not evidence else None,
    }
