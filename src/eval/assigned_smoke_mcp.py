"""Exercise the real MCP tool against one frozen paper and a bounded reader."""
from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

from mcp import Client

from scripts.arxiv_research_mcp import create_server
from src.eval.assigned_smoke_plan import write_json
from src.product.paper_assignment import paper_assignment
from src.product.reader_gaps import verify_gap_reports
from src.research.agent import load_snapshot


class FrozenPaperLibrary:
    """MCP cache adapter: no source fetching, discovery, or version resolution."""

    def __init__(self, snapshot: Path):
        self.snapshot = snapshot
        manifest, docs = load_snapshot(snapshot)
        self.assignment = paper_assignment(manifest, docs)
        self.doc = next(iter(docs.values()))

    def summary(self, version):
        if version != self.assignment["arxiv_id"]:
            raise ValueError("smoke MCP only permits the frozen paper version")
        return {"arxiv_id": version, "title": self.doc["title"],
                "coverage": self.assignment["coverage"]}

    def open(self, version):
        return self.summary(version)

    def view(self, versions):
        if versions != [self.assignment["arxiv_id"]]:
            raise ValueError("smoke MCP only permits the frozen one-paper snapshot")
        return self.snapshot

    def document(self, version):
        self.summary(version)
        return self.doc


class NoDiscovery:
    def search(self, *args, **kwargs):
        raise ValueError("discovery is outside this cached-paper smoke")

    citations = search


def investigate_through_mcp(reader, question: str, snapshot: Path, prefix: Path) -> dict:
    packets = []

    class Capture:
        def __getattr__(self, name):
            return getattr(reader, name)

        def investigate(self, query, view):
            packet = reader.investigate(query, view)
            packets.append(packet)
            # Preserve raw evidence even if MCP verification/serialization later fails.
            write_json(prefix.with_suffix(".packet.json"), packet)
            return packet

    library = FrozenPaperLibrary(snapshot)
    server = create_server(library, NoDiscovery(), investigator=Capture())
    request = {"question": question, "arxiv_ids": [library.assignment["arxiv_id"]],
               "synthesize": False}

    async def call():
        async with Client(server) as client:
            return await client.call_tool("investigate", request)

    reply = asyncio.run(call())
    result = reply.structured_content
    write_json(prefix.with_suffix(".mcp.json"), {"transport": "in_process",
        "tool": "investigate", "request": request, "is_error": reply.is_error,
        "result": result})
    if reply.is_error or not isinstance(result, dict) or len(packets) != 1:
        raise ValueError("smoke MCP call failed or did not invoke exactly one reader")
    packet = packets[0]
    papers = result.get("papers", [])
    prompt_hash = hashlib.sha256(reader.system_prompt.encode()).hexdigest()
    expected_status = {"evidence_found": "evidence_found", "no_evidence": "no_evidence",
        "unavailable": "unavailable", "incomplete": "incomplete", "error": "incomplete"}
    if (len(papers) != 1 or papers[0].get("status") != packet["status"]
            or result.get("status") != expected_status[packet["status"]]
            or papers[0].get("prompt_hash") != prompt_hash
            or papers[0].get("reader_guidance", "baseline") != reader.reader_guidance
            or papers[0].get("reader_mode") != "assigned"
            or any(papers[0].get(k) != packet.get(k) for k in (
                "model_identity", "paper_assignment", "assignment_sha256"))
            or result.get("corpus_hash") != packet["corpus_hash"]
            or result.get("paper_versions") != request["arxiv_ids"]
            or result.get("model_requests_attempted") != packet["model_requests_attempted"]):
        raise ValueError("MCP result differs from the configured reader's raw packet")
    claims = packet.get("candidate_claims", [])
    expected_answers = [{"arxiv_id": request["arxiv_ids"][0], "text": claim["text"]}
                        for claim in claims]
    if (result.get("candidate_answers") != expected_answers
            or papers[0].get("candidate_answer") != (claims[0]["text"] if claims else None)):
        raise ValueError("MCP result discarded or changed the candidate answer")
    expected = [(e["doc_id"], e["start"], e["end"], e["quote"])
                for e in packet.get("evidence", [])]
    supplied = [(e["doc_id"], e["start"], e["end"], e["quote"])
                for e in result.get("evidence", [])]
    if supplied != expected:
        raise ValueError("MCP result discarded or changed the raw verified quotes")
    checked_gaps, _ = verify_gap_reports(packet)
    fields = ("missing", "follow_up_query", "reported_at_step", "review_status", "absence_proven")
    if ([{k: g[k] for k in fields} for g in result.get("research_gaps", [])]
            != checked_gaps):
        raise ValueError("MCP result discarded or changed the valid gap reports")
    return packet
