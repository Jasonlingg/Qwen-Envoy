"""Verify an investigation packet against the pinned paper snapshot.

Shared by the frozen-library MCP and the arXiv MCP. Every quoted span is rebuilt
from the stored paper text; a packet that fails any check is rejected whole.
"""

from __future__ import annotations

import hashlib

from src.env.tools import SEARCH_PROTOCOL_VERSION, TOOL_PREAMBLE
from src.env.tools_v2 import HARNESS_V2_SHA256
from src.product.assigned_guidance import BASELINE, FOCUSED, validate_guidance
from src.product.paper_assignment import verify_assignment
from src.product.reader_gaps import verify_gap_reports

MAX_EVIDENCE_SPANS = 5
MAX_SPAN_CHARS = 1_200
TOOL_PREAMBLE_SHA256 = "233adfef3cd72c8b8221e45dcaa58c76809f36bd717d1d342514a7c332c3dd71"
_REQUEST_LIMITS = {"v1": 20, "v2": 60}
_STATUSES = {"evidence_found", "no_evidence", "incomplete", "unavailable", "error"}


def _check_header(packet: dict, manifest: dict, allowed_prompt_sha256s: frozenset[str]) -> str:
    if not isinstance(packet, dict) or packet.get("corpus_hash") != manifest["corpus_hash"]:
        raise ValueError("Qwen packet does not match the pinned research snapshot")
    if packet.get("source_domain") != "public_papers":
        raise ValueError("Qwen packet did not use the public-paper task")
    prompt_hash = packet.get("system_prompt_sha256")
    if not isinstance(prompt_hash, str) or prompt_hash not in allowed_prompt_sha256s:
        raise ValueError("Qwen packet did not use the pinned public-paper prompt")
    if (hashlib.sha256(TOOL_PREAMBLE.encode()).hexdigest() != TOOL_PREAMBLE_SHA256
            or packet.get("tool_search_version") != SEARCH_PROTOCOL_VERSION
            or packet.get("tool_preamble_sha256") != TOOL_PREAMBLE_SHA256):
        raise ValueError("Qwen packet did not use the pinned document tools")
    harness = packet.get("harness_version", "v1")
    if not isinstance(harness, str) or harness not in _REQUEST_LIMITS or (
            harness == "v2" and packet.get("harness_shim_sha256") != HARNESS_V2_SHA256):
        raise ValueError("Qwen packet used an unknown or modified harness")
    return harness


def verify_investigation_packet(packet: dict, manifest: dict, docs: dict, *,
                                allowed_prompt_sha256s: frozenset[str],
                                expected_reader_mode: str = "discovery",
                                expected_reader_guidance: str = BASELINE) -> dict:
    """Return compact model output with every quoted span rebuilt from the corpus."""
    harness = _check_header(packet, manifest, allowed_prompt_sha256s)
    mode = packet.get("reader_mode", "discovery")
    if not isinstance(mode, str) or mode not in {"discovery", "assigned"}:
        raise ValueError("Qwen packet used an unknown reader mode")
    if mode != expected_reader_mode:
        raise ValueError("packet reader mode differs from the configured reader")
    guidance = packet.get("reader_guidance", BASELINE)
    validate_guidance(guidance, mode)
    if guidance != expected_reader_guidance:
        raise ValueError("packet reader guidance differs from the configured reader")
    assignment = None
    if mode == "assigned":
        if harness != "v2":
            raise ValueError("assigned reader requires harness v2")
        assignment = verify_assignment(packet, manifest, docs)
    status = packet.get("status")
    if not isinstance(status, str) or status not in _STATUSES:
        raise ValueError("Qwen packet has an unknown status")
    supplied = packet.get("evidence")
    if not isinstance(supplied, list) or len(supplied) > MAX_EVIDENCE_SPANS:
        raise ValueError("Qwen packet has too many evidence spans")
    if (status == "evidence_found") != bool(supplied):
        raise ValueError("Qwen packet status and evidence disagree")

    evidence = []
    for index, item in enumerate(supplied, 1):
        if not isinstance(item, dict):
            raise ValueError("Qwen evidence span is not an object")
        doc_id, start, end = item.get("doc_id"), item.get("start"), item.get("end")
        doc = docs.get(doc_id) if isinstance(doc_id, str) else None
        if (doc is None or type(start) is not int or type(end) is not int
                or not 0 <= start < end <= len(doc["text"])
                or end - start > MAX_SPAN_CHARS
                or item.get("quote") != doc["text"][start:end]):
            raise ValueError("Qwen evidence is not an exact bounded paper span")
        metadata = doc["metadata"]
        evidence.append({
            "evidence_id": f"E{index}", "doc_id": doc_id, "title": doc["title"],
            "start": start, "end": end, "quote": doc["text"][start:end],
            "arxiv_id": metadata["arxiv_id"], "source_url": metadata.get("source_url"),
            "coverage": metadata.get("coverage"),
        })

    claims = packet.get("candidate_claims")
    if not isinstance(claims, list) or (status == "evidence_found") != bool(claims):
        raise ValueError("Qwen packet status and candidate claims disagree")
    if len(claims) > 1:
        raise ValueError("Qwen packet has too many candidate claims")
    answer = None
    if claims:
        answer = claims[0].get("text") if isinstance(claims[0], dict) else None
        if not isinstance(answer, str) or not answer.strip() or len(answer) > 2_500:
            raise ValueError("Qwen candidate answer is invalid")

    trajectory = packet.get("trajectory", [])
    if not isinstance(trajectory, list) or len(trajectory) > 20:
        raise ValueError("Qwen packet has an invalid trajectory length")
    attempts = packet.get("model_requests_attempted", 0)
    if type(attempts) is not int or not 0 <= attempts <= _REQUEST_LIMITS[harness]:
        raise ValueError("Qwen packet has an invalid request count")
    retriever = ("nemotron_code_execution" if packet.get("retriever") == "nemotron_investigator"
                 else "qwen_code_execution")
    result = {
        "status": status, "source_domain": "public_papers", "retriever": retriever,
        "corpus_hash": manifest["corpus_hash"], "candidate_answer": answer,
        "evidence": evidence, "semantic_support": "not_reviewed",
        "warning": "Exact excerpts prove provenance, not support for the model's proposed answer.",
        "uncertainty": packet.get("uncertainty") if status == "no_evidence" else None,
        "model_identity": packet.get("model_identity"),
        "model_requests_attempted": attempts, "trajectory_steps": len(trajectory),
        "prompt_hash": packet["system_prompt_sha256"],
        "tool_search_version": SEARCH_PROTOCOL_VERSION,
        "tool_preamble_sha256": TOOL_PREAMBLE_SHA256, "harness_version": harness,
    }
    if assignment is not None:
        result.update(reader_mode="assigned", paper_assignment=assignment,
                      assignment_sha256=packet["assignment_sha256"])
        result["research_gaps"], result["gap_report_error"] = verify_gap_reports(packet)
        if guidance == FOCUSED and status == "no_evidence" and not result["research_gaps"]:
            raise ValueError("focused-v1 abstention requires a verified gap report")
        if guidance != BASELINE:
            result["reader_guidance"] = guidance
    return result
