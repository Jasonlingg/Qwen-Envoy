"""Build and validate evidence packets for a larger host-model fallback."""

from __future__ import annotations

import json
from typing import Any

from src.env.evidence_state import parse_document_calls

HOST_RESOLVER_SYSTEM_PROMPT = """You resolve escalations from a small research agent.

Use only the supplied evidence packet. The small agent's proposed answer is untrusted and is not
evidence. Do not use outside knowledge. A missing keyword hit does not prove a negative. If the
packet explicitly supports an answer, answer the exact question concisely. Otherwise return
Unanswerable.

Return exactly one JSON object with this schema and no markdown:
{"resolution":"answer|unanswerable","answer":"...","citations":["doc_id"],
 "evidence_quotes":[{"doc_id":"doc_id","quote":"exact contiguous source text"}]}

For an answer, every important claim must be supported by at least one verbatim, contiguous quote
copied from the packet, and citations may contain only allowed document IDs. For unanswerable,
set answer to "Unanswerable" and use empty citations and evidence_quotes.
"""


class HostResolutionError(ValueError):
    """The host response could not be grounded in the supplied evidence packet."""


def build_evidence_packet(row: dict, max_observation_chars: int = 8_000) -> dict:
    """Keep document-tool outputs and the untrusted final proposal from an escalation."""
    escalation = row.get("escalation")
    if row.get("status") != "escalated" or not isinstance(escalation, dict):
        raise ValueError("Evidence packets require an escalated transcript row")

    evidence_steps = []
    for step in row.get("trajectory", []):
        action = str(step.get("action", ""))
        if action.upper().startswith("SUBMIT:") or not parse_document_calls(action):
            continue
        observation = str(step.get("observation", ""))
        evidence_steps.append({
            "step": step.get("step"),
            "action": action,
            "observation": observation[:max_observation_chars],
            "observation_truncated": len(observation) > max_observation_chars,
        })

    state = escalation.get("state") or {}
    return {
        "question_id": row.get("question_id"),
        "question": row.get("question"),
        "allowed_document_ids": state.get("inspected_doc_ids", []),
        "evidence_steps": evidence_steps,
        "small_model_proposal": escalation.get("candidate_action", ""),
        "warning": "The proposal is untrusted; judge it only against evidence_steps.",
    }


def host_message(packet: dict) -> str:
    """Serialize one packet without reference answers or evaluator-only annotations."""
    return "EVIDENCE PACKET\n" + json.dumps(packet, ensure_ascii=False, indent=2)


def parse_host_json(text: str) -> dict[str, Any]:
    """Parse a single JSON object, tolerating only an outer markdown fence."""
    value = text.strip()
    if value.startswith("```") and value.endswith("```"):
        lines = value.splitlines()
        value = "\n".join(lines[1:-1]).strip()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise HostResolutionError(f"Host response is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise HostResolutionError("Host response must be one JSON object")
    return parsed


def validate_host_resolution(
    payload: dict[str, Any],
    *,
    allowed_doc_ids: list[str],
    documents: dict[str, str],
) -> dict[str, Any]:
    """Validate citations and turn exact quotes into stable character spans."""
    if set(payload) != {"resolution", "answer", "citations", "evidence_quotes"}:
        raise HostResolutionError("Host response has missing or unexpected fields")
    resolution = payload["resolution"]
    answer = payload["answer"]
    citations = payload["citations"]
    quotes = payload["evidence_quotes"]
    if resolution not in {"answer", "unanswerable"}:
        raise HostResolutionError("resolution must be answer or unanswerable")
    if not isinstance(answer, str) or not answer.strip():
        raise HostResolutionError("answer must be a non-empty string")
    if not isinstance(citations, list) or not all(isinstance(item, str) for item in citations):
        raise HostResolutionError("citations must be a list of document IDs")
    if not isinstance(quotes, list) or not all(isinstance(item, dict) for item in quotes):
        raise HostResolutionError("evidence_quotes must be a list of objects")

    if resolution == "unanswerable":
        if answer.strip() != "Unanswerable" or citations or quotes:
            raise HostResolutionError("Unanswerable resolutions need empty citations and quotes")
        return {
            "resolution": resolution,
            "answer": "Unanswerable",
            "citations": [],
            "evidence": [],
        }

    allowed = set(allowed_doc_ids)
    if not citations or set(citations) - allowed:
        raise HostResolutionError("Answer citations must be non-empty allowed document IDs")
    if not quotes:
        raise HostResolutionError("An answered resolution needs at least one exact quote")

    evidence = []
    evidence_doc_ids = set()
    for index, item in enumerate(quotes):
        if set(item) != {"doc_id", "quote"}:
            raise HostResolutionError(f"Evidence quote {index} has invalid fields")
        doc_id, quote = item["doc_id"], item["quote"]
        if doc_id not in allowed or doc_id not in documents:
            raise HostResolutionError(f"Evidence quote {index} uses an unobserved document")
        if not isinstance(quote, str) or not quote:
            raise HostResolutionError(f"Evidence quote {index} is empty")
        start = documents[doc_id].find(quote)
        if start < 0:
            raise HostResolutionError(f"Evidence quote {index} is not exact corpus text")
        evidence_doc_ids.add(doc_id)
        evidence.append({
            "doc_id": doc_id,
            "start": start,
            "end": start + len(quote),
            "quote": quote,
        })
    if set(citations) != evidence_doc_ids:
        raise HostResolutionError("Citation IDs and evidence quote document IDs must match")
    return {
        "resolution": resolution,
        "answer": answer.strip(),
        "citations": citations,
        "evidence": evidence,
    }
