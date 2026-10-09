"""Chat orchestration over a frozen vault; evidence is checked before model use."""

from __future__ import annotations

import re
from pathlib import Path

from src.product.graph import build_graph, one_hop_neighbors
from src.product.memory import REVIEW_VERSION, _record_context, inspect_evidence
from src.research.agent import load_snapshot

CHAT_VERSION = "envoy-chat-v1"
_EVIDENCE_ID = re.compile(r"E[1-5]\Z")
_MAX_QUESTION_CHARS = 4_000
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


def _related_notes(snapshot: Path, evidence: list[dict]) -> list[dict]:
    if not evidence:
        return []
    graph = build_graph(snapshot)
    cited_docs = {item["doc_id"] for item in evidence}
    related: dict[str, dict] = {}
    for item in evidence:
        for node in one_hop_neighbors(graph, item["doc_id"]):
            if node["doc_id"] not in cited_docs:
                related.setdefault(node["doc_id"], node)
    return list(related.values())[:12]


class ChatService:
    """One chat turn; an optional Nemotron coordinator calls an optional Qwen worker.

    This service never writes the vault. The web layer retains its separate
    explicit approval endpoint for any durable memory revision.
    """

    def __init__(self, *, coordinator=None, investigator=None) -> None:
        self.coordinator = coordinator
        self.investigator = investigator

    def reply(self, question: str, snapshot: Path,
              history: list[dict] | None = None) -> dict:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be non-empty text")
        if len(question) > _MAX_QUESTION_CHARS:
            raise ValueError(f"question exceeds {_MAX_QUESTION_CHARS} characters")
        question = question.strip()
        history = [] if history is None else history
        manifest, _ = load_snapshot(snapshot)
        route = {"investigate": True, "query": question, "source": "offline_default"}
        nemotron_calls = 0
        qwen_calls = 0
        coordinator_error = None
        if self.coordinator is not None:
            try:
                nemotron_calls += 1
                planned = self.coordinator.plan(question, history=history)
                if (not isinstance(planned, dict)
                        or type(planned.get("investigate")) is not bool
                        or not isinstance(planned.get("query"), str)):
                    raise ValueError("coordinator returned an invalid route")
                route = {**planned, "source": "nemotron"}
            except Exception as exc:
                coordinator_error = f"Nemotron routing unavailable ({type(exc).__name__})."

        worker_status = "not_configured"
        worker_identity = None
        retrieval_mode = "not_requested"
        if route["investigate"]:
            query = route["query"]
            if self.investigator is not None and coordinator_error is None:
                try:
                    qwen_calls = 1  # A raised worker call may already have contacted its model.
                    packet = self.investigator.investigate(query, snapshot)
                    worker_status = packet["status"]
                    worker_identity = packet.get("model_identity")
                    qwen_calls = packet.get(
                        "model_requests_attempted", len(packet.get("trajectory", [])),
                    )
                    if worker_status == "evidence_found":
                        review = _verified_review(snapshot, packet, query)
                        retrieval_mode = "qwen_code_execution"
                    elif worker_status == "no_evidence":
                        _verified_review(snapshot, packet, query)
                        baseline = inspect_evidence(snapshot, query)
                        if baseline["evidence"]:
                            review = baseline
                            retrieval_mode = "lexical_paragraph_baseline"
                        else:
                            review = _verified_review(snapshot, packet, query)
                            retrieval_mode = "qwen_code_execution"
                    else:
                        review = inspect_evidence(snapshot, query)
                        retrieval_mode = "lexical_paragraph_baseline"
                except (TypeError, ValueError, KeyError) as exc:
                    worker_status = "invalid_evidence"
                    coordinator_error = (
                        f"Qwen evidence rejected ({type(exc).__name__}). "
                        "Lexical retrieval was used instead."
                    ) if coordinator_error is None else coordinator_error
                    review = inspect_evidence(snapshot, query)
                    retrieval_mode = "lexical_paragraph_baseline"
                except Exception as exc:
                    worker_status = "error"
                    coordinator_error = (
                        f"Qwen investigation failed ({type(exc).__name__}). "
                        "Lexical retrieval was used instead."
                    ) if coordinator_error is None else coordinator_error
                    review = inspect_evidence(snapshot, query)
                    retrieval_mode = "lexical_paragraph_baseline"
            else:
                review = inspect_evidence(snapshot, query)
                retrieval_mode = "lexical_paragraph_baseline"
            # Validate the baseline too, so the model always sees bounded spans
            # from the same frozen snapshot, regardless of retrieval path.
            review = _verified_review(snapshot, review, query)
        else:
            review = _verified_review(snapshot, {
                "status": "no_evidence", "corpus_hash": manifest["corpus_hash"],
                "retriever": "not_requested", "evidence": [],
            }, question)

        answer = None
        claims: list[dict] = []
        limitations: list[str] = []
        checks = None
        answer_model = None
        answer_served_model = None
        answer_usage: dict = {}
        answer_status = "evidence_only"
        if self.coordinator is None:
            limitations.append("Nemotron is not configured; these are source passages only.")
        elif coordinator_error is not None and coordinator_error.startswith(
            "Nemotron routing unavailable"
        ):
            limitations.append(coordinator_error)
        else:
            # Qwen's action trace and proposed answer never enter the Nemotron
            # prompt. Only independently checked, compact passage fields do.
            answer_packet = {
                "schema_version": review["schema_version"],
                "corpus_hash": review["corpus_hash"],
                "retriever_protocol": retrieval_mode,
                "warning": review["warning"],
                "evidence": review["evidence"],
            }
            try:
                nemotron_calls += 1
                explanation = self.coordinator.answer(
                    question, answer_packet, history=history,
                )
                answer = explanation["answer"]
                claims = explanation["claims"]
                limitations = explanation["limitations"]
                checks = explanation["checks"]
                answer_model = explanation["model"]
                answer_served_model = explanation.get("served_model")
                answer_usage = explanation.get("usage", {})
                answer_status = "answered"
            except Exception as exc:
                limitations.append(
                    f"Nemotron answer unavailable ({type(exc).__name__}); "
                    "source passages are available for review.",
                )
        if coordinator_error and not coordinator_error.startswith(
            "Nemotron routing unavailable"
        ):
            limitations.append(coordinator_error)
        if not review["evidence"]:
            limitations.append("No verified source passages were found in this selected snapshot.")
        if worker_status == "no_evidence" and retrieval_mode == "lexical_paragraph_baseline":
            limitations.append(
                "Qwen abstained, but lexical retrieval found passages for independent review.",
            )
        return {
            **review,
            "schema_version": CHAT_VERSION,
            "question": question,
            "answer": answer,
            "answer_status": answer_status,
            "answer_model": answer_model,
            "answer_served_model": answer_served_model,
            "answer_usage": answer_usage,
            "claims": claims,
            "limitations": limitations,
            "checks": checks,
            "route": route,
            "retrieval": {
                "mode": retrieval_mode,
                "worker_status": worker_status,
                "model_identity": worker_identity,
                "warning": review["warning"],
            },
            "related_notes": _related_notes(snapshot, review["evidence"]),
            "model_requests_attempted": nemotron_calls + qwen_calls,
            "model_calls": nemotron_calls + qwen_calls,
        }
