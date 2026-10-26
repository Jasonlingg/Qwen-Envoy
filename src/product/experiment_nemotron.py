"""One bounded Nemotron explanation of a selected public-paper research note."""

from __future__ import annotations

import json

from src.policies.bounded_http import bounded_transport
from src.research.explainer import EXPLAINER_PROMPT, check_explanation, parse_explanation

ENDPOINT = "https://api.tokenfactory.nebius.com/v1/chat/completions"
MODEL = "nvidia/Nemotron-3_5-Lightning"
MAX_PACKET_BYTES = 12_000
MAX_OUTPUT_TOKENS = 1_800


def public_paper_packet(view: dict) -> dict:
    selected = view["selected_research_note"]
    if not selected:
        raise ValueError("Select an existing research note")
    note = selected["note"]
    if note["data_kind"] != "recorded":
        raise ValueError("Nemotron explanation requires recorded, non-synthetic research")
    receipts = [receipt for receipt in selected["evidence_receipts"]
                if receipt["verification"] == "paper_span_verified"]
    if not receipts:
        raise ValueError("Research note has no verified public-paper spans")
    evidence = []
    for receipt in receipts[:5]:
        value = receipt["value"]
        metadata = receipt["metadata"]
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Verified paper span has no text")
        evidence.append({"evidence_id": f"E{len(evidence) + 1}",
            "reference_id": receipt["reference_id"],
            "quote": value[:1200], "quote_truncated": len(value) > 1200,
            "source_url": metadata["source_url"],
            "coverage": metadata["coverage"],
            "source_sha256": receipt["source_sha256"]})
    by_ref = {item["reference_id"]: item["evidence_id"] for item in evidence}
    proposals = [{"text": finding["claim"][:1000],
                  "evidence_ids": [by_ref[r] for r in finding["reference_ids"] if r in by_ref]}
                 for finding in note["findings"]]
    packet = {"schema_version": "experiment-nemotron-packet-v1",
        "experiment_id": view["experiment_id"], "research_note_id": note["note_id"],
        "research_note_sha256": selected["sha256"], "question": note["question"][:500],
        "candidate_claims": proposals, "evidence": evidence,
        "uncertainties": [value[:500] for value in note["uncertainties"][:5]],
        "warning": ("Saved findings are unreviewed proposals. Exact quotes prove "
                    "provenance, not support.")}
    if len(json.dumps(packet, ensure_ascii=False).encode()) > MAX_PACKET_BYTES:
        raise ValueError("Selected note exceeds the 12 KB Nemotron request budget")
    return packet


def explain_public_research(store, experiment_id: str, note_id: str, api_key: str,
                            *, transport=bounded_transport) -> dict:
    """Make exactly one capped request; do not save or trust the model's judgment."""
    if not api_key:
        raise ValueError("NEBIUS_API_KEY is required")
    packet = public_paper_packet(store.get(experiment_id, research_note_id=note_id))
    payload = {"model": MODEL, "max_tokens": MAX_OUTPUT_TOKENS, "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "system", "content": EXPLAINER_PROMPT},
                     {"role": "user", "content": json.dumps(packet, ensure_ascii=False)}]}
    # One transport invocation, with no redirects or automatic retries.
    response = transport(ENDPOINT, payload,
                         {"Authorization": f"Bearer {api_key}",
                          "Content-Type": "application/json"}, 60.0)
    try:
        choice = response["choices"][0]
        if choice.get("finish_reason") == "length":
            raise ValueError("Nemotron response reached its output-token cap")
        explanation = parse_explanation(choice["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Nemotron response has no usable explanation") from exc
    checks = check_explanation(explanation, packet)
    if len(explanation["claims"]) > 8 or len(explanation["answer"]) > 4000:
        raise ValueError("Nemotron explanation exceeds the response budget")
    usage = response.get("usage")
    return {"experiment_id": experiment_id, "research_note_id": note_id,
        "provider": "nebius_token_factory", "model": MODEL,
        "max_output_tokens": MAX_OUTPUT_TOKENS, "request_count": 1,
        "input_packet": packet, "explanation": explanation, "checks": checks,
        "usage": usage if isinstance(usage, dict) else None,
        "semantic_support": "not_reviewed",
        "warning": "References are mechanically valid; a person must review claim-level support."}
