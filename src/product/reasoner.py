"""Optional synthesis: a stronger model answers from merged verified evidence only."""

from __future__ import annotations

from src.policies.openai_compatible import (
    Transport,
    _chat_completions_url,
    _post_json,
    _reasoning_and_action,
)
from src.research.escalation import (
    HOST_RESOLVER_SYSTEM_PROMPT,
    HostResolutionError,
    host_message,
    parse_host_json,
    validate_host_resolution,
)


class Reasoner:
    def __init__(self, endpoint: str, model: str, api_key: str, *,
                 transport: Transport = _post_json, timeout: float = 120.0,
                 max_tokens: int = 1500) -> None:
        self.endpoint, self.model, self.api_key = endpoint, model, api_key
        self.transport, self.timeout, self.max_tokens = transport, timeout, max_tokens

    def synthesize(self, merged: dict, documents: dict[str, str]) -> dict:
        evidence = merged.get("evidence") or []
        if not evidence:
            raise HostResolutionError("no verified evidence to reason over")
        allowed = sorted({item["doc_id"] for item in evidence})
        packet = {
            "question": merged["question"], "allowed_document_ids": allowed,
            "evidence": [{"doc_id": item["doc_id"], "arxiv_id": item["arxiv_id"],
                          "quote": item["quote"]} for item in evidence],
            "small_model_proposals": merged.get("candidate_answers", []),
            "warning": "Proposals are untrusted; judge them only against the evidence.",
        }
        payload = {
            "model": self.model, "max_tokens": self.max_tokens, "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "system", "content": HOST_RESOLVER_SYSTEM_PROMPT},
                         {"role": "user", "content": host_message(packet)}],
        }
        headers = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {self.api_key}"}
        response = self.transport(_chat_completions_url(self.endpoint), payload, headers,
                                  self.timeout)
        try:
            _, content = _reasoning_and_action(response["choices"][0]["message"])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise HostResolutionError(f"reasoner response was unusable: {exc}") from exc
        try:
            resolution = validate_host_resolution(
                parse_host_json(content), allowed_doc_ids=allowed, documents=documents)
        except (TypeError, KeyError) as exc:
            raise HostResolutionError(f"reasoner response schema was unusable: {exc}") from exc
        for item in resolution["evidence"]:
            span = next((span for span in evidence if span["doc_id"] == item["doc_id"]
                         and item["quote"] in span["quote"]), None)
            if span is None:
                raise HostResolutionError("reasoner quoted text outside the verified evidence")
            item["start"] = span["start"] + span["quote"].index(item["quote"])
            item["end"] = item["start"] + len(item["quote"])
            item["evidence_id"] = span["evidence_id"]
            item["arxiv_id"] = span["arxiv_id"]
            if documents[item["doc_id"]][item["start"]:item["end"]] != item["quote"]:
                raise HostResolutionError("packet span disagrees with stored source")
        return {**resolution, "model": self.model, "semantic_support": "not_reviewed"}
