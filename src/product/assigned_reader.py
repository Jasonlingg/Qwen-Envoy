"""Opt-in assigned-paper reader using the existing endpoint configuration."""
from __future__ import annotations

from src.product.arxiv_mcp_config import build_reader
from src.product.qwen_investigator import QwenInvestigator, effective_system_prompt


def build_assigned_reader(kind: str, **configuration) -> QwenInvestigator:
    configuration.setdefault("harness_version", "v2")
    if configuration["harness_version"] != "v2":
        raise ValueError("assigned reader requires harness v2")
    reader = build_reader(kind, **configuration)
    if reader is None:
        raise ValueError("assigned reader requires qwen or nemotron")
    prompt = effective_system_prompt("public_papers", reader.protocol_variant, "v2",
                                     reader_mode="assigned")

    def policy():
        # build_reader's factory creates a fresh policy. Replace its prompt
        # before reset/dispatch; the investigator and packet use the same bytes.
        instance = reader.policy_factory()
        instance.system_prompt = prompt
        return instance

    return QwenInvestigator(
        policy, {**reader.model_identity, "reader_mode": "assigned"},
        max_steps=reader.max_steps, image=reader.image,
        code_timeout_seconds=reader.code_timeout_seconds,
        source_domain="public_papers", protocol_variant=reader.protocol_variant,
        harness_version="v2", reader_mode="assigned")
