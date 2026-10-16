"""Build the arXiv MCP's reader and reasoner from explicit configuration."""

from __future__ import annotations

import os

from src.policies.code_execution import DEFAULT_MAX_TOKENS, clean_action
from src.policies.openai_compatible import OpenAICompatiblePolicy
from src.product.harness_v2 import clean_v2_action
from src.product.qwen_investigator import (
    QwenInvestigator,
    effective_system_prompt,
)
from src.product.reasoner import Reasoner

NEBIUS_ENDPOINT = "https://api.tokenfactory.nebius.com/v1"
DEFAULT_NEMOTRON = "nvidia/Nemotron-3-Ultra-550b-a55b"


def build_reader(kind: str, *, qwen_endpoint: str | None = None, qwen_model: str | None = None,
                 qwen_checkpoint: str | None = None, nemotron_model: str = DEFAULT_NEMOTRON,
                 harness_version: str = "v2") -> QwenInvestigator | None:
    if kind == "none":
        return None
    if kind == "qwen":
        if not (qwen_endpoint and qwen_model and qwen_checkpoint):
            raise ValueError("--reader qwen needs --qwen-endpoint, --qwen-model, --qwen-checkpoint")
        endpoint, model, api_key = qwen_endpoint, qwen_model, os.environ.get("ENVOY_MODEL_API_KEY")
        variant = "default"
        extra = {"chat_template_kwargs": {"enable_thinking": False}, "top_p": 1.0, "seed": 42}
        identity = {"checkpoint": qwen_checkpoint, "served_model": qwen_model,
                    "endpoint": qwen_endpoint, "seed": 42}
    elif kind == "nemotron":
        api_key = os.environ.get("NEBIUS_API_KEY")
        if not api_key:
            raise ValueError("--reader nemotron needs NEBIUS_API_KEY in the environment")
        endpoint, model = NEBIUS_ENDPOINT, nemotron_model
        variant = "nemotron"
        extra = {"chat_template_kwargs": {"enable_thinking": False}, "top_p": 1.0}
        identity = {"provider": "nebius_token_factory", "served_model": nemotron_model}
    else:
        raise ValueError("reader must be none, qwen, or nemotron")

    prompt = effective_system_prompt("public_papers", variant, harness_version)

    def policy() -> OpenAICompatiblePolicy:
        return OpenAICompatiblePolicy(endpoint=endpoint, model=model, api_key=api_key,
                                      system_prompt=prompt, max_tokens=DEFAULT_MAX_TOKENS,
                                      temperature=0.0, extra_body=extra,
                                      action_cleaner=(clean_v2_action if harness_version == "v2"
                                                      else clean_action))

    return QwenInvestigator(
        policy_factory=policy,
        model_identity={**identity, "temperature": 0.0, "top_p": 1.0, "thinking": False,
                        "harness_version": harness_version, "max_tokens": DEFAULT_MAX_TOKENS},
        source_domain="public_papers", protocol_variant=variant,
        harness_version=harness_version,
    )


def build_reasoner(kind: str, model: str = DEFAULT_NEMOTRON) -> Reasoner | None:
    if kind == "none":
        return None
    if kind != "nemotron":
        raise ValueError("reasoner must be none or nemotron")
    api_key = os.environ.get("NEBIUS_API_KEY")
    if not api_key:
        raise ValueError("--reasoner nemotron needs NEBIUS_API_KEY in the environment")
    return Reasoner(NEBIUS_ENDPOINT, model, api_key)
