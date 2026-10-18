"""Explicit model configuration and bounded, nonretrying pilot HTTP transport."""

from __future__ import annotations

import hashlib
import json
import os
from urllib import error, request
from urllib.parse import urlsplit

from src.eval.artifacts import configuration_hash
from src.eval.research_pilot_host import SYSTEM_PROMPT
from src.policies.openai_compatible import OpenAICompatiblePolicy
from src.product.arxiv_mcp_config import build_reader

QWEN_BASE = "Qwen/Qwen3-8B"
QWEN_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
QWEN_CHECKPOINT = ("jasonlingg/qwen-envoy-qwen3-8b-qasper-sft-v5@"
                   "27b912a863ff914ad45baa03624d8911dc1e17fb/artifacts/full/checkpoint-50")
QWEN_ADAPTER_SHA256 = "7afba233aecc9fb9f1f23638514e309d139caf5c55d81af4138a05edd89d62a3"
MAX_MODEL_RESPONSE_BYTES = 1_000_000


class CachedChatCounter:
    def __init__(self, identifier, revision):
        from transformers import AutoTokenizer
        if not isinstance(revision, str) or len(revision) != 40:
            raise ValueError("Use an immutable 40-character tokenizer revision")
        self.tokenizer = AutoTokenizer.from_pretrained(identifier, revision=revision,
                                                       local_files_only=True)
        if not self.tokenizer.chat_template:
            raise ValueError("Tokenizer has no chat template")
        self.metadata = {"id": identifier, "revision": revision, "local_files_only": True,
                         "vocabulary_sha256": configuration_hash(self.tokenizer.get_vocab()),
                         "chat_template_sha256": hashlib.sha256(
                             self.tokenizer.chat_template.encode()).hexdigest(),
                         "chat_template_kwargs": {"enable_thinking": False},
                         "count_kind": "local_chat_template_tokens_not_provider_usage"}

    def __call__(self, messages):
        encoded = self.tokenizer.apply_chat_template(messages, tokenize=True,
            add_generation_prompt=True, enable_thinking=False)
        return len(encoded["input_ids"] if hasattr(encoded, "keys") else encoded)


def validate_endpoint(endpoint):
    if not isinstance(endpoint, str):
        raise ValueError("Model endpoint must be a URL")
    parsed = urlsplit(endpoint)
    if (not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.scheme not in {"http", "https"}
            or (parsed.scheme == "http"
                and parsed.hostname not in {"localhost", "127.0.0.1", "::1"})):
        raise ValueError("Use authenticated HTTPS or a loopback HTTP tunnel; no URL credentials")
    return endpoint


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Pilot model endpoint redirects are forbidden")


def bounded_transport(url, payload, headers, timeout):
    validate_endpoint(url)
    opener = request.build_opener(request.ProxyHandler({}), _NoRedirect())
    req = request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    try:
        with opener.open(req, timeout=timeout) as response:
            body = response.read(MAX_MODEL_RESPONSE_BYTES + 1)
    except error.HTTPError as exc:
        raise RuntimeError(f"Model endpoint HTTP {exc.code}; no retry") from None
    except error.URLError:
        raise RuntimeError("Model endpoint connection failed; no retry") from None
    if len(body) > MAX_MODEL_RESPONSE_BYTES:
        raise ValueError("Model response exceeds pilot byte limit")
    result = json.loads(body)
    if not isinstance(result, dict):
        raise ValueError("Model response is not an object")
    return result


def validate_config(config):
    if not isinstance(config, dict) or set(config) != {"host", "qwen", "execution"}:
        raise ValueError("Configuration needs exactly host, qwen, and execution")
    for kind in ("host", "qwen"):
        model = config[kind]
        required = {"endpoint", "model", "tokenizer_id", "tokenizer_revision", "context_window",
                    "api_key_env", "identity_note"}
        if kind == "qwen":
            required.add("checkpoint")
        if not isinstance(model, dict) or set(model) != required:
            raise ValueError(f"Invalid {kind} configuration fields")
        validate_endpoint(model["endpoint"])
        if (any(not isinstance(model[k], str) or not model[k].strip() for k in
                ("model", "tokenizer_id", "tokenizer_revision", "identity_note"))
                or type(model["context_window"]) is not int
                or not 2_048 <= model["context_window"] <= 131_072
                or not isinstance(model["api_key_env"], str)):
            raise ValueError(f"Invalid {kind} configuration values")
    qwen = config["qwen"]
    if (qwen["checkpoint"] != QWEN_CHECKPOINT or qwen["tokenizer_id"] != QWEN_BASE
            or qwen["tokenizer_revision"] != QWEN_REVISION or qwen["context_window"] != 8192):
        raise ValueError("Pilot requires pinned Envoy v5 and its 8192-token Qwen template")
    execution = config["execution"]
    if (not isinstance(execution, dict) or set(execution) != {"billing_acknowledged", "hardware"}
            or execution["billing_acknowledged"] is not True
            or not isinstance(execution["hardware"], dict)):
        raise ValueError("Explicit endpoint billing acknowledgment and hardware record required")


def build_models(config, budget, *, transport=bounded_transport, counter_factory=CachedChatCounter):
    validate_config(config)
    counters = {role: counter_factory(config[role]["tokenizer_id"],
                                     config[role]["tokenizer_revision"])
                for role in ("host", "qwen")}
    def wrapped(role, label):
        return budget.transport(label, counters[role], config[role]["context_window"], transport)

    def host_factory(question_id, arm):
        settings = config["host"]
        policy = OpenAICompatiblePolicy(endpoint=settings["endpoint"], model=settings["model"],
            api_key=os.environ.get(settings["api_key_env"]), max_tokens=1024, temperature=0.0,
            timeout=60.0, extra_body={"top_p": 1.0, "seed": 42,
                                     "chat_template_kwargs": {"enable_thinking": False}},
            transport=wrapped("host", f"host:{question_id}:{arm}"),
            system_prompt=SYSTEM_PROMPT, action_cleaner=lambda value: value.strip())
        # An absent selected key must not inherit the generic Envoy credential.
        policy.api_key = os.environ.get(settings["api_key_env"])
        return policy

    settings = config["qwen"]
    reader = build_reader("qwen", qwen_endpoint=settings["endpoint"],
        qwen_model=settings["model"], qwen_checkpoint=settings["checkpoint"],
        policy_transport=wrapped("qwen", "qwen"), max_steps=4)
    reader.model_identity.update(base_model=QWEN_BASE, base_revision=QWEN_REVISION,
        expected_adapter_sha256=QWEN_ADAPTER_SHA256, weight_attestation="not_remotely_attested")
    # The runtime factory reads this operator-selected key without persisting it.
    original_factory = reader.policy_factory
    def reader_policy():
        policy = original_factory()
        policy.api_key = os.environ.get(settings["api_key_env"])
        policy.timeout = 60.0
        return policy
    reader.policy_factory = reader_policy
    metadata = {role: counter.metadata for role, counter in counters.items()}
    return host_factory, reader, metadata
