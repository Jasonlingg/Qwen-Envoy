import copy

import pytest

from src.eval.research_pilot_budget import RequestBudget
from src.eval.research_pilot_host import SYSTEM_PROMPT
from src.eval.research_pilot_runtime import (
    QWEN_BASE,
    QWEN_CHECKPOINT,
    QWEN_REVISION,
    build_models,
    validate_config,
    validate_endpoint,
)


def config():
    base = {"endpoint": "http://127.0.0.1:8000/v1", "model": "envoy",
            "tokenizer_id": QWEN_BASE, "tokenizer_revision": QWEN_REVISION,
            "context_window": 8192, "api_key_env": "PILOT_TEST_API_KEY",
            "identity_note": "test declaration"}
    return {"host": dict(base), "qwen": {**base, "checkpoint": QWEN_CHECKPOINT},
            "execution": {"billing_acknowledged": True, "hardware": {}}}


@pytest.mark.parametrize("url", ["http://example.com/v1", "https://user:pass@example.com/v1",
    "https://example.com/v1?key=secret", "file:///tmp/model", "https://example.com/v1#secret"])
def test_endpoint_rejects_credentials_insecure_remote_or_nonhttp(url):
    with pytest.raises(ValueError):
        validate_endpoint(url)


def test_config_requires_explicit_billing_ack_and_pinned_qwen():
    data = config()
    validate_config(data)
    data["execution"]["billing_acknowledged"] = False
    with pytest.raises(ValueError, match="billing"):
        validate_config(data)
    data = config()
    data["qwen"]["checkpoint"] = "other"
    with pytest.raises(ValueError, match="pinned"):
        validate_config(data)


def test_factories_share_budget_but_keep_prompt_and_fresh_history(monkeypatch):
    seen = []
    class Counter:
        metadata = {"count_kind": "fake_test"}
        def __init__(self, *args):
            pass
        def __call__(self, messages):
            return 100
    def upstream(url, payload, headers, timeout):
        seen.append(copy.deepcopy(payload))
        return {"choices": [{"message": {"content": "{}"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 1}}
    monkeypatch.delenv("ENVOY_MODEL_EXTRA_JSON", raising=False)
    monkeypatch.setenv("ENVOY_MODEL_API_KEY", "must-not-leak-to-other-endpoint")
    monkeypatch.delenv("PILOT_TEST_API_KEY", raising=False)
    budget = RequestBudget(max_requests=2)
    factory, reader, metadata = build_models(config(), budget, transport=upstream,
                                             counter_factory=Counter)
    direct, delegated = factory("P01", "direct"), factory("P01", "delegated")
    assert direct is not delegated
    assert direct.system_prompt == delegated.system_prompt == SYSTEM_PROMPT
    assert direct.api_key is None and reader.policy_factory().api_key is None
    direct.act("obs")
    reader.policy_factory().act("obs")
    assert reader.max_steps == 4
    assert reader.harness_version == "v2"
    assert budget.summary()["reserved"]["requests"] == 2
    assert metadata["qwen"]["count_kind"] == "fake_test"
    assert seen[0]["messages"][0]["content"] == SYSTEM_PROMPT
    assert seen[1]["messages"][0]["content"] == reader.system_prompt
