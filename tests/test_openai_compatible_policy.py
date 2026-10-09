import pytest

from src.policies.openai_compatible import OpenAICompatiblePolicy
from src.policies.protocol import Policy
from src.policies.registry import POLICY_SPECS, policy_catalog, requires_anthropic_key


def test_endpoint_policy_uses_the_shared_action_contract():
    calls = []

    def transport(url, payload, headers, timeout):
        calls.append((url, payload, headers, timeout))
        return {"choices": [{"message": {"content": "```python\nprint(search('masking'))\n```"}}]}

    policy = OpenAICompatiblePolicy(
        endpoint="http://localhost:8000/v1/",
        model="any-model",
        api_key="secret",
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        transport=transport,
    )

    assert isinstance(policy, Policy)
    assert policy.act("Question: why mask tokens?") == "print(search('masking'))"
    url, payload, headers, timeout = calls[0]
    assert url == "http://localhost:8000/v1/chat/completions"
    assert payload["model"] == "any-model"
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][-1] == {"role": "user", "content": "Question: why mask tokens?"}
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert headers["Authorization"] == "Bearer secret"
    assert timeout == 120.0

    policy.reset()
    assert policy.history == []


def test_endpoint_policy_accepts_structured_text_content():
    policy = OpenAICompatiblePolicy(
        endpoint="http://localhost:11434/v1/chat/completions",
        model="local",
        transport=lambda *args: {
            "choices": [
                {"message": {"content": [{"type": "text", "text": "SUBMIT: answer CITATIONS: []"}]}}
            ]
        },
    )
    assert policy.act("Question") == "SUBMIT: answer CITATIONS: []"


@pytest.mark.parametrize("reasoning_field", ["reasoning", "reasoning_content"])
def test_endpoint_policy_separates_structured_reasoning_from_executable_action(reasoning_field):
    policy = OpenAICompatiblePolicy(
        endpoint="http://localhost/v1", model="qwen",
        transport=lambda *args: {"choices": [{"message": {
            reasoning_field: "Maybe SUBMIT: a wrong answer CITATIONS: []",
            "content": "print(search('actual evidence'))",
        }}]},
    )
    assert policy.act("Question") == "print(search('actual evidence'))"
    assert policy.last_reasoning == "Maybe SUBMIT: a wrong answer CITATIONS: []"
    assert policy.history[-1]["content"] == "print(search('actual evidence'))"


@pytest.mark.parametrize("content", [
    "<think>SUBMIT: false CITATIONS: []</think>\nprint(search('real'))",
    "SUBMIT: false CITATIONS: []</think>\nprint(search('real'))",
])
def test_endpoint_policy_strips_inline_thinking_before_clean_action(content):
    policy = OpenAICompatiblePolicy(
        endpoint="http://localhost/v1", model="qwen",
        transport=lambda *args: {"choices": [{"message": {"content": content}}]},
    )
    assert policy.act("Question") == "print(search('real'))"
    assert policy.last_reasoning == "SUBMIT: false CITATIONS: []"
    assert "<think>" not in policy.history[-1]["content"]


@pytest.mark.parametrize("content", [
    "<think>print('must never execute')",
    "<think>print('must never execute')</think>",
    "print('ambiguous')<think>reasoning</think>",
    "<think>reasoning</think>print('ok')</think>",
])
def test_endpoint_policy_fails_closed_on_unfinished_or_ambiguous_thinking(content):
    policy = OpenAICompatiblePolicy(
        endpoint="http://localhost/v1", model="qwen",
        transport=lambda *args: {"choices": [{"message": {"content": content}}]},
    )
    with pytest.raises(ValueError, match="thinking"):
        policy.act("Question")
    assert policy.last_reasoning is None
    assert len(policy.history) == 1


def test_endpoint_policy_clears_reasoning_before_a_non_thinking_turn():
    responses = iter([
        {"choices": [{"message": {"reasoning": "Inspect first", "content": "print('ok')"}}]},
        {"choices": [{"message": {"content": "SUBMIT: done CITATIONS: []"}}]},
    ])
    policy = OpenAICompatiblePolicy(
        endpoint="http://localhost/v1", model="qwen",
        transport=lambda *args: next(responses),
    )
    policy.act("Question")
    assert policy.last_reasoning == "Inspect first"
    policy.act("Observation")
    assert policy.last_reasoning is None


def test_endpoint_policy_records_bounded_sampled_logprobs_without_changing_action():
    payloads = []
    responses = iter([
        {"choices": [{
            "message": {"reasoning": "inspect", "content": "print('ok')"},
            "logprobs": {"content": [
                {"token": "print", "logprob": -0.2, "top_logprobs": [
                    {"token": "print", "logprob": -0.2},
                    {"token": "SUBMIT", "logprob": -3.1},
                ]},
                {"token": "(", "logprob": -0.1},
                {"token": "'ok'", "logprob": None},
            ]},
        }]},
        {"choices": [{"message": {"content": "SUBMIT: done CITATIONS: []"}}]},
    ])
    policy = OpenAICompatiblePolicy(
        endpoint="http://localhost/v1", model="qwen",
        extra_body={"logprobs": True, "top_logprobs": 2},
        transport=lambda _url, payload, _headers, _timeout: (
            payloads.append(payload) or next(responses)
        ),
    )
    assert policy.act("Question") == "print('ok')"
    assert payloads[0]["logprobs"] is True
    assert payloads[0]["top_logprobs"] == 2
    assert policy.last_reasoning == "inspect"
    diagnostic = policy.last_logprob_diagnostics
    assert diagnostic["reported_token_count"] == 3
    assert diagnostic["valid_logprob_count"] == 2
    assert diagnostic["sum_logprob"] == pytest.approx(-0.3)
    assert diagnostic["prefix"][0]["top_logprobs"][1] == {
        "token": "SUBMIT", "logprob": -3.1,
    }
    assert diagnostic["prefix"][2]["logprob"] is None
    assert "correctness" in diagnostic["scope"]
    assert policy.act("Observation") == "SUBMIT: done CITATIONS: []"
    assert policy.last_logprob_diagnostics is None
    policy.reset()
    assert policy.last_logprob_diagnostics is None


def test_invalid_logprob_telemetry_does_not_fail_an_action():
    policy = OpenAICompatiblePolicy(
        endpoint="http://localhost/v1", model="qwen",
        transport=lambda *args: {"choices": [{
            "message": {"content": "print('ok')"},
            "logprobs": {"content": [
                {"token": "print", "logprob": float("nan")},
                {"token": "(", "logprob": True},
                {"token": ")", "logprob": 10 ** 1000},
            ]},
        }]},
    )
    assert policy.act("Question") == "print('ok')"
    diagnostic = policy.last_logprob_diagnostics
    assert diagnostic["valid_logprob_count"] == 0
    assert diagnostic["mean_logprob"] is None
    assert all(item["logprob"] is None for item in diagnostic["prefix"])


def test_endpoint_policy_accepts_a_prompt_ablation():
    payloads = []
    policy = OpenAICompatiblePolicy(
        endpoint="http://localhost:8000/v1",
        model="local",
        system_prompt="diagnostic prompt",
        transport=lambda _url, payload, _headers, _timeout: (
            payloads.append(payload)
            or {"choices": [{"message": {"content": "SUBMIT: answer CITATIONS: []"}}]}
        ),
    )
    policy.act("Question")
    assert payloads[0]["messages"][0] == {
        "role": "system",
        "content": "diagnostic prompt",
    }


def test_endpoint_policy_requires_endpoint_and_model(monkeypatch):
    for name in ("ENVOY_MODEL_ENDPOINT", "ENVOY_MODEL_ID"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="ENVOY_MODEL_ENDPOINT"):
        OpenAICompatiblePolicy()


def test_policy_catalog_reports_endpoint_configuration(monkeypatch):
    monkeypatch.delenv("ENVOY_MODEL_ENDPOINT", raising=False)
    monkeypatch.delenv("ENVOY_MODEL_ID", raising=False)
    unavailable = {item["name"]: item for item in policy_catalog()}
    assert not unavailable["openai_compatible"]["available"]

    monkeypatch.setenv("ENVOY_MODEL_ENDPOINT", "http://localhost:8000/v1")
    monkeypatch.setenv("ENVOY_MODEL_ID", "qwen")
    available = {item["name"]: item for item in policy_catalog()}
    assert available["openai_compatible"]["available"]


def test_endpoint_policy_does_not_require_anthropic_credentials():
    assert not requires_anthropic_key(["openai_compatible"])
    assert requires_anthropic_key(["claude_policy"])
    assert not POLICY_SPECS["openai_compatible"].select_by_default


def _response(usage):
    return {
        "choices": [{"message": {"content": "print('inspect')"}}],
        "usage": usage,
    }


def test_usage_tracks_all_calls_and_exposes_only_numeric_metadata(monkeypatch):
    clock = iter([10.0, 10.25, 11.0, 11.5])
    monkeypatch.setattr("src.policies.openai_compatible.time.monotonic", lambda: next(clock))
    responses = iter([
        _response({
            "prompt_tokens": 12, "completion_tokens": 3, "secret": "never-copy",
            "prompt_tokens_details": {"cached_tokens": 8, "secret": "never-copy"},
        }),
        _response({
            "prompt_tokens": 20, "completion_tokens": 0,
            "prompt_tokens_details": {"cached_tokens": 0},
        }),
    ])
    policy = OpenAICompatiblePolicy(
        endpoint="https://private-endpoint.invalid/v1", model="private-model",
        api_key="private-api-key", transport=lambda *args: next(responses),
    )
    policy.act("private question")
    policy.act("private observation")
    usage = policy.eval_metadata()["token_usage"]
    assert usage["request_count"] == usage["usage_observed_request_count"] == 2
    assert usage["usage_missing_request_count"] == usage["failed_request_count"] == 0
    assert usage["usage_complete"] is True
    assert usage["prompt_tokens"] == usage["observed_prompt_tokens"] == 32
    assert usage["completion_tokens"] == usage["observed_completion_tokens"] == 3
    assert usage["cached_prompt_tokens"] == usage["observed_cached_prompt_tokens"] == 8
    assert usage["cached_usage_complete"] is True
    assert usage["cached_usage_observed_request_count"] == 2
    assert usage["cached_usage_missing_request_count"] == 0
    assert usage["requests"] == [
        {
            "prompt_tokens": 12, "completion_tokens": 3, "cached_prompt_tokens": 8,
            "duration_seconds": 0.25, "failed": False,
        },
        {
            "prompt_tokens": 20, "completion_tokens": 0, "cached_prompt_tokens": 0,
            "duration_seconds": 0.5, "failed": False,
        },
    ]
    assert "private" not in str(policy.eval_metadata())
    assert "never-copy" not in str(policy.eval_metadata())
    usage["requests"][0]["prompt_tokens"] = 999
    assert policy.eval_metadata()["token_usage"]["observed_prompt_tokens"] == 32
    policy.reset()
    reset = policy.eval_metadata()["token_usage"]
    assert reset["request_count"] == 0
    assert reset["prompt_tokens"] is reset["observed_prompt_tokens"] is None
    assert reset["cached_prompt_tokens"] is reset["observed_cached_prompt_tokens"] is None
    assert reset["cached_usage_complete"] is False
    assert reset["cached_usage_observed_request_count"] == 0
    assert reset["cached_usage_missing_request_count"] == 0
    assert reset["requests"] == []


@pytest.mark.parametrize("invalid", [None, "4", True, -1, 4.5, float("nan")])
def test_missing_or_invalid_provider_usage_cannot_become_zero_or_full_totals(invalid):
    responses = iter([
        _response({"prompt_tokens": 12, "completion_tokens": 3}),
        _response({"prompt_tokens": 8, "completion_tokens": invalid}),
    ])
    policy = OpenAICompatiblePolicy(
        endpoint="http://localhost/v1", model="local", transport=lambda *args: next(responses),
    )
    policy.act("q")
    policy.act("observation")
    usage = policy.eval_metadata()["token_usage"]
    assert usage["usage_complete"] is False
    assert usage["usage_missing_request_count"] == 1
    assert usage["prompt_tokens"] is usage["completion_tokens"] is None
    assert usage["observed_prompt_tokens"] == 20
    assert usage["observed_completion_tokens"] == 3
    assert usage["requests"][1]["completion_tokens"] is None


def test_failed_request_retains_observed_tokens_but_invalidates_full_totals(monkeypatch):
    clock = iter([10.0, 10.25, 11.0, 15.0])
    monkeypatch.setattr("src.policies.openai_compatible.time.monotonic", lambda: next(clock))
    calls = 0

    def transport(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("private endpoint failure detail")
        return _response({
            "prompt_tokens": 12, "completion_tokens": 3,
            "prompt_tokens_details": {"cached_tokens": 12},
        })

    policy = OpenAICompatiblePolicy(
        endpoint="http://localhost/v1", model="local", transport=transport,
    )
    policy.act("q")
    with pytest.raises(RuntimeError, match="private endpoint"):
        policy.act("observation")
    usage = policy.eval_metadata()["token_usage"]
    assert usage["request_count"] == 2
    assert usage["failed_request_count"] == usage["usage_missing_request_count"] == 1
    assert usage["observed_prompt_tokens"] == 12
    assert usage["observed_completion_tokens"] == 3
    assert usage["prompt_tokens"] is usage["completion_tokens"] is None
    assert usage["observed_cached_prompt_tokens"] == 12
    assert usage["cached_prompt_tokens"] is None
    assert usage["cached_usage_complete"] is False
    assert usage["cached_usage_observed_request_count"] == 1
    assert usage["cached_usage_missing_request_count"] == 1
    assert [record["duration_seconds"] for record in usage["requests"]] == [0.25, 4.0]
    assert "private endpoint" not in str(usage)


def test_unparseable_response_preserves_reported_usage_as_observed_only(monkeypatch):
    clock = iter([10.0, 10.75])
    monkeypatch.setattr("src.policies.openai_compatible.time.monotonic", lambda: next(clock))
    policy = OpenAICompatiblePolicy(
        endpoint="http://localhost/v1", model="local", transport=lambda *args: {
            "choices": [], "usage": {
                "prompt_tokens": 12, "completion_tokens": 3,
                "prompt_tokens_details": {"cached_tokens": 8},
            },
        },
    )
    with pytest.raises(ValueError, match="response shape"):
        policy.act("q")
    usage = policy.eval_metadata()["token_usage"]
    assert usage["request_count"] == usage["failed_request_count"] == 1
    assert usage["usage_observed_request_count"] == 1
    assert usage["usage_complete"] is False
    assert usage["observed_prompt_tokens"] == 12
    assert usage["prompt_tokens"] is None
    assert usage["observed_cached_prompt_tokens"] == 8
    assert usage["cached_prompt_tokens"] is None
    assert usage["cached_usage_complete"] is False
    assert usage["cached_usage_observed_request_count"] == 1
    assert usage["cached_usage_missing_request_count"] == 0
    assert usage["requests"][0]["duration_seconds"] == 0.75


def test_absent_usage_stays_unknown():
    policy = OpenAICompatiblePolicy(
        endpoint="http://localhost/v1", model="local",
        transport=lambda *args: {"choices": [{"message": {"content": "print('ok')"}}]},
    )
    policy.act("q")
    usage = policy.eval_metadata()["token_usage"]
    assert usage["request_count"] == usage["usage_missing_request_count"] == 1
    assert usage["observed_prompt_tokens"] is usage["observed_completion_tokens"] is None
    assert usage["prompt_tokens"] is usage["completion_tokens"] is None
    assert usage["observed_cached_prompt_tokens"] is usage["cached_prompt_tokens"] is None
    assert usage["cached_usage_complete"] is False
    assert usage["cached_usage_missing_request_count"] == 1


@pytest.mark.parametrize("details", [
    None, [], "cached_tokens", 0, {},
    {"cached_tokens": None}, {"cached_tokens": "4"}, {"cached_tokens": True},
    {"cached_tokens": -1}, {"cached_tokens": 4.5}, {"cached_tokens": float("nan")},
    {"cached_tokens": 13},
])
def test_missing_or_invalid_cached_usage_preserves_normal_token_totals(details):
    responses = iter([
        _response({
            "prompt_tokens": 12, "completion_tokens": 3,
            "prompt_tokens_details": {"cached_tokens": 4},
        }),
        _response({
            "prompt_tokens": 12, "completion_tokens": 3,
            "prompt_tokens_details": details,
        }),
    ])
    policy = OpenAICompatiblePolicy(
        endpoint="http://localhost/v1", model="local", transport=lambda *args: next(responses),
    )
    policy.act("q")
    policy.act("observation")
    usage = policy.eval_metadata()["token_usage"]
    assert usage["usage_complete"] is True
    assert usage["prompt_tokens"] == 24
    assert usage["completion_tokens"] == 6
    assert usage["cached_usage_complete"] is False
    assert usage["cached_usage_observed_request_count"] == 1
    assert usage["cached_usage_missing_request_count"] == 1
    assert usage["observed_cached_prompt_tokens"] == 4
    assert usage["cached_prompt_tokens"] is None
    assert usage["requests"][1]["cached_prompt_tokens"] is None


@pytest.mark.parametrize("prompt_tokens", [None, "4", True, -1, 4.5])
def test_cached_usage_can_be_reported_when_prompt_token_total_is_unknown(prompt_tokens):
    policy = OpenAICompatiblePolicy(
        endpoint="http://localhost/v1", model="local", transport=lambda *args: _response({
            "prompt_tokens": prompt_tokens, "completion_tokens": 3,
            "prompt_tokens_details": {"cached_tokens": 8},
        }),
    )
    policy.act("q")
    usage = policy.eval_metadata()["token_usage"]
    assert usage["usage_complete"] is False
    assert usage["prompt_tokens"] is None
    assert usage["cached_usage_complete"] is True
    assert usage["cached_prompt_tokens"] == usage["observed_cached_prompt_tokens"] == 8


def test_reported_zero_cached_tokens_is_complete_usage():
    policy = OpenAICompatiblePolicy(
        endpoint="http://localhost/v1", model="local", transport=lambda *args: _response({
            "prompt_tokens": 0, "completion_tokens": 0,
            "prompt_tokens_details": {"cached_tokens": 0},
        }),
    )
    policy.act("q")
    usage = policy.eval_metadata()["token_usage"]
    assert usage["cached_usage_complete"] is True
    assert usage["cached_prompt_tokens"] == usage["observed_cached_prompt_tokens"] == 0
