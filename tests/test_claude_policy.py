"""Offline checks for ClaudePolicy's model/temperature handling."""

import anthropic
import pytest

from src.policies.claude_policy import ClaudePolicy


class _Response:
    def __init__(self, text: str, thinking: str | None = None):
        blocks = []
        if thinking is not None:
            blocks.append(type("ThinkingBlock", (), {"thinking": thinking})())
        blocks.append(type("TextBlock", (), {"text": text})())
        self.content = blocks


class _FakeMessages:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        if hasattr(result, "content"):
            return result
        return _Response(result)


def _bad_request(message: str) -> anthropic.BadRequestError:
    request = anthropic._base_client.httpx.Request("POST", "https://api.anthropic.com")
    response = anthropic._base_client.httpx.Response(400, request=request, json={
        "type": "error", "error": {"type": "invalid_request_error", "message": message},
    })
    return anthropic.BadRequestError(message, response=response, body=None)


def test_model_override_env_var(monkeypatch):
    monkeypatch.setenv("CLAUDE_MODEL_PATH", "claude-sonnet-5")
    monkeypatch.setattr(anthropic, "Anthropic", lambda api_key=None: object())
    policy = ClaudePolicy()
    assert policy.model == "claude-sonnet-5"


def test_explicit_model_wins_over_env_var(monkeypatch):
    monkeypatch.setenv("CLAUDE_MODEL_PATH", "claude-sonnet-5")
    monkeypatch.setattr(anthropic, "Anthropic", lambda api_key=None: object())
    policy = ClaudePolicy(model="claude-haiku-4-5-20251001")
    assert policy.model == "claude-haiku-4-5-20251001"


def test_teacher_can_use_student_prompt_without_changing_default(monkeypatch):
    from src.policies.claude_prompts import SYSTEM_PROMPT as DEFAULT_PROMPT
    from src.policies.code_execution import SYSTEM_PROMPT as STUDENT_PROMPT

    fake = _FakeMessages(["print(1)", "print(2)"])
    monkeypatch.setattr(
        anthropic, "Anthropic",
        lambda api_key=None: type("Client", (), {"messages": fake})(),
    )
    ClaudePolicy(system_prompt=STUDENT_PROMPT).act("teacher question")
    ClaudePolicy().act("ordinary question")
    assert fake.calls[0]["system"] == STUDENT_PROMPT
    assert fake.calls[1]["system"] == DEFAULT_PROMPT


def test_eval_registry_passes_shared_prompt_and_action_budget(monkeypatch):
    from src.policies.registry import build_policies

    monkeypatch.setenv("ENVOY_CLAUDE_MAX_TOKENS", "1024")
    monkeypatch.setattr(anthropic, "Anthropic", lambda api_key=None: object())
    policy = build_policies(
        None, ["claude_policy"], system_prompt="shared evaluation prompt"
    )["claude_policy"]
    assert policy.system_prompt == "shared evaluation prompt"
    assert policy.max_tokens == 1024


def test_sonnet_eval_disables_thinking_and_omits_temperature(monkeypatch):
    from src.policies.registry import build_policies

    fake = _FakeMessages(["print(1)"])
    monkeypatch.setenv("ENVOY_CLAUDE_THINKING_DISABLED", "1")
    monkeypatch.setenv("CLAUDE_MODEL_PATH", "claude-sonnet-5")
    monkeypatch.setattr(
        anthropic, "Anthropic",
        lambda api_key=None: type("Client", (), {"messages": fake})(),
    )
    policy = build_policies(None, ["claude_policy"])["claude_policy"]
    policy.act("question")
    assert fake.calls[0]["thinking"] == {"type": "disabled"}
    assert "temperature" not in fake.calls[0]


def test_records_provider_usage_and_clears_it_between_episodes(monkeypatch):
    response = _Response("print(1)")
    response.model = "resolved-model-id"
    response.stop_reason = "end_turn"
    response.usage = type("Usage", (), {
        "input_tokens": 120, "output_tokens": 15,
        "cache_creation_input_tokens": 30, "cache_read_input_tokens": 0,
    })()
    fake = _FakeMessages([response])
    monkeypatch.setattr(
        anthropic, "Anthropic",
        lambda api_key=None: type("Client", (), {"messages": fake})(),
    )
    policy = ClaudePolicy()
    policy.act("question")
    assert policy.usage_records == [{
        "model": "resolved-model-id", "stop_reason": "end_turn", "input_tokens": 120,
        "output_tokens": 15, "cache_creation_input_tokens": 30, "cache_read_input_tokens": 0,
    }]
    policy.reset()
    assert policy.usage_records == []


def test_falls_back_when_model_rejects_temperature(monkeypatch):
    fake = _FakeMessages([
        _bad_request("`temperature` is deprecated for this model."),
        "print(1)",
    ])
    monkeypatch.setattr(
        anthropic, "Anthropic",
        lambda api_key=None: type("Client", (), {"messages": fake})(),
    )
    policy = ClaudePolicy(model="claude-sonnet-5")
    action = policy.act("go")
    assert action.strip() == "print(1)"
    assert "temperature" in fake.calls[0]
    assert "temperature" not in fake.calls[1]
    assert all(call["model"] == "claude-sonnet-5" for call in fake.calls)


def test_second_turn_does_not_resend_temperature_after_fallback(monkeypatch):
    fake = _FakeMessages([
        _bad_request("`temperature` is deprecated for this model."),
        "step_one()",
        "step_two()",
    ])
    monkeypatch.setattr(
        anthropic, "Anthropic",
        lambda api_key=None: type("Client", (), {"messages": fake})(),
    )
    policy = ClaudePolicy(model="claude-sonnet-5")
    policy.act("first")
    policy.act("second")
    assert all("temperature" not in call for call in fake.calls[1:])


def test_skips_thinking_block_and_uses_the_final_text_block(monkeypatch):
    """A model with native thinking on by default (e.g. claude-sonnet-5) returns
    a ThinkingBlock before the actual text response; content[0].text would
    crash on it since ThinkingBlock has no .text attribute."""
    fake = _FakeMessages([_Response("search('x')", thinking="reasoning about x...")])
    monkeypatch.setattr(
        anthropic, "Anthropic",
        lambda api_key=None: type("Client", (), {"messages": fake})(),
    )
    policy = ClaudePolicy(model="claude-sonnet-5")
    action = policy.act("go")
    assert action.strip() == "search('x')"


def test_raises_a_clear_error_when_response_has_no_text_block(monkeypatch):
    fake = _FakeMessages([type("Empty", (), {"content": []})()])
    monkeypatch.setattr(
        anthropic, "Anthropic",
        lambda api_key=None: type("Client", (), {"messages": fake})(),
    )
    policy = ClaudePolicy(model="claude-sonnet-5")
    with pytest.raises(ValueError, match="no text content"):
        policy.act("go")


def test_unrelated_bad_request_is_not_swallowed(monkeypatch):
    fake = _FakeMessages([_bad_request("invalid api key")])
    monkeypatch.setattr(
        anthropic, "Anthropic",
        lambda api_key=None: type("Client", (), {"messages": fake})(),
    )
    policy = ClaudePolicy(model="claude-sonnet-5")
    with pytest.raises(anthropic.BadRequestError):
        policy.act("go")


def test_teacher_pin_rejects_haiku_before_constructing_client(monkeypatch):
    from src.policies.claude_policy import ModelIdentityError

    def unexpected_client(**_):
        pytest.fail("Wrong teacher must fail before creating an API client")

    monkeypatch.setattr(anthropic, "Anthropic", unexpected_client)
    with pytest.raises(ModelIdentityError, match="refusing API calls"):
        ClaudePolicy(model="claude-haiku-4-5-20251001", expected_model="claude-sonnet-5")


@pytest.mark.parametrize("returned_model", ["claude-haiku-4-5-20251001", None])
def test_teacher_pin_rejects_wrong_or_missing_provider_identity(monkeypatch, returned_model):
    from src.policies.claude_policy import ModelIdentityError

    response = _Response("print('must not execute')")
    response.model = returned_model
    fake = _FakeMessages([response])
    monkeypatch.setattr(anthropic, "Anthropic",
        lambda api_key=None: type("Client", (), {"messages": fake})())
    policy = ClaudePolicy(model="claude-sonnet-5", expected_model="claude-sonnet-5")
    with pytest.raises(ModelIdentityError, match="refusing to use this action"):
        policy.act("question")
    assert len(fake.calls) == 1
    assert not any(m["role"] == "assistant" for m in policy.history)


def test_sonnet_teacher_ignores_haiku_environment_and_records_provider_identity(monkeypatch):
    from scripts.generate_qasper_teacher_batch import SonnetTeacherPolicy, _stop

    _stop.clear()
    monkeypatch.setenv("CLAUDE_MODEL_PATH", "claude-haiku-4-5-20251001")
    response = _Response("print(1)")
    response.model = "claude-sonnet-5"
    fake = _FakeMessages([response])
    monkeypatch.setattr(anthropic, "Anthropic",
        lambda api_key=None: type("Client", (), {"messages": fake})())
    policy = SonnetTeacherPolicy(model="claude-sonnet-5")
    assert policy.act("question") == "print(1)"
    assert fake.calls[0]["model"] == policy.usage_records[0]["model"] == "claude-sonnet-5"


def test_teacher_model_mismatch_stops_subsequent_requests(monkeypatch):
    from scripts.generate_qasper_teacher_batch import SonnetTeacherPolicy, _stop
    from src.policies.claude_policy import ModelIdentityError

    _stop.clear()
    response = _Response("print('wrong model')")
    response.model = "claude-haiku-4-5-20251001"
    fake = _FakeMessages([response])
    monkeypatch.setattr(anthropic, "Anthropic",
        lambda api_key=None: type("Client", (), {"messages": fake})())
    try:
        first = SonnetTeacherPolicy(model="claude-sonnet-5")
        second = SonnetTeacherPolicy(model="claude-sonnet-5")
        with pytest.raises(ModelIdentityError):
            first.act("question")
        with pytest.raises(RuntimeError, match="refusing another API request"):
            second.act("question")
        assert len(fake.calls) == 1
    finally:
        _stop.clear()
