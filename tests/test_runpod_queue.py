import json

import pytest

from src.policies.runpod_queue import RunpodQueueTransport


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == dict(tokenize=True, add_generation_prompt=True, enable_thinking=False)
        return [1] * len(messages[0]["content"])


def payload(**updates):
    return {"model": "envoy", "messages": [{"role": "user", "content": "x"}],
            "max_tokens": 1024, "chat_template_kwargs": {"enable_thinking": False}, **updates}


def test_success_uses_one_submission_and_enforces_shared_budget(tmp_path):
    calls = []
    def http(method, path, body=None):
        calls.append((method, path, body))
        return {"id": "job-1", "status": "COMPLETED", "output": [{
            "model": "envoy", "choices": [{"message": {"content": "print(1)"}}]}]}
    audit = tmp_path / "audit.jsonl"
    transport = RunpodQueueTransport("abc", "secret", Tokenizer(), audit,
                                      requests_max=1, http=http)
    assert transport("unused", payload(), {}, 120)["model"] == "envoy"
    with pytest.raises(RuntimeError, match="budget exhausted"):
        transport("unused", payload(), {}, 120)
    assert len(calls) == 1
    assert calls[0][2]["policy"]["executionTimeout"] == 120000
    assert "secret" not in audit.read_text()
    with pytest.raises(FileExistsError):
        RunpodQueueTransport("abc", "secret", Tokenizer(), audit)


def test_uncertain_submission_is_not_retried_or_refunded(tmp_path):
    calls = []
    def http(*args):
        calls.append(args)
        raise RuntimeError("lost connection")
    audit = tmp_path / "audit.jsonl"
    transport = RunpodQueueTransport("abc", "secret", Tokenizer(), audit,
                                      requests_max=1, http=http)
    with pytest.raises(RuntimeError, match="lost connection"):
        transport("unused", payload(), {}, 120)
    with pytest.raises(RuntimeError, match="budget exhausted"):
        transport("unused", payload(), {}, 120)
    assert len(calls) == 1
    assert json.loads(audit.read_text().splitlines()[-1])["reservation_retained"] is True


@pytest.mark.parametrize("changes", [{"model": "envoy-base-only"}, {"stream": True},
                                     {"max_tokens": 2048}, {"n": 2},
                                     {"chat_template_kwargs": {"enable_thinking": True}}])
def test_request_cannot_bypass_worker_limits(tmp_path, changes):
    transport = RunpodQueueTransport("abc", "secret", Tokenizer(), tmp_path / "audit")
    with pytest.raises(ValueError):
        transport("unused", payload(**changes), {}, 120)
    assert transport.remaining == 12


def test_context_limit_fails_before_submission(tmp_path):
    transport = RunpodQueueTransport("abc", "secret", Tokenizer(), tmp_path / "audit")
    with pytest.raises(ValueError, match="input limit"):
        transport("unused", payload(messages=[{"content": "x" * 7169}]), {}, 120)
    assert transport.remaining == 12
