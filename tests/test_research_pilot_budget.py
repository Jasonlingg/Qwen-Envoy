import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.eval.research_pilot_budget import BudgetStop, RequestBudget


def payload(**extra):
    return {"model": "envoy", "messages": [{"role": "user", "content": "question"}],
            "max_tokens": 32, **extra}


def response(prompt=12, completion=8):
    return {"choices": [{"message": {"content": "answer"}}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion}}


def test_complete_chat_and_output_are_reserved_before_upstream_and_observed():
    snapshots, counted, dispatched = [], [], []
    budget = RequestBudget(observer=snapshots.append)
    body = payload(messages=[{"role": "system", "content": "instructions"},
                             {"role": "user", "content": "first"},
                             {"role": "assistant", "content": "previous action"},
                             {"role": "user", "content": "tool observation"}])

    def count(messages):
        counted.append(messages)
        return 64

    def upstream(url, data, headers, timeout):
        assert snapshots[-1]["records"][0]["status"] == "in_flight"
        assert snapshots[-1]["reserved"] == {"requests": 1, "input_tokens": 64, "output_tokens": 32}
        dispatched.append((url, data, headers, timeout))
        return response()

    wrapped = budget.transport("qwen", count, 224, upstream)
    assert wrapped("https://private", body, {"Authorization": "secret"}, 12) == response()
    assert counted == [body["messages"]]
    assert dispatched == [("https://private", body, {"Authorization": "secret"}, 12)]
    snapshot = budget.summary()
    assert snapshot["actual_usage"] == {"prompt_tokens": 12, "completion_tokens": 8}
    assert snapshot["usage_complete"] is True
    row = snapshot["records"][0]
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    assert row["payload_sha256"] == hashlib.sha256(encoded).hexdigest()
    assert row["label"] == "qwen" and row["status"] == "ok"
    assert row["elapsed_seconds"] >= 0
    assert len(snapshots) == 2
    assert snapshots[0]["actual_usage"] == {"prompt_tokens": None, "completion_tokens": None}
    snapshot["records"].clear()
    snapshots[-1]["records"].clear()
    assert len(budget.summary()["records"]) == 1
    assert "secret" not in json.dumps(budget.summary())
    assert "question" not in json.dumps(budget.summary())
    assert "private" not in json.dumps(budget.summary())


def test_concurrent_host_and_qwen_requests_cannot_race_past_shared_cap():
    barrier, dispatched, lock = threading.Barrier(20), [], threading.Lock()
    budget = RequestBudget(max_requests=2, max_input_tokens=100, max_output_tokens=64)

    def count(_messages):
        barrier.wait(timeout=5)
        return 50

    def upstream(*_args):
        with lock:
            dispatched.append(1)
        return response()

    transports = [budget.transport(label, count, 1024, upstream) for label in ("host", "qwen")]

    def attempt(index):
        try:
            transports[index % 2]("unused", payload(), {}, 1)
            return "ok"
        except BudgetStop:
            return "blocked"

    with ThreadPoolExecutor(max_workers=20) as pool:
        outcomes = list(pool.map(attempt, range(20)))
    assert outcomes.count("ok") == len(dispatched) == 2
    assert outcomes.count("blocked") == 18
    assert budget.summary()["reserved"] == {"requests": 2, "input_tokens": 100, "output_tokens": 64}


@pytest.mark.parametrize("limits,body", [
    ({"max_requests": 1}, payload()),
    ({"max_input_tokens": 39}, payload()),
    ({"max_output_tokens": 63}, payload()),
])
def test_each_reservation_cap_is_hard_and_success_does_not_refund(limits, body):
    calls = []
    budget = RequestBudget(**limits)
    wrapped = budget.transport("host", lambda _messages: 20, 1024,
                               lambda *args: calls.append(args) or response())
    wrapped("unused", body, {}, 1)
    with pytest.raises(BudgetStop, match="limit"):
        wrapped("unused", body, {}, 1)
    assert len(calls) == 1
    assert budget.summary()["reserved"]["output_tokens"] == 32


def test_context_overflow_rejects_before_reservation_or_dispatch():
    calls = []
    budget = RequestBudget()
    wrapped = budget.transport("host", lambda messages: 64, 223,
                               lambda *args: calls.append(args))
    with pytest.raises(BudgetStop, match="context window"):
        wrapped("unused", payload(), {}, 1)
    assert not calls
    assert budget.summary()["reserved"]["requests"] == 0


@pytest.mark.parametrize("usage", [None, {}, {"prompt_tokens": 12},
                                    {"prompt_tokens": True, "completion_tokens": 8},
                                    {"prompt_tokens": 12, "completion_tokens": float("nan")},
                                    {"prompt_tokens": -1, "completion_tokens": 8}])
def test_unknown_usage_returns_existing_response_and_blocks_future_calls(usage):
    calls = []
    reply = {"choices": [{"message": {"content": "keep this evidence"}}], "usage": usage}
    budget = RequestBudget()
    wrapped = budget.transport("qwen", lambda _messages: 50, 1024,
                               lambda *args: calls.append(args) or reply)
    assert wrapped("unused", payload(), {}, 1) is reply
    with pytest.raises(BudgetStop, match="usage"):
        wrapped("unused", payload(), {}, 1)
    snapshot = budget.summary()
    assert len(calls) == 1
    assert snapshot["usage_complete"] is False
    assert snapshot["records"][0]["status"] == "usage_unknown"
    assert snapshot["reserved"]["output_tokens"] == 32
    assert snapshot["actual_usage"] == {"prompt_tokens": None, "completion_tokens": None}


def test_zero_provider_usage_is_known_while_missing_is_not_zero():
    budget = RequestBudget()
    wrapped = budget.transport("host", lambda _messages: 1, 1024, lambda *_args: response(0, 0))
    wrapped("unused", payload(), {}, 1)
    assert budget.summary()["actual_usage"] == {"prompt_tokens": 0, "completion_tokens": 0}
    assert budget.summary()["usage_complete"] is True
    unknown = budget.transport("host", lambda _messages: 1, 1024, lambda *_args: {})
    unknown("unused", payload(), {}, 1)
    snapshot = budget.summary()
    assert snapshot["actual_usage"] == {"prompt_tokens": None, "completion_tokens": None}
    assert snapshot["observed_usage"] == {"prompt_tokens": 0, "completion_tokens": 0}


@pytest.mark.parametrize("prompt,completion", [(51, 8), (12, 33)])
def test_usage_exceeding_reservation_blocks_future_dispatch(prompt, completion):
    budget = RequestBudget()
    wrapped = budget.transport("host", lambda _messages: 50, 1024,
                               lambda *_args: response(prompt, completion))
    wrapped("unused", payload(), {}, 1)
    assert budget.summary()["records"][0]["status"] == "usage_exceeded"
    assert budget.summary()["actual_usage"] == {
        "prompt_tokens": prompt, "completion_tokens": completion,
    }
    with pytest.raises(BudgetStop, match="exceeded"):
        wrapped("unused", payload(), {}, 1)


def test_transport_exception_keeps_reservation_records_only_class_and_never_retries():
    calls = []

    def upstream(*args):
        calls.append(args)
        raise TimeoutError("Authorization: secret; question content")

    budget = RequestBudget()
    wrapped = budget.transport("qwen", lambda _messages: 50, 1024, upstream)
    with pytest.raises(TimeoutError):
        wrapped("unused", payload(), {"Authorization": "secret"}, 1)
    with pytest.raises(BudgetStop, match="TimeoutError"):
        wrapped("unused", payload(), {}, 1)
    snapshot = budget.summary()
    assert len(calls) == 1
    assert snapshot["records"][0]["exception_class"] == "TimeoutError"
    assert snapshot["records"][0]["status"] == "error"
    assert snapshot["actual_usage"] == {"prompt_tokens": None, "completion_tokens": None}
    assert snapshot["reserved"] == {"requests": 1, "input_tokens": 50, "output_tokens": 32}
    assert "secret" not in json.dumps(snapshot)
    assert "question content" not in json.dumps(snapshot)


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, float("nan"), None])
def test_invalid_local_counts_do_not_dispatch(value):
    calls = []
    wrapped = RequestBudget().transport("qwen", lambda _messages: value, 1024,
                                        lambda *args: calls.append(args))
    with pytest.raises(ValueError, match="count"):
        wrapped("unused", payload(), {}, 1)
    assert not calls


@pytest.mark.parametrize("extra", [
    {"model": ""}, {"messages": []}, {"messages": [{"role": "user", "content": []}]},
    {"max_tokens": True}, {"max_tokens": 0}, {"max_tokens": 1025},
    {"max_tokens": float("nan")}, {"n": 2}, {"n": True}, {"stream": True},
    {"max_completion_tokens": 32}, {"temperature": float("nan")},
])
def test_invalid_or_unbudgeted_payload_forms_reject_before_dispatch(extra):
    calls = []
    wrapped = RequestBudget().transport("host", lambda _messages: 10, 1024,
                                        lambda *args: calls.append(args))
    with pytest.raises(ValueError):
        wrapped("unused", payload(**extra), {}, 1)
    assert not calls


@pytest.mark.parametrize("field", ["max_requests", "max_input_tokens", "max_output_tokens"])
@pytest.mark.parametrize("value", [True, 0, -1, float("nan")])
def test_invalid_constructor_limits_rejected(field, value):
    with pytest.raises(ValueError, match="positive integers"):
        RequestBudget(**{field: value})


def test_observer_failure_before_dispatch_blocks_without_leaking_error_details():
    calls = []

    def observer(_snapshot):
        raise OSError("private filesystem information")

    budget = RequestBudget(observer=observer)
    wrapped = budget.transport("host", lambda _messages: 10, 1024,
                               lambda *args: calls.append(args))
    with pytest.raises(BudgetStop, match="observer failed"):
        wrapped("unused", payload(), {}, 1)
    assert not calls
    snapshot = budget.summary()
    assert snapshot["records"][0]["status"] == "observer_error"
    assert snapshot["reserved"]["requests"] == 1
    assert "private filesystem information" not in json.dumps(snapshot)
