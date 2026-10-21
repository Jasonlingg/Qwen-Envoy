"""Reservations bound both inference roles, including malformed/failed responses."""
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.product.experiment_budget import BudgetExhaustedError, ExperimentBudget


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": True, "add_generation_prompt": True,
                          "enable_thinking": False, "return_dict": False}
        return [1] * 20


PAYLOAD = {"messages": [], "max_tokens": 1024,
           "chat_template_kwargs": {"enable_thinking": False}}


def test_shared_reservations_no_refund_and_no_retry():
    calls = []
    def failing(*args):
        calls.append(args)
        raise TimeoutError("outcome unknown")
    budget = ExperimentBudget(Tokenizer(), failing, requests_max=3, tokens_max=8192)
    with pytest.raises(TimeoutError):
        budget("url", PAYLOAD, {}, 1)
    with pytest.raises(BudgetExhaustedError):
        budget("url", PAYLOAD, {}, 1)
    assert len(calls) == 1 and budget.remaining == 0
    assert budget.summary()["records"][0]["status"] == "failed_or_uncertain"
    assert not budget.summary()["usage_complete"]


def test_parallel_attempts_share_atomic_budget():
    calls = []
    def transport(*args):
        calls.append(args)
        return {"usage": {"prompt_tokens": 20, "completion_tokens": 8}}
    budget = ExperimentBudget(Tokenizer(), transport, requests_max=2)
    def attempt(_):
        try:
            budget("url", PAYLOAD, {}, 1)
            return True
        except BudgetExhaustedError:
            return False
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(attempt, range(10))) == 2
    assert len(calls) == 2 and budget.summary()["tokens_reserved"] == 16384
    assert budget.summary()["usage_complete"]


@pytest.mark.parametrize("change", [{"max_tokens": 2048}, {"stream": True}, {"n": 2},
                                   {"chat_template_kwargs": {"enable_thinking": True}}])
def test_unbounded_request_rejected_before_dispatch(change):
    budget = ExperimentBudget(Tokenizer(), lambda *args: pytest.fail("dispatched"))
    with pytest.raises(ValueError):
        budget("url", {**PAYLOAD, **change}, {}, 1)
    assert budget.summary()["requests_dispatched"] == 0


def test_oversized_prompt_rejected_before_dispatch():
    class Oversized(Tokenizer):
        def apply_chat_template(self, *args, **kwargs):
            return [1] * 7169
    budget = ExperimentBudget(Oversized(), lambda *args: pytest.fail("dispatched"))
    with pytest.raises(ValueError, match="7168"):
        budget("url", PAYLOAD, {}, 1)
    assert budget.remaining == 18
