import pytest

from benchmarks.envoybench.budget import InferenceBudget

CONFIG = {
    "max_estimated_usd": 0.01, "input_usd_per_million": 1,
    "output_usd_per_million": 3, "max_requests": 2, "max_input_utf8_bytes": 1000,
}
PAYLOAD = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 100}


def test_usage_settles_reservation_and_count_blocks_dispatch():
    calls = []
    def transport(*args):
        calls.append(args)
        return {"usage": {"prompt_tokens": 20, "completion_tokens": 10}}
    budget = InferenceBudget(CONFIG, transport)
    for _ in range(2):
        budget("url", PAYLOAD, {}, 1)
    assert budget.estimated_usd == pytest.approx(0.0001)
    assert budget.snapshot()["prompt_tokens"] == 40
    with pytest.raises(RuntimeError, match="count"):
        budget("url", PAYLOAD, {}, 1)
    assert len(calls) == 2


def test_cost_and_size_ceilings_block_before_transport():
    for config, message in [({"max_estimated_usd": 0.00001}, "cost"),
                            ({"max_input_utf8_bytes": 2}, "byte")]:
        budget = InferenceBudget({**CONFIG, **config}, lambda *a: pytest.fail("dispatched"))
        with pytest.raises(RuntimeError, match=message):
            budget("url", PAYLOAD, {}, 1)
        assert budget.requests == 0


def test_failed_request_retains_reservation_and_never_retries():
    calls = []
    def transport(*args):
        calls.append(args)
        raise RuntimeError("HTTP 402")
    budget = InferenceBudget(CONFIG, transport)
    with pytest.raises(RuntimeError, match="402"):
        budget("url", PAYLOAD, {}, 1)
    assert budget.estimated_usd > 0
    with pytest.raises(RuntimeError, match="no automatic retry"):
        budget("url", PAYLOAD, {}, 1)
    assert len(calls) == 1


def test_absent_usage_keeps_reservation_and_underestimate_stops_followup():
    budget = InferenceBudget(CONFIG, lambda *a: {})
    budget("url", PAYLOAD, {}, 1)
    assert budget.missing_usage_requests == 1
    assert budget.estimated_usd == pytest.approx((1029 + 300) / 1e6)
    budget = InferenceBudget(CONFIG, lambda *a: {
        "usage": {"prompt_tokens": 10000, "completion_tokens": 100},
    })
    budget("url", PAYLOAD, {}, 1)
    with pytest.raises(RuntimeError, match="exceeded"):
        budget("url", PAYLOAD, {}, 1)
