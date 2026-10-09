import pytest

from scripts.run_claude_qasper_eval import PRICES, BudgetStop, SpendCap


def make_cap(tmp_path, cap_usd=1.0, model="claude-sonnet-5"):
    return SpendCap(model, cap_usd, tmp_path / "spend.json")


def test_settle_charges_actual_usage_at_the_model_rate(tmp_path):
    cap = make_cap(tmp_path)
    reserved = cap.reserve(input_chars=4_000, max_output_tokens=1_000)
    cost = cap.settle(reserved, [{"input_tokens": 1_000_000, "output_tokens": 100_000}])
    in_rate, out_rate = PRICES["claude-sonnet-5"]
    assert cost == pytest.approx(in_rate + 0.1 * out_rate)
    assert cap.in_flight == pytest.approx(0)
    assert cap.spent == pytest.approx(cost)


def test_reservation_blocks_a_request_the_cap_cannot_cover(tmp_path):
    cap = make_cap(tmp_path, cap_usd=0.01)
    with pytest.raises(BudgetStop):
        cap.reserve(input_chars=400_000, max_output_tokens=4_096)


def test_in_flight_reservations_count_against_the_cap(tmp_path):
    cap = make_cap(tmp_path, cap_usd=0.05)  # one worst-case reservation is ~$0.042
    cap.reserve(input_chars=1_000, max_output_tokens=4_096)
    with pytest.raises(BudgetStop):
        cap.reserve(input_chars=1_000, max_output_tokens=4_096)


def test_failed_request_with_no_usage_releases_its_reservation_for_free(tmp_path):
    cap = make_cap(tmp_path)
    reserved = cap.reserve(input_chars=1_000, max_output_tokens=4_096)
    assert cap.settle(reserved, []) == 0
    assert cap.in_flight == pytest.approx(0) and cap.spent == 0


def test_spend_survives_a_restart(tmp_path):
    cap = make_cap(tmp_path)
    cap.settle(cap.reserve(1_000, 100), [{"input_tokens": 500_000, "output_tokens": 0}])
    resumed = make_cap(tmp_path)
    assert resumed.spent == pytest.approx(cap.spent)


def test_unpriced_model_is_refused(tmp_path):
    with pytest.raises(ValueError):
        SpendCap("claude-opus-5", 1.0, tmp_path / "spend.json")
