"""Offline fault injection for one paid attempt and provisional judge comparison."""
import json
from pathlib import Path

import pytest

from src.eval.research_judge_plan import MODEL, encoded, prepare, request_payload
from src.eval.research_judge_run import run, validate_preflight
from src.eval.research_judge_score import compare, score_saved

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "benchmarks/research_judge_checks_v1"
CATALOG = [{"status": "active", "flavors": [{"model_id": MODEL,
    "input_price_per_million_tokens": 0.3, "output_price_per_million_tokens": 0.9,
    "max_model_len": 262_144}]}]
RECEIPT = {"usage_complete_and_within_reservation": True, "served_model_matches_request": True}


def read(name):
    return json.loads((ROOT / f"{name}.json").read_text())


def response(rows):
    return {"model": MODEL, "usage": {"prompt_tokens": 40000, "completion_tokens": 6000},
        "choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps({"rows": rows})}}]}


def expected_response():
    """Synthetic oracle only to test comparison logic; never an observed judge result."""
    return response([{"case_id": r["case_id"], "fact_judgments": r["fact_judgments"],
        "all_material_assertions_supported": r["all_material_assertions_supported"],
        "reason": "Offline synthetic test."} for r in read("labels")])


@pytest.fixture
def prepared(tmp_path):
    folder = tmp_path / "run"
    prepare(ROOT, folder, CATALOG, repo=REPO)
    return folder


def test_request_contains_only_exported_inputs_and_one_copy_per_paper():
    payload = request_payload(ROOT)
    data = json.loads(payload["messages"][1]["content"])
    assert len(data["cases"]) == 32 and len(data["sources"]) == 2
    assert all("source_text" not in r for r in data["cases"])
    assert all(k not in payload["messages"][1]["content"] for k in (
        '"expected_score"', '"critical_type"', '"all_material_assertions_supported"'))
    assert payload["n"] == 1 and payload["max_completion_tokens"] == 16384


def test_success_consumes_only_attempt_and_persists_usage(prepared):
    calls = []
    def transport(url, **kwargs):
        calls.append(url)
        return expected_response()
    receipt = run(prepared, ROOT, REPO, "fake-key", transport=transport, catalog=CATALOG,
                      allowance_path=prepared.parent / "allowance.json")
    assert receipt["catalog_cost_usd"] == "0.0174"
    assert receipt["usage_complete_and_within_reservation"]
    assert receipt["served_model_matches_request"]
    assert len(calls) == 1
    with pytest.raises(FileExistsError):
        run(prepared, ROOT, REPO, "fake-key", transport=transport, catalog=CATALOG,
                      allowance_path=prepared.parent / "allowance.json")
    assert len(calls) == 1


def test_timeout_consumes_attempt_and_unknown_usage_is_not_zero(prepared):
    calls = []
    def transport(url, **kwargs):
        calls.append(url)
        raise TimeoutError()
    receipt = run(prepared, ROOT, REPO, "fake-key", transport=transport, catalog=CATALOG,
                      allowance_path=prepared.parent / "allowance.json")
    assert receipt["usage"] is None and receipt["catalog_cost_usd"] is None
    with pytest.raises(FileExistsError):
        run(prepared, ROOT, REPO, "fake-key", transport=transport, catalog=CATALOG,
                      allowance_path=prepared.parent / "allowance.json")
    assert len(calls) == 1


@pytest.mark.parametrize("mutation", ["price", "request", "implementation", "tokens"])
def test_preflight_failure_prevents_paid_dispatch(prepared, mutation):
    catalog = json.loads(json.dumps(CATALOG))
    if mutation == "price":
        catalog[0]["flavors"][0]["input_price_per_million_tokens"] = 2
    elif mutation == "request":
        path = prepared / "request.json"
        payload = json.loads(path.read_text())
        payload["max_completion_tokens"] += 1
        path.write_bytes(encoded(payload))
    else:
        path = prepared / "manifest.json"
        manifest = json.loads(path.read_text())
        if mutation == "tokens":
            manifest["input_tokens_reserved"] += 1
        else:
            manifest["implementation_sha256"]["src/eval/research_judge_run.py"] = "0" * 64
        path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        validate_preflight(prepared, ROOT, REPO, catalog)
    assert not (prepared / "dispatch.json").exists()


def test_expected_labels_can_check_scorer_but_are_not_judge_evidence():
    result = compare(ROOT, expected_response(), RECEIPT)
    assert result["scores_agree"] == result["fact_labels_agree"] == 32
    assert result["status"] == "scores_match_unattested_not_a_gate_result"
    assert result["training_approved"] is False


def test_unknown_and_missing_fail_gate_without_discarding_other_results():
    value = expected_response()
    rows = json.loads(value["choices"][0]["message"]["content"])["rows"]
    rows.pop()
    rows[0]["fact_judgments"][0]["correct_and_supported"] = None
    result = compare(ROOT, response(rows), RECEIPT)
    assert result["scores_agree"] == 30
    assert result["metrics"]["missing_cases"] == 1
    assert result["metrics"]["unknown_labels_or_judgments_on_clear_validation_cases"] == 2
    assert result["results"][0]["judge_score"] is None
    assert result["status"] == "sentinel_gate_failed"


def test_critical_false_acceptance_and_partial_error_counted():
    value = expected_response()
    rows = json.loads(value["choices"][0]["message"]["content"])["rows"]
    labels = read("labels")
    for group in ("critical_negative", "partial"):
        i = next(i for i,r in enumerate(labels) if r["group"] == group)
        rows[i]["all_material_assertions_supported"] = True
        for fact in rows[i]["fact_judgments"]:
            fact["correct_and_supported"] = True
    result = compare(ROOT, response(rows), RECEIPT)
    assert result["metrics"]["critical_false_acceptances"] == 1
    assert result["metrics"]["incorrect_partial_C_scores"] == 1
    assert result["status"] == "sentinel_gate_failed"


@pytest.mark.parametrize("fault", ["duplicate", "truncated", "unknown_case", "bad_type"])
def test_corrupt_rows_or_response_never_pass(fault):
    value = expected_response()
    rows = json.loads(value["choices"][0]["message"]["content"])["rows"]
    if fault == "duplicate":
        rows.append(rows[0].copy())
    elif fault == "unknown_case":
        rows.append({**rows[0], "case_id": "other"})
    elif fault == "bad_type":
        rows[0]["fact_judgments"][0]["correct_and_supported"] = 1
    value = response(rows)
    if fault == "truncated":
        value["choices"][0]["finish_reason"] = "length"
    assert compare(ROOT, value, RECEIPT)["status"] == "sentinel_gate_failed"


def test_missing_usage_or_wrong_model_fails_even_when_scores_match():
    assert compare(ROOT, expected_response(), {})["status"] == "sentinel_gate_failed"


def test_alternate_directory_cannot_refresh_paid_allowance(prepared):
    other = prepared.parent / "another-run"
    prepare(ROOT, other, CATALOG, repo=REPO)
    calls = []
    def transport(url, **kwargs):
        calls.append(url)
        return expected_response()
    allowance = prepared.parent / "allowance.json"
    run(prepared, ROOT, REPO, "fake", transport=transport, catalog=CATALOG,
        allowance_path=allowance)
    with pytest.raises(FileExistsError):
        run(other, ROOT, REPO, "fake", transport=transport, catalog=CATALOG,
            allowance_path=allowance)
    assert len(calls) == 1 and not (other / "dispatch.json").exists()


@pytest.mark.parametrize("tamper", [None, "response", "receipt", "model"])
def test_named_judge_gate_requires_saved_run_bindings(prepared, tamper):
    value = expected_response()
    run(prepared, ROOT, REPO, "fake", transport=lambda *a, **kw: value, catalog=CATALOG,
        allowance_path=prepared.parent / "allowance.json")
    if tamper in {"response", "model"}:
        path = prepared / "response.json"
        value = json.loads(path.read_text())
        value["model" if tamper == "model" else "id"] = "another-value"
        path.write_text(json.dumps(value))
    elif tamper == "receipt":
        path = prepared / "receipt.json"
        value = json.loads(path.read_text())
        value["request_sha256"] = "0" * 64
        path.write_text(json.dumps(value))
    result = score_saved(prepared, ROOT, REPO)
    assert result["status"] == ("inconclusive_binding_or_identity_failure" if tamper
                                else "alias_sentinel_scores_passed_not_training_readiness")


def test_malformed_unidentified_row_does_not_discard_other_judgments():
    rows = json.loads(expected_response()["choices"][0]["message"]["content"])["rows"]
    result = compare(ROOT, response([*rows, {"nonsense": True}]), RECEIPT)
    assert result["scores_agree"] == 32 and result["provider_or_schema_error"]
    assert result["status"] == "sentinel_gate_failed"


@pytest.mark.parametrize("fault", ["redirect", "oversized"])
def test_transport_rejects_redirect_and_bounds_response(monkeypatch, fault):
    import httpx

    from src.eval.research_judge_plan import LIMITS, URL
    from src.eval.research_judge_run import bounded_request
    seen = []
    def handler(request):
        seen.append(request)
        if fault == "redirect":
            return httpx.Response(302, headers={"Location": "https://elsewhere.invalid"})
        return httpx.Response(200, content=b"x" * 101)
    monkeypatch.setattr(httpx, "HTTPTransport", lambda **kw: httpx.MockTransport(handler))
    monkeypatch.setitem(LIMITS, "response_bytes", 100)
    with pytest.raises((httpx.HTTPStatusError, ValueError)):
        bounded_request(URL, payload={"test": True}, api_key="fake")
    assert len(seen) == 1 and seen[0].url.host == "api.tokenfactory.us-central1.nebius.com"


def test_fact_order_is_irrelevant_but_duplicate_ids_are_rejected():
    rows = json.loads(expected_response()["choices"][0]["message"]["content"])["rows"]
    for row in rows:
        row["fact_judgments"].reverse()
    result = compare(ROOT, response(rows), RECEIPT)
    assert result["scores_agree"] == result["fact_labels_agree"] == 32
    rows[0]["fact_judgments"][0] = rows[0]["fact_judgments"][1].copy()
    result = compare(ROOT, response(rows), RECEIPT)
    assert result["scores_agree"] == 31
    assert result["metrics"]["unknown_labels_or_judgments_on_clear_validation_cases"] == 1


@pytest.mark.parametrize("body", [None, [], {"usage": None}, {"usage": []}])
def test_malformed_saved_body_or_usage_is_inconclusive(prepared, body):
    run(prepared, ROOT, REPO, "fake", transport=lambda *a, **kw: body, catalog=CATALOG,
        allowance_path=prepared.parent / "allowance.json")
    result = score_saved(prepared, ROOT, REPO)
    assert result["status"] == "inconclusive_binding_or_identity_failure"
    assert result["scores_agree"] == 0 and result["metrics"]["missing_cases"] == 32
