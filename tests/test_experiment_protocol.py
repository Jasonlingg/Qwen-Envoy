"""Model text is parsed as bounded literal requests, never executed on the host."""
import pytest

from src.product.experiment_protocol import controller_prompt, parse_request, validate_brief


@pytest.mark.parametrize("action", [
    '__import__("os").system("id")', 'search_papers(open("secret").read())',
    'import os', 'search_papers("x"); read_paper("2609.12345", "x")',
    'search_papers(**{"query":"x"})', 'search_papers(query="x", query="y")',
    'x = search_papers("x")', 'search_papers("x" * 100000)',
    'finish([x for x in []], [], {})', 'while True: pass',
])
def test_requests_cannot_execute_python(action):
    with pytest.raises(ValueError):
        parse_request(action)


def test_positional_and_keyword_search_requests():
    assert parse_request('search_papers("cache")') == parse_request(
        'search_papers(query="cache")') == ("search", {"query": "cache"})


def test_finish_requires_real_evidence_ids_and_labels_proposal():
    value = {"findings": [{"claim": "Memory decreased in the paper's setup.",
                            "evidence_ids": ["E1"]}],
             "uncertainties": ["Our hardware is not evaluated."], "proposed_test": {}}
    with pytest.raises(ValueError, match="returned evidence"):
        validate_brief(value, [])
    result = validate_brief(value, [{"evidence_id": "E1"}])
    assert result["semantic_support"] == "not_reviewed"
    assert result["proposal_status"] == "not_executed_or_validated"


def test_no_evidence_cannot_turn_into_experiment_recommendation():
    with pytest.raises(ValueError):
        validate_brief({"findings": [], "uncertainties": ["No evidence."],
                        "proposed_test": {"hypothesis": "try it"}}, [])


def test_effective_prompt_exposes_lowered_shared_limits():
    prompt = controller_prompt(12, 98304)
    assert "12 total model requests" in prompt and "98304 reserved tokens" in prompt
    assert "18 total model requests" not in prompt
