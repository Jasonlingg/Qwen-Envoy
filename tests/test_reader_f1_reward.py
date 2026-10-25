"""Candidate reward checks and lexical counterexamples; no model evaluation."""

import copy
from pathlib import Path

import pytest

from scripts.audit_reader_f1_reward import audit
from src.eval.reader_improvement import digest
from src.eval.reader_rl_audit import submit
from src.training.reader_data import load_bundle
from src.training.reader_f1_reward import assess_f1


@pytest.fixture(scope="module")
def source():
    _, roles, contexts, cards = load_bundle(Path("data/rl/reader-facts-v1"))
    cid = roles["reward_validation"][1]["case_id"]
    context, card = contexts[cid], cards[cid]
    spans = [{"doc_id": context["doc_id"], "start": s["start"], "end": s["end"]}
             for claim in card["claims"] for s in claim["support"]]
    record = {"context_id": cid, "action": submit("1500 sentences", context["doc_id"], spans),
              "inspected_windows": context["windows"]}
    return record, context, card


def test_candidate_scores_original_answer_without_mutation(source):
    record, context, card = copy.deepcopy(source)
    before = digest((record, context, card))
    record["action"] = record["action"].replace("1500 sentences", "A corpus of 1500 sentences.")
    changed = digest((record, context, card))
    result = assess_f1(record, context, card)
    assert 0 < result["reward"] < 1
    assert result["reward"] == result["answer_f1"]
    assert digest((record, context, card)) == changed != before
    assert result["training_ready"] is False


@pytest.mark.parametrize("action", [None, "SUBMIT: 1500 sentences", "x" * 8001])
def test_invalid_terminal_zero(source, action):
    record, context, card = copy.deepcopy(source)
    record["action"] = action
    assert assess_f1(record, context, card)["reward"] == 0


def test_original_citations_not_repaired_by_private_reference_substitution(source):
    record, context, card = copy.deepcopy(source)
    record["action"] = record["action"].replace(
        'CITATIONS: ["' + context["doc_id"] + '"]', 'CITATIONS: ["wrong-document"]')
    result = assess_f1(record, context, card)
    assert result["reward"] == 0 and result["reason"] == "invalid_citations"


def test_missing_scope_unavailable_and_stale_or_locked_references_error(source):
    record, context, card = copy.deepcopy(source)
    missing = {**context, "scope_answerability": "missing"}
    assert assess_f1(record, missing, None)["reward"] is None
    with pytest.raises(ValueError):
        assess_f1(record, {**context, "question": "changed"}, card)
    with pytest.raises(ValueError):
        assess_f1(record, context, {**card, "role": "model_evaluation_locked"})


def test_numeric_normalization_collision_is_reported_not_hidden(source):
    record, context, card = copy.deepcopy(source)
    record["action"] = record["action"].replace("1500 sentences", "15.00 sentences")
    assert assess_f1(record, context, card)["reward"] == 1


def test_audit_keeps_original_boundaries_and_exposes_failed_development_gate():
    report = audit(Path("data/rl/reader-facts-v1"))
    assert len(report["rows"]) == 65
    assert {r["role"] for r in report["rows"]} == {"reward_validation"}
    assert report["checks"]["reference_controls_full_credit"]
    assert report["checks"]["proposed_paraphrases_nonzero"]
    assert report["checks"]["invalid_evidence_zero"]
    assert not report["checks"]["authored_wrong_answers_less_than_full_credit"]
    assert not report["development_gate_passed"]
    assert not report["training_ready"]
