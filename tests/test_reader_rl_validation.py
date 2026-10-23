"""Verify reward counterexamples and honest, non-launching readiness reporting."""
import copy
import json
import socket
import subprocess
from pathlib import Path

import pytest

from src.eval import reader_rl_validation as validation
from src.eval.qasper_reward import parse_strict
from src.eval.reader_improvement import digest, load
from src.eval.reader_rl_audit import submit

KEYS = Path("examples/reader-rl-validation/reviewed-keys.json")


@pytest.fixture
def saved():
    inputs = load(validation.AUDIT / "inputs.json")
    keys = load(KEYS)
    cards = validation.check_keys(keys, inputs)
    return inputs, keys, cards


def selected(saved, prefix="ad7b1357"):
    inputs, _, cards = saved
    record = next(r for r in inputs["records"] if prefix in r["id"]
                  and r["expected"] == "acceptable" and r["context_id"] in cards)
    return copy.deepcopy(record), copy.deepcopy(inputs["contexts"][record["context_id"]]), cards[
        record["context_id"]]


def test_order_binding_negation_extra_claims_and_wrong_answers_do_not_get_reward(saved):
    inputs, _, cards = saved
    negatives = [r for r in inputs["records"] if r["expected"] == "unacceptable"
                 and r["context_id"] in cards]
    assert len(negatives) == 74
    assert all(validation.assess(r, inputs["contexts"][r["context_id"]],
                                cards[r["context_id"]])["reward"] == 0 for r in negatives)
    assert any(r["variant"] == "swapped_dataset_tasks" for r in negatives)


def test_original_positive_disagreement_is_visible_and_narrow_repair_passes(saved):
    record, context, card = selected(saved, "515e10a7")
    assert "independent" in record["action"]
    assert card["original_control_review"] == "needs_support_revision"
    assert validation.assess(record, context, card)["reward"] == 0
    _, citations, spans = parse_strict(record["action"])
    for variant in card["accepted_answers"]:
        assert "independent" not in variant["answer"]
        proposed = {**record, "action": submit(variant["answer"], citations[0], spans)}
        assert validation.assess(proposed, context, card)["reward"] == 1


def test_missing_scope_has_no_numeric_reward_even_for_well_formed_gaps(saved):
    inputs, _, cards = saved
    missing = [r for r in inputs["records"] if r["context_id"] not in cards]
    assert len(missing) == 50
    assert all(validation.assess(r, inputs["contexts"][r["context_id"]], None) == {
        "eligible": False, "reward": None, "reason": "gap_quality_not_scored"} for r in missing)


def test_support_must_survive_actual_packet_quote_clipping(saved):
    record, context, card = selected(saved, "4c1847f0")
    answer, citations, _ = parse_strict(record["action"])
    support = card["claims"][0]["support"][0]
    # The raw citation includes the key, but its first 1,200 characters do not.
    start, end = support["start"] - 1250, support["end"]
    context["windows"] = [{"start": start, "end": end}]
    record["inspected_windows"] = context["windows"]
    record["action"] = submit(answer, citations[0], [
        dict(doc_id=citations[0], start=start, end=end)])
    card = {**card, "context_sha256": digest(context)}
    result = validation.assess(record, context, card)
    assert result["reward"] == 0
    assert result["reason"] == "reviewed_support_missing_after_quote_clipping"


def test_valid_single_inspection_is_sufficient_and_no_duplicate_read_reward(saved):
    record, context, card = selected(saved, "4c1847f0")
    _, _, spans = parse_strict(record["action"])
    window = {k: spans[0][k] for k in ("start", "end")}
    record["inspected_windows"] = [window]
    assert validation.assess(record, context, card)["reward"] == 1
    record["inspected_windows"] = [window, window]
    assert validation.assess(record, context, card)["reward"] == 1
    record["inspected_windows"] = []
    assert validation.assess(record, context, card)["reward"] == 0


def test_cannot_pad_a_good_answer_with_an_unrelated_real_citation(saved):
    record, context, card = selected(saved, "4c1847f0")
    answer, citations, spans = parse_strict(record["action"])
    other = record["inspected_windows"][-1]
    extra = {"doc_id": citations[0], "start": other["start"], "end": other["start"] + 50}
    record["action"] = submit(answer, citations[0], spans + [extra])
    result = validation.assess(record, context, card)
    assert result["reward"] == 0
    assert result["reason"] == "additional_citation_without_reviewed_support"


def test_receipts_cannot_be_borrowed_from_another_question(saved):
    record, context, card = selected(saved)
    record["context_id"] = "other-question"
    with pytest.raises(ValueError, match="stale"):
        validation.assess(record, context, card)


@pytest.mark.parametrize("bounds", [(-1, 10), (True, 6000), (5326, False),
                                    (5326.0, 5990), (5326, 999999), (5990, 5326)])
def test_malformed_or_uninspected_spans_fail_closed(saved, bounds):
    record, context, card = selected(saved, "4c1847f0")
    answer, citations, _ = parse_strict(record["action"])
    record["action"] = submit(answer, citations[0], [dict(doc_id=citations[0],
                                                       start=bounds[0], end=bounds[1])])
    assert validation.assess(record, context, card)["reward"] == 0


@pytest.mark.parametrize("value", ["the question", "source text", "scope"])
def test_changed_source_question_or_scope_is_an_input_error_not_a_bad_episode(saved, value):
    record, context, card = selected(saved)
    if value == "the question":
        context["question"] += " changed"
    elif value == "source text":
        context["docs"][context["doc_id"]]["text"] += " changed"
    else:
        context["windows"][0]["start"] += 1
    with pytest.raises(ValueError, match="stale"):
        validation.assess(record, context, card)


def test_source_key_tamper_duplicate_context_and_missing_claim_review_rejected(saved):
    inputs, original, _ = saved
    for kind in ["quote", "duplicate", "claims", "variants"]:
        keys = copy.deepcopy(original)
        card = keys["cards"][0]
        if kind == "quote":
            card["claims"][0]["support"][0]["quote"] += " invented"
        elif kind == "duplicate":
            keys["cards"][1] = card
        elif kind == "claims":
            card["claims"] = []
        else:
            card["accepted_answers"][0]["claim_ids"] = []
        with pytest.raises(ValueError):
            validation.check_keys(keys, inputs)


def test_numeric_units_and_punctuation_are_not_erased():
    assert validation.canonical("  64 MB\nper request. ") == "64 mb per request."
    for a, b in [("64 MB", "64 GB"), ("0.2", "02"), ("a>b", "a<b"),
                 ("not supported", "supported"), ("VATEX/video", "video/VATEX")]:
        assert validation.canonical(a) != validation.canonical(b)


def test_unlisted_valid_paraphrase_is_zero_without_calling_it_false(saved):
    record, context, card = selected(saved, "4c1847f0")
    _, citations, spans = parse_strict(record["action"])
    record["action"] = submit("The implementation language is JavaScript.", citations[0], spans)
    assert validation.assess(record, context, card)["reason"] == "answer_not_in_reviewed_variants"


def test_static_generation_fix_inspection_does_not_trust_comments(tmp_path):
    path = tmp_path / "trainer.py"
    path.write_text("# use_model_defaults=False\ndef generation_kwargs():\n"
                    " return {'use_model_defaults': True}\n")
    assert validation.generation_fix(path) is False
    path.write_text("def generation_kwargs():\n return {'use_model_defaults': False}\n")
    assert validation.generation_fix(path) is True


def test_full_offline_report_preserves_failures_and_cannot_launch(tmp_path, monkeypatch):
    from src.env.repl import PersistentREPL

    def forbidden(*args, **kwargs):
        raise AssertionError("no network, subprocess, Docker or model launch")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(PersistentREPL, "start_session", forbidden)
    output = tmp_path / "report"
    summary = validation.validate(KEYS, output)
    assert summary["records"] == 150 and summary["eligible_answerable_records"] == 100
    assert summary["known_answerable_negatives_positive"] == 0
    assert summary["original_positive_controls_rewarded"] == 13
    assert summary["reviewed_variants_rewarded"] == summary["reviewed_complete_variants"] == 28
    assert len(summary["source_review_disagreements"]) == 1
    assert summary["exposed_development_gate_passed"] is True
    assert summary["training_reward_ready"] is False
    report = load(output / "preflight.json")
    assert report["status"] == "blocked" and report["training_ready"] is False
    assert all(report["frozen_runtime"].values()) and report["v1_preamble_unchanged"]
    assert report["legacy_trainer"]["generation_defaults_override_fix_present"]
    assert report["legacy_trainer"]["environment_constructors"][0]["use_docker"] is False
    with pytest.raises(FileExistsError):
        validation.validate(KEYS, output)


def test_cli_reports_not_ready_as_nonzero(tmp_path):
    result = subprocess.run(["python", "scripts/validate_reader_rl.py", "--reviewed-keys",
        str(KEYS), "--output", str(tmp_path / "report")], capture_output=True, text=True)
    assert result.returncode == 2, result.stderr
    assert json.loads(result.stdout)["training_reward_ready"] is False


def test_cannot_write_reports_inside_previous_evidence():
    with pytest.raises(ValueError, match="separate"):
        validation.validate(KEYS, validation.AUDIT / "new-output")
