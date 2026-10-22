"""Keep the offline audit honest about reward exploits and saved provenance."""
import copy

import pytest

from src.eval import reader_rl_audit as audit
from src.eval.reader_improvement import load, write


@pytest.fixture
def example():
    doc_id = "qasper_2000_00001"
    text = "JavaScript " * 10 + "distractor " * 500
    spans = [{"doc_id": doc_id, "start": 0, "end": 100}]
    windows = [{"start": 0, "end": 1600}, {"start": 2000, "end": 3600}]
    context = {"doc_id": doc_id, "docs": {doc_id: {"text": text}},
        "annotations": [{"answer_type": "extractive", "answer_text": "JavaScript",
                         "unanswerable": False, "evidence": spans}],
        "windows": windows, "scope_answerability": "present"}
    record = {"action": audit.submit("JavaScript", doc_id, spans), "inspected_windows": windows}
    return context, record


def test_overlap_rewards_uninspected_gold_quote_but_inspection_gate_blocks_it(example):
    context, record = example
    record["inspected_windows"] = [context["windows"][-1]]
    scores = audit.score(record, context)
    assert scores["legacy_overlap"] == 1.0
    assert scores["scoped_overlap"] == scores["scoped_exact"] == 0.0


def test_exact_match_still_rewards_irrelevant_real_quote(example):
    context, record = example
    record["action"] = audit.submit("JavaScript", context["doc_id"],
                                    [{"doc_id": context["doc_id"], "start": 2000, "end": 2100}])
    scores = audit.score(record, context)
    assert scores["scoped_exact"] == 1.0
    assert scores["scoped_overlap"] == 0.5
    assert scores["source_support"] == "not_mechanically_proven"


def test_gap_shape_is_not_gap_usefulness_or_absence_proof(example):
    context, record = example
    context["scope_answerability"] = "missing"
    record["action"] = 'submit_gap(missing="Information unavailable", follow_up_query="research")'
    scores = audit.score(record, context)
    assert scores["legacy_overlap"] == 0.0  # Original paper annotation remains answerable.
    assert scores["scoped_overlap"] == scores["scoped_exact"] == 1.0
    context["scope_answerability"] = "present"
    scores = audit.score(record, context)
    assert scores["scoped_overlap"] == scores["scoped_exact"] == 0.0


@pytest.mark.parametrize("action", ["print('done')",
    "submit_gap(missing=str(1), follow_up_query='x')",
    "import os\nsubmit_gap(missing='x', follow_up_query='y')"])
def test_nonliteral_gap_does_not_execute_python_and_cannot_get_scoped_reward(example, action):
    context, record = example
    record["action"] = action
    scores = audit.score(record, context)
    assert scores["scoped_overlap"] == scores["scoped_exact"] == 0.0


def test_duplicate_reads_and_out_of_scope_spans_are_not_credited(example):
    context, record = example
    record["inspected_windows"] = [context["windows"][0]] * 2
    assert audit.score(record, context)["scoped_overlap"] == 0.0
    record["inspected_windows"] = context["windows"]
    # Full raw span must be inspected before quote clipping occurs.
    record["action"] = audit.submit("JavaScript", context["doc_id"],
                                    [{"doc_id": context["doc_id"], "start": 0, "end": 1700}])
    assert audit.score(record, context)["scoped_overlap"] == 0.0


def test_long_inspected_quote_matches_existing_runtime_clipping(example):
    context, record = example
    record["action"] = audit.submit("JavaScript", context["doc_id"],
                                    [{"doc_id": context["doc_id"], "start": 0, "end": 1600}])
    scores = audit.score(record, context)
    assert scores["terminal_eligibility"] == "answer"
    assert scores["scoped_overlap"] == pytest.approx(0.5 + 0.5 * 200 / 1300)


def test_order_sensitive_facts_are_indistinguishable_to_bag_of_words(example):
    context, record = example
    context["annotations"][0]["answer_text"] = "VATEX for video; WMT for translation"
    correct = audit.submit("VATEX for video; WMT for translation", context["doc_id"],
                           context["annotations"][0]["evidence"])
    swapped = audit.submit("WMT for video; VATEX for translation", context["doc_id"],
                           context["annotations"][0]["evidence"])
    record["action"] = correct
    a = audit.score(record, context)
    record["action"] = swapped
    b = audit.score(record, context)
    assert a["scoped_overlap"] == b["scoped_overlap"] == 1.0
    assert a["scoped_exact"] == 1.0 and b["scoped_exact"] == 0.0


def test_audit_retains_full_reward_exploit_as_a_failed_gate():
    good = {"id": "case:reviewed", "context_id": "case", "variant": "reviewed",
            "expected": "acceptable", "scores": {s: 1.0 for s in audit.SCORERS}}
    bad = {**good, "id": "case:bad", "variant": "generic_gap", "expected": "unacceptable"}
    summary = audit.summarize([good, bad])
    assert all(not s["adversarial_gate_passed"] for s in summary.values())
    assert all(s["unacceptable_full_reward"] == ["case:bad"] for s in summary.values())


def test_source_changes_block_offline_audit(tmp_path):
    (tmp_path / "source.json").write_text('{}')
    write(tmp_path / "validation.json", {"artifact_sha256": {"source.json": "wrong"}})
    with pytest.raises(ValueError, match="saved audit input changed"):
        audit.checked_bundle(tmp_path)


def test_no_training_cli_is_imported_to_inspect_legacy_sandbox():
    facts = audit.trainer_observations()
    assert facts["environment_constructors"][0]["use_docker"] is False
    assert facts["uses_current_investigator"] is facts["uses_submit_gap"] is False
    assert facts["training_launcher_executed"] is False


def test_acceptable_target_with_zero_score_is_reported_not_hidden():
    row = {"id": "case:reviewed", "context_id": "case", "variant": "reviewed",
           "expected": "acceptable", "scores": {s: 0.0 for s in audit.SCORERS}}
    result = audit.summarize([row])
    assert all(s["acceptable_zero_reward"] == [row["id"]] for s in result.values())


def test_mutating_inspection_counterfactual_does_not_rewrite_original(example):
    _, record = example
    saved = copy.deepcopy(record)
    proposed = {**record, "inspected_windows": []}
    assert record == saved and proposed["inspected_windows"] != record["inspected_windows"]


def test_real_saved_audit_is_reproducible_and_never_calls_a_model_or_repl(tmp_path, monkeypatch):
    import socket
    import subprocess

    from src.env.repl import PersistentREPL

    def forbidden(*args, **kwargs):
        raise AssertionError("offline audit cannot invoke networking, Python execution or training")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(PersistentREPL, "start_session", forbidden)
    destination = tmp_path / "audit"
    result = audit.audit(destination)
    assert result["records"] == 150
    assert result["origins"] == {"saved_reviewed_demonstration": 24,
        "authored_counterfactual_terminal_not_executed": 121, "saved_actual_v5": 3,
        "saved_replayed_correction": 2}
    assert [len(result["scorers"][s]["unacceptable_full_reward"]) for s in audit.SCORERS] == [
        12, 24, 36]
    assert all(result[k] == 0 for k in ("new_model_calls", "gpu_activations", "optimizer_updates"))
    assert result["training_reward_ready"] is False and result["human_review"] is None
    inputs = load(destination / "inputs.json")
    assert inputs["training_eligible"] is False
    pilot = [c for c in inputs["contexts"] if "--" in c]
    assert len(pilot) == 24 and all(c.startswith("qasper_train_") for c in pilot)
    results = load(destination / "results.json")
    original = next(r for r in results if r["variant"] == "reviewed" and
                    "ad7b1357" in r["id"] and "--present" in r["id"])
    swapped = next(r for r in results if r["variant"] == "swapped_dataset_tasks")
    assert original["scores"]["scoped_overlap"] == swapped["scores"]["scoped_overlap"]
    with pytest.raises(FileExistsError):
        audit.audit(destination)


def test_supplemental_history_flags_prior_pool_membership_and_missing_history(tmp_path):
    path = tmp_path / "out/research/train-pool/benchmark.json"
    path.parent.mkdir(parents=True)
    write(path, {"questions": [{"id": "qasper_train_" + "1" * 40,
                               "target_doc_ids": ["qasper_2000_00001"]}]})
    registry = {"files": [], "excluded_doc_ids": [], "excluded_question_ids": []}
    cases = [{"case_id": "training-pair", "doc_id": "qasper_2000_00001",
              "question_id": "qasper_train_" + "1" * 40, "role": "train"},
             {"case_id": "dev", "doc_id": "qasper_2000_00002",
              "question_id": "qasper_validation_" + "2" * 40, "role": "development"}]
    result = audit.supplemental_history(registry, cases, tmp_path)
    assert result["affected_case_ids"] == ["training-pair"]
    assert result["affected_development_papers"] == []
    assert result["missing_local_required_history"]
    assert result["training_ready"] is False
