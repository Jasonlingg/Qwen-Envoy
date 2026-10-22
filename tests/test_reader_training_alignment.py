"""Data integrity and protocol coverage are not semantic or causal evidence."""
import copy
import json

import pytest

from scripts.audit_reader_training_alignment import audit, row_audit
from src.eval.reader_improvement import load


def pair():
    from scripts.audit_reader_training_alignment import ROOT
    data = ROOT / "data/sft/qasper-v5"
    manifest = load(data / "manifest.json")
    qid = manifest["splits"]["train"]["question_ids"][0]
    row = json.loads((data / "train.jsonl").read_text().splitlines()[0])
    candidates = {c["question_id"]: c for c in map(
        json.loads, (data / "candidates.jsonl").read_text().splitlines())}
    return row, candidates[qid]


def test_source_observation_or_action_changes_cannot_enter_a_verified_audit():
    row, candidate = pair()
    row["messages"][3]["content"] = "fabricated observation"
    with pytest.raises(ValueError, match="observations"):
        row_audit(row, candidate, "current prompt")
    row, candidate = pair()
    row["messages"][2]["content"] = "print('changed action')"
    with pytest.raises(ValueError, match="actions"):
        row_audit(row, candidate, "current prompt")


def test_uncertainty_marker_never_becomes_a_semantic_failure_label():
    row, candidate = pair()
    row, candidate = copy.deepcopy(row), copy.deepcopy(candidate)
    action = 'SUBMIT: I cannot establish this. CITATIONS: ["qasper_2001_05467"]'
    candidate["trajectory"][-1]["action"] = action
    row["messages"][-1]["content"] = action
    report = row_audit(row, candidate, "current prompt")
    assert report["uncertainty_phrase_flag"] is True
    assert report["expected_answerability"] == "sufficient"
    assert report["semantic_support_review"] is None
    assert report["terminal_is_submit_gap"] is False


def test_actual_v5_data_is_balanced_but_uses_the_old_terminal_contract():
    report = audit()
    train, val = report["splits"]["train"], report["splits"]["val"]
    assert train["answerability"] == {"sufficient": 19, "insufficient": 16}
    assert val["answerability"] == {"insufficient": 2, "sufficient": 7}
    assert (train["assistant_actions"], val["assistant_actions"]) == (106, 26)
    assert train["flags"]["insufficient_label_with_cited_prose"] == 16
    assert val["flags"]["insufficient_label_with_cited_prose"] == 2
    for split in (train, val):
        assert not split["flags"].get("terminal_has_evidence_spans", 0)
        assert not split["flags"].get("terminal_is_submit_gap", 0)
        assert not split["flags"].get("system_prompt_matches_current", 0)
        assert all(r["semantic_support_review"] is None for r in split["rows"])
    assert report["train_val_paper_overlap"] == []
    assert all(not x["question_overlap"] and not x["paper_overlap"]
               for x in report["evaluation_exclusions"])
    assert report["evaluation_exclusions"][2]["status"] == "consumed_by_completed_v6_comparison"
    assert report["training_performed"] is False and report["new_model_calls"] == 0


def test_changed_historical_split_is_rejected_without_training(monkeypatch):
    from scripts import audit_reader_training_alignment as runner
    original = runner.file_hash
    monkeypatch.setattr(runner, "file_hash", lambda path: "changed" if path.name == "train.jsonl"
                        else original(path))
    with pytest.raises(ValueError, match="training split changed"):
        runner.audit()


def test_existing_output_is_never_overwritten(tmp_path, monkeypatch):
    from scripts import audit_reader_training_alignment as runner
    output = tmp_path / "audit.json"
    output.write_text("preserve this")
    monkeypatch.setattr("sys.argv", ["audit", "--output", str(output)])
    with pytest.raises(SystemExit):
        runner.main()
    assert output.read_text() == "preserve this"
