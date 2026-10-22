"""Failure review must not silently become training data or semantic validation."""
import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.eval.reader_improvement import (
    build_queue,
    check,
    check_review,
    digest,
    prepare,
    review_template,
    write,
)

BUNDLE = Path(__file__).resolve().parents[1] / "examples/qasper-hybrid-v5"


@pytest.fixture(scope="module")
def saved_queue():
    return build_queue(BUNDLE)


def completed_review(queue):
    review = review_template(queue)
    review.update(reviewer_kind="model_assisted", reviewer_identity="test fixture",
                  reviewed_at="2026-10-22T17:00:00Z")
    for row in review["rows"]:
        row.update(verdict="pass", category="none", notes="Synthetic test judgment")
    return review


def correction(queue, review, *, gap=False):
    record, row = queue["records"][0], review["rows"][0]
    window = record["original_packet"]["inspected_passage_windows"][0]
    start, end = window["start"], window["start"] + 30
    span = {"doc_id": record["doc_id"], "start": start, "end": end,
            "quote": record["source"]["text"][start:end]}
    evidence = [{k: span[k] for k in ("doc_id", "start", "end")}]
    action = (f'SUBMIT: Synthetic proposed answer. CITATIONS: {json.dumps([record["doc_id"]])}'
              f' EVIDENCE: {json.dumps(evidence)}')
    if gap:
        action = ('submit_gap(missing="Selected passages do not establish the requested fact.", '
                  'follow_up_query="Search for a study measuring the requested property")')
    row.update(verdict="fail", category="gap_reporting" if gap else "evidence_not_used",
               notes="Synthetic fixture, not an actual semantic review",
               correction={"replacement_final_action": action, "rationale": "Fixture rationale",
                           "source_support_reviewed": True, "supporting_spans": [span]})
    return row["correction"]


def test_actual_bundle_preserves_all_cases_and_exact_final_input(saved_queue):
    assert len(saved_queue["records"]) == 3
    for record in saved_queue["records"]:
        assert record["data_policy"] == {"usage": "regression_only", "training_eligible": False}
        assert record["semantic_verdict"] is None
        assert len(record["original_packet"]["trajectory"]) == 3
        assert record["actual_final_request"]["messages"][0]["role"] == "system"
    humor = saved_queue["records"][0]
    assert "alliteration" in json.dumps(humor["actual_final_request"]).lower()
    assert "antonymy" in json.dumps(humor["actual_final_request"]).lower()


def test_review_and_correction_remain_quarantined_even_with_true_attestation(saved_queue):
    review = completed_review(saved_queue)
    correction(saved_queue, review)
    report = check_review(saved_queue, review)
    assert report["human_review"] is None
    assert report["independent_human_review"] is None
    assert report["training_ready"] == 0
    candidate, = report["correction_drafts"]
    assert candidate["training_eligible"] is False
    assert candidate["docker_replay"] == "required_not_performed"
    assert "messages" not in candidate and "trajectory" not in candidate
    assert candidate["source_support"] == "reviewer_attestation_not_mechanically_proven"


@pytest.mark.parametrize("field,value", [
    ("reviewer_kind", None), ("reviewer_kind", "independent_human"),
    ("reviewer_identity", ""), ("reviewed_at", None), ("queue_sha256", "changed"),
])
def test_missing_or_unbound_attribution_is_rejected(saved_queue, field, value):
    review = completed_review(saved_queue)
    review[field] = value
    with pytest.raises(ValueError):
        check_review(saved_queue, review)


def test_human_attribution_never_asserts_independence(saved_queue):
    review = completed_review(saved_queue)
    review["reviewer_kind"] = "human"
    report = check_review(saved_queue, review)
    assert report["human_review"] == "self_reported"
    assert report["independent_human_review"] is None


@pytest.mark.parametrize("change", ["duplicate", "missing", "null_verdict", "bad_hash",
                                     "missing_notes", "bad_category", "pass_with_correction"])
def test_incomplete_or_changed_case_reviews_fail_closed(saved_queue, change):
    review = completed_review(saved_queue)
    if change == "duplicate":
        review["rows"][1] = review["rows"][0]
    elif change == "missing":
        review["rows"].pop()
    elif change == "null_verdict":
        review["rows"][0]["verdict"] = None
    elif change == "bad_hash":
        review["rows"][0]["record_sha256"] = "changed"
    elif change == "missing_notes":
        review["rows"][0]["notes"] = ""
    elif change == "bad_category":
        review["rows"][0]["category"] = "evidence_not_used"
    else:
        correction(saved_queue, review)
        review["rows"][0].update(verdict="pass", category="none")
    with pytest.raises(ValueError):
        check_review(saved_queue, review)


@pytest.mark.parametrize("change", ["quote", "outside", "bool_offset", "too_long", "doc_id",
                                     "no_attestation", "not_literal", "mixed_code", "junk_json"])
def test_bad_provenance_and_unsafe_corrections_rejected_without_execution(saved_queue, change):
    review = completed_review(saved_queue)
    candidate = correction(saved_queue, review)
    span = candidate["supporting_spans"][0]
    if change == "quote":
        span["quote"] = "fabricated"
    elif change == "outside":
        span.update(start=0, end=30, quote=saved_queue["records"][0]["source"]["text"][:30])
    elif change == "bool_offset":
        span["start"] = True
    elif change == "too_long":
        span["end"] = span["start"] + 1201
    elif change == "doc_id":
        span["doc_id"] = "another_paper"
    elif change == "no_attestation":
        candidate["source_support_reviewed"] = False
    elif change == "not_literal":
        candidate["replacement_final_action"] = (
            'submit_gap(missing=open("/tmp/unsafe"), follow_up_query="q")')
    elif change == "mixed_code":
        candidate["replacement_final_action"] = 'import os; os.system("touch /tmp/unsafe")'
    else:
        candidate["replacement_final_action"] += '; print("trailing code")'
    with pytest.raises(ValueError):
        check_review(saved_queue, review)


def test_literal_gap_is_a_draft_not_verified_absence(saved_queue):
    review = completed_review(saved_queue)
    correction(saved_queue, review, gap=True)
    report = check_review(saved_queue, review)
    assert report["correction_drafts"][0]["kind"] == "gap"
    assert report["training_ready"] == 0


def test_bundle_tampering_rejected_before_output(tmp_path):
    bundle = tmp_path / "run"
    bundle.mkdir()
    write(bundle / "validation.json", {"artifact_sha256": {
        k: "wrong" for k in ["frozen/plan.json", "frozen/cases.json", "frozen/first-inputs.json",
                             "session.requests.jsonl", "live/summary.json", "runtime.json"]}})
    (bundle / "frozen").mkdir()
    write(bundle / "frozen/plan.json", {})
    output = tmp_path / "queue"
    with pytest.raises(ValueError, match="saved artifact changed"):
        prepare(bundle, output)
    assert not output.exists()


def test_bundle_path_cannot_escape_source_directory(tmp_path):
    write(tmp_path / "validation.json", {"artifact_sha256": {
        k: "wrong" for k in ["../secret", "frozen/plan.json", "frozen/cases.json",
                             "frozen/first-inputs.json", "session.requests.jsonl",
                             "live/summary.json", "runtime.json"]}})
    with pytest.raises(ValueError, match="escapes"):
        build_queue(tmp_path)


def test_queue_rebuild_prevents_training_flag_or_record_tampering(tmp_path):
    queue_dir = tmp_path / "queue"
    prepare(BUNDLE, queue_dir)
    queue = json.loads((queue_dir / "queue.json").read_text())
    queue["records"][0]["data_policy"]["training_eligible"] = True
    write(queue_dir / "queue.json", queue)
    review_path = tmp_path / "review.json"
    write(review_path, completed_review(queue))
    with pytest.raises(ValueError, match="frozen original"):
        check(BUNDLE, queue_dir, review_path, tmp_path / "checked")


def test_new_output_only_and_actual_cli_workflow(tmp_path):
    queue_dir = tmp_path / "queue"
    script = Path(__file__).resolve().parents[1] / "scripts/reader_improvement.py"
    proc = subprocess.run([sys.executable, str(script), "prepare", "--run", str(BUNDLE),
                           "--output", str(queue_dir)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["records"] == 3
    with pytest.raises(FileExistsError):
        prepare(BUNDLE, queue_dir)
    queue = json.loads((queue_dir / "queue.json").read_text())
    review = completed_review(queue)
    correction(queue, review)
    path = tmp_path / "review.json"
    write(path, review)
    proc = subprocess.run([sys.executable, str(script), "check", "--run", str(BUNDLE),
                           "--queue", str(queue_dir), "--review", str(path),
                           "--output", str(tmp_path / "checked")], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["training_ready"] == 0


def test_blank_template_cannot_be_accepted(saved_queue):
    with pytest.raises(ValueError, match="attribution"):
        check_review(saved_queue, review_template(saved_queue))
    untouched = copy.deepcopy(saved_queue)
    assert digest(saved_queue) == digest(untouched)
