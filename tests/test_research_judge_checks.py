"""Fixture contracts are mechanically checked; semantic correctness is reviewed separately."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from src.eval.research_judge_checks import check_bundle, judge_inputs, sha256

ROOT = Path(__file__).resolve().parents[1] / "benchmarks/research_judge_checks_v1"


def read(root, name):
    return json.loads((root / name).read_text())


def write(root, name, value):
    (root / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def rebind(root):
    """For structural mutation tests only; this does not perform a new source review."""
    files = read(root, "freeze-lock.json")["files"]
    files = {n: sha256(root / n) for n in files if n != "reviews/source-review.json"}
    review = read(root, "reviews/source-review.json")
    review["reviewed_file_sha256"] = files.copy()
    write(root, "reviews/source-review.json", review)
    files["reviews/source-review.json"] = sha256(root / "reviews/source-review.json")
    write(root, "freeze-lock.json", {"files": files})


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "pack"
    shutil.copytree(ROOT, root)
    return root


def test_real_pack_has_predeclared_counts_and_no_judge_result():
    result = check_bundle(ROOT)
    assert result["cases"] == 32 and sum(result["groups"].values()) == 32
    assert result["judge_results"] is None and result["human_review"] == "not_performed"
    labels = read(ROOT, "labels.json")
    assert sorted(label["expected_score"] for label in labels if label["group"] == "partial") == [
        2 / 5, 2 / 4, 2 / 3, 5 / 6]


def test_judge_export_excludes_verdicts_categories_and_includes_own_evidence():
    rows = judge_inputs(ROOT)
    assert len(rows) == 32
    assert all(set(r) == {"case_id", "paper_version", "question", "requested_facts",
                         "reference_contexts", "source_text", "candidate",
                         "inspection_precondition"} for r in rows)
    assert all(r["candidate"]["evidence"] and "synthetic_fixture" in r[
        "inspection_precondition"] for r in rows)
    assert not {"expected_score", "group", "critical_type", "correct_and_supported",
                "all_material_assertions_supported", "rationale"} & set(
                    key for row in rows for key in row)


def test_nested_unknown_fields_never_export_as_labels(bundle):
    cases, qs = read(bundle, "cases.json"), read(bundle, "questions.json")
    for row in cases:
        row["candidate"]["expected_score"] = "LEAK_MARKER"
        row["candidate"]["evidence"][0]["verdict"] = "LEAK_MARKER"
    for q in qs.values():
        q["requested_facts"][0]["verdict"] = "LEAK_MARKER"
        q["reference_contexts"][0]["verdict"] = "LEAK_MARKER"
    write(bundle, "cases.json", cases)
    write(bundle, "questions.json", qs)
    rebind(bundle)
    assert "LEAK_MARKER" not in json.dumps(judge_inputs(bundle))


def test_byte_tamper_rejected_before_export(bundle):
    with (bundle / "cases.json").open("a") as output:
        output.write(" ")
    with pytest.raises(ValueError, match="frozen fixture changed"):
        judge_inputs(bundle)


def test_valid_characters_are_not_semantic_support_or_fabrication():
    cases = {r["case_id"]: r for r in read(ROOT, "cases.json")}
    labels = read(ROOT, "labels.json")
    clipped = next(label for label in labels if label["subtype"] == "true_values_clipped_evidence")
    row = cases[clipped["case_id"]]
    assert "310 ms" in row["candidate"]["answer"]
    assert "310 ms" not in row["candidate"]["evidence"][0]["text"]
    assert clipped["expected_score"] == 0
    assert sum(label["subtype"] == "fabricated_target_value" for label in labels) == 2


@pytest.mark.parametrize("mutation,error", [
    ("offset", "source span"), ("score", "authored score"),
    ("count", "32 aligned"), ("category", "two of each"),
    ("gap_gate", "authored score"), ("human", "human review must"),
    ("judge_result", "cannot assert judge"), ("reserved_source", "only two development"),
    ("review_binding", "complete independent"),
])
def test_structural_faults_rejected_after_rebinding(bundle, mutation, error):
    rows, labels = read(bundle, "cases.json"), read(bundle, "labels.json")
    if mutation == "offset":
        rows[0]["candidate"]["evidence"][0]["start"] += 1
    elif mutation == "score":
        next(label for label in labels if label["group"] == "partial")["expected_score"] = 1
    elif mutation == "count":
        rows.pop()
    elif mutation == "category":
        label = next(label for label in labels if label["group"] == "critical_negative")
        label["critical_type"] = "other"
    elif mutation == "gap_gate":
        label = next(label for label in labels if label["critical_type"] ==
                     "valid target gap plus unsupported partial literature claim")
        label["expected_score"] = 1
    elif mutation == "human":
        obj = read(bundle, "reviews/human-template.json")
        obj["rows"][0]["verdict"] = "model_assistant_filled"
        write(bundle, "reviews/human-template.json", obj)
    elif mutation in {"judge_result", "reserved_source"}:
        obj = read(bundle, "manifest.json")
        if mutation == "judge_result":
            obj["actual_results"] = {"passed": True}
        else:
            obj["sources"]["2402.02750v1"] = {}
        write(bundle, "manifest.json", obj)
    write(bundle, "cases.json", rows)
    write(bundle, "labels.json", labels)
    rebind(bundle)
    if mutation == "review_binding":
        review = read(bundle, "reviews/source-review.json")
        review["reviewed_file_sha256"]["labels.json"] = "0" * 64
        write(bundle, "reviews/source-review.json", review)
        lock = read(bundle, "freeze-lock.json")
        lock["files"]["reviews/source-review.json"] = sha256(bundle / "reviews/source-review.json")
        write(bundle, "freeze-lock.json", lock)
    with pytest.raises(ValueError, match=error):
        check_bundle(bundle)


def test_linked_source_rejected(bundle, tmp_path):
    source = bundle / "sources/2307.08691v1.json"
    target = tmp_path / "source.json"
    source.replace(target)
    source.symlink_to(target)
    with pytest.raises(ValueError, match="linked"):
        check_bundle(bundle)


def test_cli_is_offline_and_export_never_overwrites(tmp_path):
    script = ROOT.parents[1] / "scripts/check_research_judge_cases.py"
    target = tmp_path / "inputs.json"
    cmd = [sys.executable, str(script), "--export-judge-inputs", str(target)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    before = target.read_bytes()
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode != 0 and "FileExistsError" in result.stderr
    assert target.read_bytes() == before


@pytest.mark.parametrize("field", ["answer", "gap_report"])
def test_structured_label_in_allowed_value_is_rejected(bundle, field):
    rows = read(bundle, "cases.json")
    row = next(r for r in rows if r["candidate"]["answer"] != "Unanswerable")
    row["candidate"][field] = ({"text": "answer", "expected_score": 0} if field == "answer"
        else {"missing": {"expected_score": 0}, "follow_up_query": "query"})
    write(bundle, "cases.json", rows)
    rebind(bundle)
    with pytest.raises(ValueError, match="bounded"):
        judge_inputs(bundle)
