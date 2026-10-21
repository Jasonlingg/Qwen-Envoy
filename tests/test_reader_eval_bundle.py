"""Integrity is separate from semantic support and training authorization."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from src.eval.reader_eval_bundle import check_bundle, model_cases, sha256

ROOT = Path(__file__).resolve().parents[1] / "benchmarks/research_reader_v1"


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def relock(root):
    write(root / "freeze-lock.json", {"files": {
        str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*.json"))
        if p.name != "freeze-lock.json"}})


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "bundle"
    shutil.copytree(ROOT, root)
    return root


def test_real_pack_checked_offline_and_only_model_fields_exported():
    result = check_bundle(ROOT)
    assert result["split_counts"] == {"development": 12, "evaluation": 12}
    assert result["source_count"] == 8 and result["human_review"] == "not_performed"
    for split in ("development", "evaluation"):
        cases = model_cases(ROOT, split)
        assert len(cases) == 12
        assert all(set(case) == {"question", "paper_version"} for case in cases)
        assert "criterion" not in json.dumps(cases) and "target_gap" not in json.dumps(cases)


def test_changed_reference_rejected_even_when_parseable(bundle):
    p = bundle / "references/evaluation.json"
    rows = json.loads(p.read_text())
    rows[0]["facts"][0]["criterion"] = "A different grading target"
    write(p, rows)
    with pytest.raises(ValueError, match="frozen bundle changed"):
        check_bundle(bundle)


def test_relocked_inexact_span_is_not_validated_as_provenance(bundle):
    p = bundle / "references/evaluation.json"
    rows = json.loads(p.read_text())
    rows[0]["source_contexts"][0]["text"] = "invented source text"
    write(p, rows)
    relock(bundle)
    with pytest.raises(ValueError, match="exact bounded source span"):
        check_bundle(bundle)


def test_relocking_new_criterion_cannot_inherit_old_source_review(bundle):
    p = bundle / "references/evaluation.json"
    rows = json.loads(p.read_text())
    rows[0]["facts"][0]["criterion"] = "A new unreviewed criterion"
    write(p, rows)
    relock(bundle)
    with pytest.raises(ValueError, match="source review differs"):
        check_bundle(bundle)


def test_paper_family_leakage_rejected_after_relocking(bundle):
    p = bundle / "cases/evaluation.json"
    cases = json.loads(p.read_text())
    dev = json.loads((bundle / "cases/development.json").read_text())
    for case in cases[:3]:
        case["paper_version"] = dev[0]["paper_version"]
    write(p, cases)
    p = bundle / "references/evaluation.json"
    refs = json.loads(p.read_text())
    dev_ref = json.loads((bundle / "references/development.json").read_text())[0]
    for ref in refs[:3]:
        ref.update({key: dev_ref[key] for key in ("facts", "source_contexts")})
    write(p, refs)
    relock(bundle)
    with pytest.raises(ValueError, match="split disjoint"):
        check_bundle(bundle)


def test_linked_source_cannot_escape_bundle(bundle, tmp_path):
    source = next((bundle / "sources").glob("*.json"))
    external = tmp_path / "external.json"
    source.replace(external)
    source.symlink_to(external)
    with pytest.raises(ValueError, match="linked"):
        check_bundle(bundle)


def test_freeze_lock_cannot_read_parent_file(bundle):
    write(bundle / "freeze-lock.json", {"files": {"../secret.json": "untrusted"}})
    with pytest.raises(ValueError, match="inside"):
        check_bundle(bundle)


def test_provisional_model_review_cannot_be_labeled_human(bundle):
    p = bundle / "reviews/source-review.json"
    review = json.loads(p.read_text())
    review["review_kind"] = "human"
    write(p, review)
    relock(bundle)
    with pytest.raises(ValueError, match="provisional source review"):
        check_bundle(bundle)


def test_human_answer_template_cannot_be_filled_by_preparation(bundle):
    p = bundle / "reviews/human-answer-template.json"
    template = json.loads(p.read_text())
    template["rows"][0]["answer_verdict"] = "pass"
    write(p, template)
    relock(bundle)
    with pytest.raises(ValueError, match="must stay blank"):
        check_bundle(bundle)


def test_no_implicit_training_split():
    with pytest.raises(ValueError, match="explicit"):
        model_cases(ROOT, "train")


def test_read_only_cli_has_no_execution_or_training_command():
    result = subprocess.run([sys.executable, "scripts/check_research_reader_eval.py"],
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["status"] == "mechanical_checks_passed_not_semantic_grades"
