"""Official scoring preserves all annotations and never rewards missing answers."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.envoybench.qasper_official import (
    DEFAULT_REFERENCES,
    EVALUATOR_SHA256,
    REFERENCE_VERSION,
    REFERENCES_SHA256,
    ROOT,
    VENDOR,
    file_sha256,
    load_gold,
    official_evaluator,
    score_predictions,
    score_run,
)


def _annotation(*, spans=None, free="", yes_no=None, unanswerable=False):
    return {
        "answer": {
            "extractive_spans": spans or [],
            "free_form_answer": free,
            "yes_no": yes_no,
            "unanswerable": unanswerable,
            "evidence": [],
        }
    }


def _bundle(answers: list[list[dict]]) -> dict:
    return {
        "schema_version": REFERENCE_VERSION,
        "selection": {"question_ids": [f"q{i}" for i in range(len(answers))]},
        "questions": [
            {
                "question_id": f"q{i}",
                "source_question_id": f"source{i}",
                "source_paper_id": "paper",
                "answers": items,
            }
            for i, items in enumerate(answers)
        ],
    }


def test_upstream_files_and_all_original_annotations_are_pinned() -> None:
    assert file_sha256(VENDOR / "evaluator.py") == EVALUATOR_SHA256
    assert file_sha256(DEFAULT_REFERENCES) == REFERENCES_SHA256
    bundle = json.loads(DEFAULT_REFERENCES.read_text())
    gold, references = load_gold(bundle)
    assert len(gold) == len(references) == 40
    assert sum(len(items) for items in gold.values()) == 86
    assert sum(len(items) > 1 for items in gold.values()) == 37
    assert bundle["source"]["license"] == "CC BY 4.0"


@pytest.mark.parametrize(
    ("prediction", "reference", "expected"),
    [
        ("The, CAT!", "cat", 1.0),
        ("cat cat dog", "cat dog dog", 2 / 3),
        ("", "", 0.0),
        ("", "Unanswerable", 0.0),
        ("Unanswerable.", "Unanswerable", 1.0),
        ("The paper does not provide that information.", "Unanswerable", 0.0),
    ],
)
def test_official_token_f1_semantics(prediction, reference, expected) -> None:
    assert official_evaluator().token_f1_score(prediction, reference) == pytest.approx(expected)


def test_multiple_original_answers_and_annotation_precedence() -> None:
    bundle = _bundle(
        [
            [_annotation(spans=["red"]), _annotation(free="blue")],
            [_annotation(yes_no=False)],
            [_annotation(yes_no=True)],
            [_annotation(unanswerable=True, spans=["ignored"], yes_no=True)],
            [_annotation(spans=["alpha", "beta"], free="ignored", yes_no=False)],
        ]
    )
    metrics, rows = score_predictions(
        bundle,
        {
            "q0": "blue",
            "q1": "No",
            "q2": "Yes",
            "q3": "Unanswerable",
            "q4": "alpha beta",
        },
    )
    assert metrics["answer_f1"] == 1.0
    assert rows[0]["matched_reference_index"] == 1
    assert rows[0]["matched_answer_type"] == "abstractive"
    assert rows[3]["matched_answer_type"] == "none"
    gold, _ = load_gold(bundle)
    assert gold["source4"][0]["answer"] == "alpha, beta"


def test_missing_and_empty_answers_stay_zero_and_overall_denominator_is_fixed() -> None:
    bundle = _bundle([[_annotation(unanswerable=True)]] * 3)
    metrics, rows = score_predictions(bundle, {"q0": "Unanswerable", "q1": ""})
    assert metrics["answer_f1"] == pytest.approx(1 / 3)
    assert metrics["missing_predictions"] == 1
    assert metrics["answer_f1_by_type"]["none"] == 0.5
    assert metrics["answer_f1_by_type_denominators"]["none"] == 2
    assert rows[1]["prediction_missing"] is False
    assert rows[1]["answer_f1"] == 0.0
    assert rows[2]["prediction_missing"] is True
    assert rows[2]["answer_f1"] == 0.0


def test_equal_f1_ties_keep_first_source_annotation_type() -> None:
    metrics, rows = score_predictions(
        _bundle(
            [
                [_annotation(spans=["blue"]), _annotation(free="blue")],
            ]
        ),
        {"q0": "blue"},
    )
    assert rows[0]["matched_answer_type"] == "extractive"
    assert metrics["answer_f1_by_type_denominators"]["abstractive"] == 0


def test_saved_scores_equal_unmodified_official_cli(tmp_path: Path) -> None:
    bundle = json.loads(DEFAULT_REFERENCES.read_text())
    source_gold = {}
    for row in bundle["questions"]:
        source_gold.setdefault(row["source_paper_id"], {"qas": []})["qas"].append(
            {
                "question_id": row["source_question_id"],
                "answers": row["answers"],
            }
        )
    gold_path = tmp_path / "gold.json"
    gold_path.write_text(json.dumps(source_gold))
    results = json.loads((ROOT / "release/envoybench-v0.1/run/results.json").read_text())
    report = score_run(ROOT / "release/envoybench-v0.1/run")
    for model in ("qwen_base", "qwen_v5"):
        predictions_path = tmp_path / f"{model}.jsonl"
        predictions_path.write_text(
            "".join(
                json.dumps(
                    {
                        "question_id": row["question_id"].removeprefix("qasper_test_"),
                        "predicted_answer": row["predicted_answer"],
                        "predicted_evidence": [],
                    }
                )
                + "\n"
                for row in results
                if row["model_key"] == model and row["status"] == "submitted"
            )
        )
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                str(VENDOR / "evaluator.py"),
                "--gold",
                str(gold_path),
                "--predictions",
                str(predictions_path),
            ],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )
        official = json.loads(completed.stdout)
        assert report["models"][model]["answer_f1"] == official["Answer F1"]
        assert report["models"][model]["missing_predictions"] == official["Missing predictions"]
    assert report["models"]["qwen_base"]["answer_f1"] == pytest.approx(0.1991537448471281)
    assert report["models"]["qwen_v5"]["answer_f1"] == pytest.approx(0.30687301402774547)
    assert "evidence_f1" not in report["models"]["qwen_base"]


def _copy_run(tmp_path: Path) -> Path:
    source = ROOT / "release/envoybench-v0.1/run"
    run = tmp_path / "run"
    run.mkdir()
    for name in ("manifest.json", "results.json"):
        (run / name).write_bytes((source / name).read_bytes())
    return run


def test_failed_unsubmitted_and_unrecorded_results_are_missing_even_with_answer_text(
    tmp_path: Path,
) -> None:
    run = _copy_run(tmp_path)
    path = run / "results.json"
    results = json.loads(path.read_text())
    for index, row in enumerate(results):
        row["status"] = ("error", "no_submission", "escalated")[index % 3]
        row["predicted_answer"] = "Unanswerable"
    path.write_text(json.dumps(results[:-1]))
    report = score_run(run)
    for model in report["models"].values():
        assert model["answer_f1"] == 0.0
        assert model["missing_predictions"] == model["question_count"] == 40
    assert len(report["rows"]) == 80
    assert all(row["prediction_missing"] for row in report["rows"])


def test_single_model_run_uses_same_question_denominator(tmp_path: Path) -> None:
    run = _copy_run(tmp_path)
    manifest_path, results_path = run / "manifest.json", run / "results.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["models"] = [manifest["models"][1]]
    manifest_path.write_text(json.dumps(manifest))
    results = json.loads(results_path.read_text())
    results_path.write_text(json.dumps([row for row in results if row["model_key"] == "qwen_v5"]))
    report = score_run(run)
    assert list(report["models"]) == ["qwen_v5"]
    assert report["models"]["qwen_v5"]["question_count"] == 40
    assert len(report["rows"]) == 40


def test_modified_original_annotation_is_rejected(tmp_path: Path) -> None:
    bundle = json.loads(DEFAULT_REFERENCES.read_text())
    bundle["questions"][0]["answers"][0]["answer"]["free_form_answer"] = "changed"
    references = tmp_path / "references.json"
    references.write_text(json.dumps(bundle))
    with pytest.raises(ValueError, match="references SHA256"):
        score_run(ROOT / "release/envoybench-v0.1/run", references=references)


@pytest.mark.parametrize("change", ["duplicate", "question", "binding"])
def test_run_identity_and_coverage_reject_mismatches(tmp_path: Path, change: str) -> None:
    run = _copy_run(tmp_path)
    path = run / "results.json"
    results = json.loads(path.read_text())
    if change == "duplicate":
        results.append(copy.deepcopy(results[0]))
    elif change == "question":
        results[0]["question_id"] = "not-in-source"
    else:
        results[0]["corpus_hash"] = "changed"
    path.write_text(json.dumps(results))
    with pytest.raises(ValueError):
        score_run(run)


def test_scorer_cli_works_without_site_packages_or_repository_cwd(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(ROOT / "benchmarks/envoybench/qasper_official.py"),
            "score",
            "--run-dir",
            str(ROOT / "release/envoybench-v0.1/run"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(completed.stdout)
    assert report["models"]["qwen_base"]["missing_predictions"] == 11
    assert report["prediction_policy"] == "literal_submitted_answer_v1"
