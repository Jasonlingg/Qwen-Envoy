"""Diagnostics must expose baseline weaknesses without inventing semantic labels."""

from __future__ import annotations

import copy
import json
import re

import pytest

from benchmarks.envoybench.calibration import prepare_sample
from benchmarks.envoybench.diagnostics import (
    case_diagnostics,
    known_source_id,
    prepared_retrieval_status,
    refusal_baseline,
    response_behavior,
)
from benchmarks.envoybench.export import snapshot_html
from src.eval.artifacts import configuration_hash


def test_always_refuse_is_expected_not_measured_and_never_useful():
    benchmark = {"questions": [
        {"expected_answerability": value}
        for value in ["sufficient"] * 20 + ["insufficient"] * 20
    ]}
    result = refusal_baseline(benchmark)
    assert result["expected_pass"] == 20
    assert result["total"] == 40
    assert result["expected_useful_answers"] == 0
    assert result["expected_false_refusals"] == 20
    assert result["status"] == "expected_not_executed"


def test_unanswerable_question_still_resolves_disclosed_paper():
    question = {
        "question": 'Use the known paper "Paper" (doc_id: "qasper_1905_00472") to answer: Q?',
        "source_paper_id": "1905.00472", "required_doc_ids": [],
    }
    assert known_source_id(question, {"qasper_1905_00472": {}}) == "qasper_1905_00472"


def test_case_labels_are_bound_to_exact_results_and_source(tmp_path):
    benchmark = {"corpus_hash": "corpus", "questions": [{"id": "q1"}]}
    results = [{"question_id": "q1", "model_key": "base", "predicted_answer": "blue"}]
    documents = {"paper": {"text": "Blue is the answer."}}
    sheet = {
        "schema_version": "envoybench-case-study-v1",
        "benchmark_hash": configuration_hash(benchmark), "corpus_hash": "corpus",
        "results_hash": configuration_hash({"results": results}),
        "reviewer_kind": "model_assisted_unblinded",
        "featured_cases": [{"question_id": "q1", "label": "Citation error", "reason": "Check"}],
        "rows": [{
            "question_id": "q1", "model_key": "base",
            "result_hash": configuration_hash(results[0]),
            "components": {
                "answer_correctness": {"status": "correct", "reason": "Paper says Blue."},
                "evidence_support": {"status": "unsupported", "reason": "Wrong span."},
                "answer_behavior": {"status": "substantive", "reason": "Answers blue."},
            },
            "source_checks": [{"doc_id": "paper", "start": 0, "end": 4, "quote": "Blue"}],
        }],
    }
    path = tmp_path / "diagnostics.json"
    path.write_text(json.dumps(sheet))
    labels, cases, provenance = case_diagnostics(benchmark, results, documents, path)
    assert labels[("q1", "base")]["answer_correctness"]["status"] == "correct"
    assert labels[("q1", "base")]["evidence_support"]["status"] == "unsupported"
    assert provenance["reviewed_question_count"] == 1 and len(cases) == 1
    changed = copy.deepcopy(results)
    changed[0]["predicted_answer"] = "red"
    assert case_diagnostics(benchmark, changed, documents, path)[0] == {}
    sheet["rows"][0]["source_checks"][0]["quote"] = "fake"
    path.write_text(json.dumps(sheet))
    with pytest.raises(ValueError, match="source quote"):
        case_diagnostics(benchmark, results, documents, path)


def test_snapshot_preserves_payload_without_script_injection():
    payload = {"active": {"answer": '</script><script>alert("bad")</script> & Ω'}}
    html = snapshot_html(payload)
    encoded = re.search(
        r'<script id="envoybench-data" type="application/json">(.*?)</script>', html, re.S
    ).group(1)
    assert "<" not in encoded
    assert json.loads(encoded) == payload
    assert html.count('<script id="envoybench-data"') == 1


def test_calibration_is_repeatable_anonymous_and_unjudged():
    questions, rows = [], []
    for index in range(8):
        label = "sufficient" if index < 4 else "insufficient"
        question = {"id": f"q{index}", "source_paper_id": "paper",
                    "question": f"Question {index}", "expected_answerability": label}
        questions.append(question)
        for candidate in range(2):
            rows.append({
                "blind_id": f"R{index}-{candidate}", "question_id": question["id"],
                "question": question["question"], "expected_answerability": label,
                "reference_answer": "blue", "grader_notes": [], "status": "submitted",
                "answer": "blue", "citations": ["paper"], "evidence": [],
                "verdict": "pass" if index % 2 else "fail", "notes": "SECRET PRIOR GRADE",
                "relevant_source_passage": "SECRET PRIOR PASSAGE", "model_key": "SECRET MODEL",
            })
    benchmark = {"corpus_hash": "corpus", "questions": questions}
    review = {"status": "complete", "reviewer_kind": "model_assisted",
              "benchmark_hash": configuration_hash(benchmark), "corpus_hash": "corpus",
              "results_hash": "results", "rows": rows}
    documents = {"paper": {"doc_id": "paper", "text": "Blue is the answer."}}
    packet = prepare_sample(review, benchmark, documents, question_count=4)
    assert packet == prepare_sample(review, benchmark, documents, question_count=4)
    assert len(packet["questions"]) == 4
    assert "SECRET" not in json.dumps(packet)
    assert packet["reviewer_kind"] is None and packet["status"] == "incomplete"
    assert all(row["verdict"] is None for q in packet["questions"] for row in q["answers"])
    assert {questions[int(q[1:])]["expected_answerability"]
            for q in packet["selected_question_ids"]} == {"sufficient", "insufficient"}


def test_behavior_counts_do_not_equate_substantive_with_correct(tmp_path):
    benchmark = {"corpus_hash": "corpus", "questions": [
        {"id": f"q{i}", "expected_answerability": "sufficient"} for i in range(4)
    ]}
    results = [
        {"question_id": f"q{i}", "model_key": "base",
         "status": "error" if i == 3 else "submitted"}
        for i in range(4)
    ]
    sheet = {
        "schema_version": "envoybench-response-behavior-v1",
        "benchmark_hash": configuration_hash(benchmark), "corpus_hash": "corpus",
        "results_hash": configuration_hash({"results": results}),
        "reviewer_kind": "model_assisted", "method": "anonymous answer-only review",
        "rows": [
            {"question_id": r["question_id"], "model_key": "base",
             "result_hash": configuration_hash(r), "behavior": label, "reason": "Read answer"}
            for r, label in zip(results, ["abstention", "substantive", "mixed", "no_submission"])
        ],
    }
    path = tmp_path / "behavior.json"
    path.write_text(json.dumps(sheet))
    annotations, summaries, provenance = response_behavior(benchmark, results, path)
    assert summaries["base"]["false_refusal_count"] == 1
    assert summaries["base"]["substantive_response_count"] == 1
    assert summaries["base"]["no_submission_count"] == 1
    assert summaries["base"]["mixed_response_count"] == 1
    assert "pass" not in summaries["base"]
    assert annotations[("q1", "base")]["status"] == "substantive"
    assert provenance["status"] == "model_assisted"
    assert response_behavior(benchmark, [*results, {"new": True}], path)[0] == {}
    sheet["rows"][-1]["behavior"] = "abstention"
    path.write_text(json.dumps(sheet))
    with pytest.raises(ValueError, match="submission status"):
        response_behavior(benchmark, results, path)
    sheet["rows"].pop()
    path.write_text(json.dumps(sheet))
    with pytest.raises(ValueError, match="complete run"):
        response_behavior(benchmark, results, path)


def test_prepared_baseline_status_requires_matching_run_and_packets(tmp_path):
    benchmark = {"corpus_hash": "corpus", "questions": [{"id": "q1"}]}
    manifest = {"run_id": "actual"}
    packets = {"packets": [{"question_id": "q1", "messages": []}]}
    plan = {
        "schema_version": "envoybench-retrieval-prepared-v1", "status": "prepared_not_run",
        "benchmark_hash": configuration_hash(benchmark), "corpus_hash": "corpus",
        "source_manifest_hash": configuration_hash(manifest), "question_ids": ["q1"],
        "packets_hash": configuration_hash(packets), "requests_planned": 2,
        "requests_executed": 0,
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    packets_path = tmp_path / "packets.json"
    packets_path.write_text(json.dumps(packets))
    status = prepared_retrieval_status(benchmark, manifest, path)
    assert status["status"] == "prepared_not_run" and status["requests_executed"] == 0
    assert status["prepared_question_count"] == 1 and status["planned_request_count"] == 2
    assert prepared_retrieval_status(benchmark, {"run_id": "another"}, path)["status"] == "not_run"
    packets_path.write_text(json.dumps({"packets": []}))
    with pytest.raises(ValueError, match="packets do not match"):
        prepared_retrieval_status(benchmark, manifest, path)
