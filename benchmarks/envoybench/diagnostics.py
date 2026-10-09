"""Offline baselines and explicitly scoped, source-bound case diagnoses.

These annotations explain saved examples; they never replace blind grades or
infer semantic correctness from a failed end-to-end verdict.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.eval.artifacts import configuration_hash

DEFAULT_CASE_STUDY = Path(__file__).with_name("case_study_20260930.json")
DEFAULT_BEHAVIOR_REVIEW = Path(__file__).with_name("response_behavior_20260930.json")
COMPONENT_STATUSES = {
    "answer_correctness": {"correct", "partial", "incorrect", "not_applicable", "unknown"},
    "evidence_support": {"supported", "partial", "unsupported", "not_applicable", "unknown"},
    "answer_behavior": {"substantive", "abstention", "no_submission", "unknown"},
}


def known_source_id(question: dict, documents: dict[str, dict]) -> str:
    """Resolve the disclosed paper even when unanswerable references cite none."""
    match = re.search(r'\(doc_id: "([^"/\\]+)"\)', question.get("question", ""))
    if match and match.group(1) in documents:
        return match.group(1)
    direct = question.get("source_paper_id")
    if direct in documents:
        return direct
    required = question.get("required_doc_ids", [])
    if len(required) == 1 and required[0] in documents:
        return required[0]
    raise ValueError("question has no resolvable frozen source paper")


def refusal_baseline(benchmark: dict) -> dict:
    """A rubric-derived expectation, not an executed or model-judged run."""
    labels = [row["expected_answerability"] for row in benchmark["questions"]]
    if any(label not in {"sufficient", "insufficient"} for label in labels):
        raise ValueError("refusal baseline requires known answerability labels")
    answerable = labels.count("sufficient")
    unanswerable = labels.count("insufficient")
    return {
        "key": "always_refuse", "label": "Always refuse",
        "kind": "analytical", "status": "expected_not_executed",
        "expected_pass": unanswerable, "total": len(labels),
        "expected_useful_answers": 0,
        "expected_answerable_pass": 0, "answerable_count": answerable,
        "expected_correct_abstentions": unanswerable,
        "unanswerable_count": unanswerable,
        "expected_false_refusals": answerable,
        "source_reference_status": "frozen_labels_not_independently_validated_here",
        "reason": (
            "Under the frozen labels and rubric, always submitting an honest refusal "
            "passes unanswerable questions and fails every answerable question. "
            "This policy provides zero substantive answers. Its expected score exposes "
            "why the aggregate alone does not measure useful answering. No model was run."
        ),
    }


def unknown_diagnostics() -> dict:
    return {
        field: {"status": "unknown", "reason": "Not separately reviewed.",
                "assessment_level": "unreviewed"}
        for field in COMPONENT_STATUSES
    }


def response_behavior(
    benchmark: dict, results: list[dict], path: Path = DEFAULT_BEHAVIOR_REVIEW,
) -> tuple[dict, dict, dict]:
    """Summarize explicit response-only labels, without deriving correctness."""
    if not path.is_file():
        return {}, {}, {"status": "not_reviewed"}
    sheet = json.loads(path.read_text(encoding="utf-8"))
    if sheet.get("schema_version") != "envoybench-response-behavior-v1":
        raise ValueError("invalid response-behavior schema")
    expected = {
        "benchmark_hash": configuration_hash(benchmark),
        "corpus_hash": benchmark["corpus_hash"],
        "results_hash": configuration_hash({"results": results}),
    }
    if any(sheet.get(key) != value for key, value in expected.items()):
        return {}, {}, {"status": "not_applicable_to_this_run"}
    if sheet.get("reviewer_kind") != "model_assisted":
        raise ValueError("response behavior must disclose model assistance")
    by_pair = {(row["question_id"], row["model_key"]): row for row in results}
    question_labels = {q["id"]: q["expected_answerability"] for q in benchmark["questions"]}
    kinds = {"substantive", "abstention", "mixed", "no_submission", "uncertain"}
    annotations, summaries = {}, {}
    for row in sheet["rows"]:
        pair = (row["question_id"], row["model_key"])
        if pair not in by_pair or pair in annotations:
            raise ValueError("response behavior has a duplicate or unknown result")
        result = by_pair[pair]
        if row.get("result_hash") != configuration_hash(result):
            raise ValueError("response behavior does not match its saved result")
        behavior, reason = row.get("behavior"), row.get("reason")
        if behavior not in kinds or not isinstance(reason, str) or not reason.strip():
            raise ValueError("response behavior needs a valid label and explanation")
        if (result["status"] == "submitted") == (behavior == "no_submission"):
            raise ValueError("response behavior contradicts submission status")
        annotations[pair] = {
            "status": behavior, "reason": reason, "assessment_level": "model_assisted",
        }
        summary = summaries.setdefault(pair[1], {
            "behavior_review_kind": "model_assisted", "behavior_reviewed_count": 0,
            "substantive_response_count": 0, "abstention_count": 0,
            "mixed_response_count": 0, "no_submission_count": 0,
            "uncertain_behavior_count": 0, "false_refusal_count": 0,
        })
        field = {
            "substantive": "substantive_response_count", "abstention": "abstention_count",
            "mixed": "mixed_response_count", "no_submission": "no_submission_count",
            "uncertain": "uncertain_behavior_count",
        }[behavior]
        summary[field] += 1
        summary["behavior_reviewed_count"] += 1
        summary["false_refusal_count"] += int(
            behavior == "abstention" and question_labels[pair[0]] == "sufficient"
        )
    if set(annotations) != set(by_pair):
        raise ValueError("response behavior review must cover the complete run")
    return annotations, summaries, {
        "status": "model_assisted", "reviewed_answer_count": len(annotations),
        "artifact_hash": configuration_hash(sheet), "method": sheet["method"],
        "note": (
            "Fresh-context anonymous answer-only classification; no correctness or support "
            "judgment. False refusals are abstentions on questions labeled answerable in "
            "the unvalidated frozen references. Mixed answers are kept separate."
        ),
    }


def prepared_retrieval_status(benchmark: dict, manifest: dict, path: Path) -> dict:
    """Expose a prepared comparison only if its source run and packets match."""
    status = {
        "status": "not_run",
        "description": (
            "Deterministic within-paper retrieval followed by one answer; "
            "implemented as a comparator, with no measured result yet."
        ),
    }
    if not path.is_file():
        return status
    plan = json.loads(path.read_text(encoding="utf-8"))
    if (plan.get("schema_version") != "envoybench-retrieval-prepared-v1"
            or plan.get("benchmark_hash") != configuration_hash(benchmark)
            or plan.get("corpus_hash") != benchmark["corpus_hash"]
            or plan.get("source_manifest_hash") != configuration_hash(manifest)):
        return status
    packets_path = path.with_name("packets.json")
    if (plan.get("status") != "prepared_not_run" or plan.get("requests_executed") != 0
            or not packets_path.is_file()):
        return status
    packets = json.loads(packets_path.read_text(encoding="utf-8"))
    if configuration_hash(packets) != plan.get("packets_hash"):
        raise ValueError("prepared retrieval packets do not match plan")
    return {
        **status, "status": "prepared_not_run",
        "prepared_question_count": len(plan["question_ids"]),
        "planned_request_count": plan["requests_planned"], "requests_executed": 0,
        "plan_hash": configuration_hash(plan),
    }


def case_diagnostics(
    benchmark: dict, results: list[dict], documents: dict[str, dict],
    path: Path = DEFAULT_CASE_STUDY,
) -> tuple[dict[tuple[str, str], dict], list[dict], dict]:
    """Load matching case notes only; reject corrupt notes for a matching run.

    A different run gets no annotations. Exact result and source bindings stop
    old diagnoses from silently following updated answers or corpus bytes.
    """
    if not path.is_file():
        return {}, [], {"status": "not_reviewed", "reviewed_question_count": 0}
    sheet = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(sheet, dict) or sheet.get("schema_version") != "envoybench-case-study-v1":
        raise ValueError("invalid case-study schema")
    expected = {
        "benchmark_hash": configuration_hash(benchmark),
        "corpus_hash": benchmark["corpus_hash"],
        "results_hash": configuration_hash({"results": results}),
    }
    if any(sheet.get(key) != value for key, value in expected.items()):
        return {}, [], {"status": "not_applicable_to_this_run", "reviewed_question_count": 0}
    if sheet.get("reviewer_kind") != "model_assisted_unblinded":
        raise ValueError("curated case diagnoses must declare unblinded model assistance")
    by_pair = {(row["question_id"], row["model_key"]): row for row in results}
    question_ids = {row["id"] for row in benchmark["questions"]}
    annotations: dict[tuple[str, str], dict] = {}
    for row in sheet["rows"]:
        pair = (row["question_id"], row["model_key"])
        if pair in annotations or pair not in by_pair:
            raise ValueError("case diagnosis has a duplicate or unknown result")
        if row.get("result_hash") != configuration_hash(by_pair[pair]):
            raise ValueError("case diagnosis does not match its saved result")
        components = row["components"]
        if set(components) != set(COMPONENT_STATUSES):
            raise ValueError("case diagnosis needs all component fields")
        for name, values in COMPONENT_STATUSES.items():
            component = components[name]
            if (component.get("status") not in values
                    or not isinstance(component.get("reason"), str)
                    or not component["reason"].strip()):
                raise ValueError("case diagnosis has an invalid component judgment")
        checks = row.get("source_checks", [])
        for check in checks:
            document = documents.get(check.get("doc_id"))
            start, end = check.get("start"), check.get("end")
            if (document is None or type(start) is not int or type(end) is not int
                    or not 0 <= start < end <= len(document["text"])
                    or document["text"][start:end] != check.get("quote")):
                raise ValueError("case diagnosis source quote does not match frozen paper")
        annotations[pair] = {
            name: {**value, "assessment_level": sheet["reviewer_kind"]}
            for name, value in components.items()
        }
        annotations[pair]["source_checks"] = checks
    featured = sheet["featured_cases"]
    seen = set()
    for case in featured:
        qid = case.get("question_id")
        if (qid not in question_ids or qid in seen
                or not any(pair[0] == qid for pair in annotations)
                or not case.get("label") or not case.get("reason")):
            raise ValueError("invalid featured case")
        seen.add(qid)
    return annotations, featured, {
        "status": "curated_model_assisted_unblinded",
        "reviewed_question_count": len({pair[0] for pair in annotations}),
        "reviewed_answer_count": len(annotations),
        "total_question_count": len(question_ids),
        "artifact_hash": configuration_hash(sheet),
        "note": (
            "These are selected explanatory cases, not a representative accuracy sample. "
            "Component labels are unblinded AI-assisted diagnoses. Other answers remain "
            "unreviewed on these dimensions; original blind verdicts are unchanged."
        ),
    }
