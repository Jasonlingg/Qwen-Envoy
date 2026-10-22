"""Offline source-bound triage; correction drafts are never training-ready.

Only saved passage-smoke bundles are supported. No inference or Python execution
occurs here. Source review and Docker replay are separate gates.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.product.packet_verifier import MAX_SPAN_CHARS, verify_investigation_packet
from src.product.passage_scope import contains
from src.product.reader_gaps import parse_gap_submission

SCHEMA = "reader-improvement-queue-v1"
CATEGORIES = {
    "none", "evidence_not_used", "incomplete_answer", "unsupported_claim",
    "gap_reporting", "retrieval", "tool_execution", "prompt_or_protocol", "other",
}


def digest(value):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def inside(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("bundle path escapes its source directory")
    return path


def build_queue(bundle: Path):
    """Rebuild from hash-bound originals, including controls, without semantic grading."""
    validation = load(bundle / "validation.json")
    hashes = validation["artifact_sha256"]
    required = {"frozen/plan.json", "frozen/cases.json", "frozen/first-inputs.json",
                "session.requests.jsonl", "live/summary.json", "runtime.json"}
    if not required <= hashes.keys():
        raise ValueError("bundle lacks bound plan, input, ledger, summary or runtime")
    for name, expected in hashes.items():
        if file_hash(inside(bundle, name)) != expected:
            raise ValueError(f"saved artifact changed: {name}")
    plan = load(bundle / "frozen/plan.json")
    cases = load(bundle / "frozen/cases.json")
    first = load(bundle / "frozen/first-inputs.json")
    summary = load(bundle / "live/summary.json")
    cells = {c["case_id"]: c for c in summary["cases"]}
    ids = [c["case_id"] for c in cases]
    if (not ids or len(set(ids)) != len(ids) or set(ids) != set(first)
            or set(ids) != set(cells) or len(cells) != len(summary["cases"])
            or summary.get("unattempted_cases")):
        raise ValueError("complete unique case coverage required")
    ledger = [json.loads(line) for line in
              (bundle / "session.requests.jsonl").read_text().splitlines() if line.strip()]
    reserved = [r for r in ledger if r.get("event") == "reserved"]
    if len({r["request"] for r in reserved}) != len(reserved):
        raise ValueError("duplicate model request in ledger")
    records, offset = [], 0
    for case in cases:
        case_id = case["case_id"]
        # IDs are labels, never trusted filesystem paths.
        if Path(case_id).name != case_id or case_id in {".", ".."}:
            raise ValueError("invalid case ID")
        packet_name = f"live/{case_id}.packet.json"
        snapshot = "frozen/" + case["snapshot"]
        source_name = f"{snapshot}/corpus/{case['doc_id']}.json"
        manifest_name = f"{snapshot}/manifest.json"
        if not {packet_name, source_name, manifest_name} <= hashes.keys():
            raise ValueError("unbound source or packet")
        packet, source = load(inside(bundle, packet_name)), load(inside(bundle, source_name))
        initial = first[case_id]
        prompt = initial["messages"][0]
        if prompt["role"] != "system" or packet["question"] != case["question"]:
            raise ValueError("question or prompt identity mismatch")
        verified = verify_investigation_packet(
            packet, load(inside(bundle, manifest_name)), {case["doc_id"]: source},
            allowed_prompt_sha256s=frozenset({hashlib.sha256(prompt["content"].encode()).hexdigest()}),
            expected_reader_mode="assigned", expected_reader_guidance=plan["reader_guidance"],
            expected_passage_windows=case["windows"],
        )
        count = packet["model_requests_attempted"]
        requests = reserved[offset:offset + count]
        if not requests or requests[0]["payload"] != initial:
            raise ValueError("actual first request differs from frozen input")
        offset += count
        traces = packet["trajectory"]
        # This adapter is intentionally limited to one model request per action.
        if len(traces) != count:
            raise ValueError("format retries require a different trace adapter")
        for request, trace in zip(requests, traces):
            completed = [r for r in ledger if r.get("event") == "completed"
                         and r.get("request") == request["request"]]
            if (len(completed) != 1 or completed[0]["response"]["choices"][0]
                    ["message"]["content"] != trace["action"]):
                raise ValueError("raw response differs from recorded action")
        final = requests[-1]["payload"]
        if final["messages"][0] != prompt:
            raise ValueError("effective system prompt changed within the case")
        cell = cells[case_id]
        records.append({
            "case_id": case_id, "question": case["question"], "doc_id": case["doc_id"],
            "source": source,
            "source_text_sha256": hashlib.sha256(source["text"].encode()).hexdigest(),
            "original_packet": packet, "verified_original": verified,
            "actual_final_request": final,
            "mechanical_signals": {k: cell.get(k) for k in (
                "execution_error_steps", "action_rejection_steps", "submission_rejection_steps",
                "gap_rejection_steps", "mcp_verification")},
            "semantic_verdict": None,
            "data_policy": {"usage": "regression_only", "training_eligible": False},
        })
    if offset != len(reserved):
        raise ValueError("unassigned requests in ledger")
    return {
        "schema_version": SCHEMA, "source_bundle_sha256": file_hash(bundle / "validation.json"),
        "baseline": {"frozen_plan_sha256": hashes["frozen/plan.json"],
                     "plan": plan, "runtime": load(bundle / "runtime.json")},
        "records": records, "human_review": None,
        "instructions": "Mechanical checks establish provenance, not semantic support. Review all "
        "cases, including successes. Diagnose before proposing changes. Only the final action can "
        "be replaced; do not invent tool observations. All records in this evaluation adapter are "
        "regression-only, never training data. Corrected actions require source review and fresh "
        "Docker replay. No model call, training, reward or promotion is performed.",
    }


def review_template(queue):
    return {"schema_version": SCHEMA, "queue_sha256": digest(queue),
            "reviewer_kind": None, "reviewer_identity": None, "reviewed_at": None,
            "rows": [{"case_id": r["case_id"], "record_sha256": digest(r),
                      "verdict": None, "category": None, "notes": "",
                      "correction": None} for r in queue["records"]],
            "correction_format": {"replacement_final_action": "literal terminal action",
                                  "rationale": "why this addresses the failure",
                                  "source_support_reviewed": True,
                                  "supporting_spans": [{"doc_id": "...", "start": 0,
                                                        "end": 1, "quote": "..."}]}}


def prepare(bundle, output):
    queue = build_queue(bundle)
    if output.resolve().is_relative_to(bundle.resolve()):
        raise ValueError("output must be outside the frozen bundle")
    output.mkdir(parents=True, exist_ok=False)
    write(output / "queue.json", queue)
    write(output / "review-template.json", review_template(queue))
    return {"records": len(queue["records"]), "status": "awaiting_review",
            "training_eligible": 0, "model_calls": 0}


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _span(span, record):
    if (not isinstance(span, dict) or set(span) != {"doc_id", "start", "end", "quote"}
            or span["doc_id"] != record["doc_id"]
            or not contains(record["original_packet"]["inspected_passage_windows"],
                            span["start"], span["end"])
            or span["end"] - span["start"] > MAX_SPAN_CHARS
            or span["quote"] != record["source"]["text"][span["start"]:span["end"]]):
        raise ValueError("correction evidence must exactly match actually inspected source")


def check_correction(correction, record):
    if (not isinstance(correction, dict) or set(correction) != {
            "replacement_final_action", "rationale", "source_support_reviewed", "supporting_spans"}
            or not _text(correction["rationale"])
            or correction["source_support_reviewed"] is not True):
        raise ValueError("correction needs explicit source review and rationale")
    action, spans = correction["replacement_final_action"], correction["supporting_spans"]
    if (not _text(action) or len(action) > 8000 or not isinstance(spans, list)
            or not 1 <= len(spans) <= 5):
        raise ValueError("bounded terminal correction and supporting spans required")
    for span in spans:
        _span(span, record)
    packet = record["original_packet"]
    prefix = packet["trajectory"][:-1]
    inspections = {t["action"] for t in prefix if t.get("source_inspection") is True}
    if len(prefix) < 2 or len(inspections) < 2:
        raise ValueError("final-action repair requires two preceding source inspections")
    if action.startswith("SUBMIT: "):
        # Strict draft syntax, avoiding legacy parser acceptance of trailing junk.
        answer, separator, tail = action[8:].partition(" CITATIONS: ")
        citations, separator2, evidence = tail.partition(" EVIDENCE: ")
        if not separator or not separator2 or not answer.strip() or "\n" in action:
            raise ValueError("use one exact SUBMIT line with citations and evidence")
        try:
            citations, evidence = json.loads(citations), json.loads(evidence)
        except json.JSONDecodeError as exc:
            raise ValueError("terminal evidence requires JSON without trailing code") from exc
        expected = [{k: s[k] for k in ("doc_id", "start", "end")} for s in spans]
        if citations != [record["doc_id"]] or evidence != expected:
            raise ValueError("submitted spans must match reviewed supporting spans")
        kind = "answer"
    else:
        gap = parse_gap_submission(action)
        if gap is None or packet["reader_guidance"] != "passages-v2":
            raise ValueError("correction must be a SUBMIT line or a literal submit_gap action")
        kind = "gap"
    if action == packet["trajectory"][-1]["action"]:
        raise ValueError("correction must change the original final action")
    return {"kind": kind, "proposed_final_action": action,
            "original_prefix_sha256": digest(prefix),
            "actual_final_request_sha256": digest(record["actual_final_request"]),
            "rationale": correction["rationale"], "supporting_spans": spans,
            "source_support": "reviewer_attestation_not_mechanically_proven",
            "docker_replay": "required_not_performed", "training_ready": False,
            "training_eligible": False, "usage": "regression_only"}


def check_review(queue, review):
    if review.get("schema_version") != SCHEMA or review.get("queue_sha256") != digest(queue):
        raise ValueError("review is bound to a different queue")
    if (review.get("reviewer_kind") not in {"human", "model_assisted"}
            or not _text(review.get("reviewer_identity")) or not _text(review.get("reviewed_at"))):
        raise ValueError("explicit review attribution required; human review cannot be inferred")
    rows = review.get("rows", [])
    records = {r["case_id"]: r for r in queue["records"]}
    if (not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows)
            or len(rows) != len(records) or {r.get("case_id") for r in rows} != set(records)):
        raise ValueError("review must cover each case exactly once, including controls")
    candidates, counts = [], {"pass": 0, "partial": 0, "fail": 0}
    for row in rows:
        record = records[row["case_id"]]
        verdict, category = row.get("verdict"), row.get("category")
        if (row.get("record_sha256") != digest(record) or verdict not in counts
                or category not in CATEGORIES or not _text(row.get("notes"))
                or (verdict == "pass") != (category == "none")):
            raise ValueError("complete bound verdict, consistent category and notes required")
        correction = row.get("correction")
        if verdict == "pass" and correction is not None:
            raise ValueError("passing control should not receive a failure correction")
        if correction is not None:
            candidates.append({"case_id": row["case_id"], "record_sha256": digest(record),
                               **check_correction(correction, record)})
        counts[verdict] += 1
    return {"schema_version": SCHEMA, "queue_sha256": digest(queue),
            "review_sha256": digest(review), "reviewer_kind": review["reviewer_kind"],
            "reviewer_identity": review["reviewer_identity"], "reviewed_at": review["reviewed_at"],
            "human_review": None if review["reviewer_kind"] != "human" else "self_reported",
            "independent_human_review": None, "reviewed_counts": counts,
            "correction_drafts": candidates, "training_ready": 0, "model_calls": 0,
            "status": "reviewed_drafts_pending_docker_replay",
            "limitation": "No semantic correctness, model improvement or promotion is proven. "
            "Baseline evaluation papers/questions are excluded from training. "
            "This is not an SFT export."}


def check(bundle, queue_dir, review_path, output):
    queue = load(queue_dir / "queue.json")
    if queue != build_queue(bundle):
        raise ValueError("queue differs from its frozen original run")
    report = check_review(queue, load(review_path))
    if output.resolve().is_relative_to(bundle.resolve()):
        raise ValueError("output must be outside the frozen bundle")
    output.mkdir(parents=True, exist_ok=False)
    write(output / "correction-drafts.json", report)
    return {"records": len(queue["records"]), "correction_drafts": len(report["correction_drafts"]),
            "training_ready": 0, "model_calls": 0, "status": report["status"]}
