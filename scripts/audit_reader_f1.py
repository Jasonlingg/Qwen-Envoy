#!/usr/bin/env python3
"""Offline metric comparison; never change rewards, call models or train."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.envoybench.qasper_official import (  # noqa: E402
    EVALUATOR_SHA256,
    official_evaluator,
)
from scripts.prepare_reader_semantic_reward import build  # noqa: E402
from src.eval.qasper_reward import evidence_scores  # noqa: E402
from src.eval.reader_improvement import digest, file_hash  # noqa: E402
from src.eval.reader_rl_audit import submit  # noqa: E402
from src.eval.reader_rl_validation import assess, covered  # noqa: E402
from src.training.reader_data import load_bundle  # noqa: E402
from src.training.reader_semantic_reward import prepare  # noqa: E402


def paragraphs(text):
    """Reverse this corpus converter's double-newline paragraph concatenation."""
    result, start = [], 0
    for part in text.split("\n\n"):
        if part:
            result.append({"start": start, "end": start + len(part), "text": part})
        start += len(part) + 2
    return result


def metrics(case, context, evaluator):
    body = case["input"]
    text = context["docs"][context["doc_id"]]["text"]
    predicted = body["candidate_evidence"]
    blocks = paragraphs(text)
    # Map against ALL source paragraphs, never only gold paragraphs. This mapping
    # is an audited candidate adaptation, not an official QASPER submission.
    mapped = list(dict.fromkeys(p["text"] for p in blocks if any(
        max(p["start"], s["start"]) < min(p["end"], s["end"]) for s in predicted)))
    gold = [a for a in context["annotations"]
            if not a["unanswerable"] and not a["requires_nontext_evidence"]]
    if not gold or any(s["text"] not in {p["text"] for p in blocks}
                       for a in gold for s in a["evidence"]):
        raise ValueError("original text annotation/paragraph mapping unavailable")
    doc_id = context["doc_id"]
    spans = [{"doc_id": doc_id, "start": s["start"], "end": s["end"]} for s in predicted]
    return {
        "upstream_answer_f1": max(evaluator.token_f1_score(
            body["candidate_answer"], a["answer_text"]) for a in gold),
        "literal_quotes_as_paragraphs_f1": max(evaluator.paragraph_f1_score(
            [s["quote"] for s in predicted], [s["text"] for s in a["evidence"]]) for a in gold),
        "overlap_mapped_paragraph_f1_not_official": max(evaluator.paragraph_f1_score(
            mapped, [s["text"] for s in a["evidence"]]) for a in gold),
        "character_coverage_f1_not_official": max(evidence_scores(
            spans, a["evidence"])[0] for a in gold),
        "reference_scope": [{"annotation_id": a["annotation_id"],
            "fully_available_paragraphs": sum(covered(s, context["windows"])
                                               for s in a["evidence"]),
            "total_paragraphs": len(a["evidence"])} for a in gold],
    }


def audit(bundle, smoke=None):
    pack = build(bundle, smoke)
    _, roles, contexts, cards = load_bundle(bundle)
    rows = pack["cases"]
    for c in roles["reward_validation"]:
        cid = c["case_id"]
        context, card = contexts[cid], cards[cid]
        for kind, spans in [
            ("all_assigned_windows", context["windows"]),
            ("unrelated_first_character", [{"start": 0, "end": 1}]),
        ]:
            record = {"context_id": cid, "action": submit(
                context["annotations"][0]["answer_text"], c["doc_id"],
                [{"doc_id": c["doc_id"], **s} for s in spans]),
                "inspected_windows": context["windows"]}
            rows.append({"case": prepare(record, context, context_id=cid,
                context_sha256=card["context_sha256"], role="reward_validation"),
                "role": "reward_validation", "kind": kind,
                "origin": "authored_fixture_assumed_inspection_not_a_model_rollout",
                "finite_reference_reward": assess(record, context, card)["reward"]})
    evaluator = official_evaluator()
    output, groups = [], defaultdict(list)
    for row in rows:
        case = row["case"]
        result = {k: row[k] for k in ("role", "kind", "origin", "finite_reference_reward")}
        result.update(case_sha256=case["case_sha256"], context_id=case["input"]["context_id"],
                      candidate_answer=case["input"]["candidate_answer"],
                      candidate_evidence=case["input"]["candidate_evidence"])
        result.update(metrics(case, contexts[result["context_id"]], evaluator))
        output.append(result)
        groups[row["kind"]].append(result)
    keys = ["finite_reference_reward", "upstream_answer_f1", "literal_quotes_as_paragraphs_f1",
            "overlap_mapped_paragraph_f1_not_official", "character_coverage_f1_not_official"]
    return {"version": "reader-f1-offline-audit-v1", "source_manifest_sha256":
            pack["source_manifest_sha256"], "source_revision": pack["source_revision"],
            "upstream_evaluator_sha256": EVALUATOR_SHA256,
            "audit_script_sha256": file_hash(Path(__file__)),
            "cases_sha256": digest([r["case"] for r in rows]), "rows": output,
            "by_kind": {kind: {"count": len(rs), **{k: sum(r[k] for r in rs) / len(rs)
                         for k in keys}} for kind, rs in sorted(groups.items())},
            "training_ready": False, "active_reward_changed": False,
            "independent_human_review": None, "new_model_calls": 0, "gpu_activations": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("data/rl/reader-facts-v1"))
    parser.add_argument("--saved-rollouts", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.bundle, args.saved_rollouts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as out:
        out.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["by_kind"], indent=2))
