#!/usr/bin/env python3
"""Test the F1 reward candidate offline; never enable it or launch training."""

import argparse
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.envoybench.qasper_official import EVALUATOR_SHA256  # noqa: E402
from scripts.prepare_reader_semantic_reward import build  # noqa: E402
from src.eval.qasper_reward import parse_strict  # noqa: E402
from src.eval.reader_improvement import digest, file_hash  # noqa: E402
from src.eval.reader_rl_audit import submit  # noqa: E402
from src.eval.reader_rl_validation import assess  # noqa: E402
from src.training.reader_data import load_bundle  # noqa: E402
from src.training.reader_f1_reward import VERSION, assess_f1  # noqa: E402


def audit(bundle, smoke=None):
    pack = build(bundle, smoke, include_records=True)
    _, _, contexts, cards = load_bundle(bundle)
    rows = pack["cases"]
    for row in list(rows):
        if row["kind"] != "reference":
            continue
        cid = row["record"]["context_id"]
        context = contexts[cid]
        answer, citations, evidence = parse_strict(row["record"]["action"])
        for kind in ("uninspected", "extra_unrelated_quote", "all_assigned_windows",
                     "negated_reference", "decimal_normalization_collision"):
            if kind == "decimal_normalization_collision" and answer.strip() != "1500 sentences":
                continue
            record = copy.deepcopy(row["record"])
            text, spans = answer, evidence
            if kind == "uninspected":
                record["inspected_windows"] = []
            elif kind == "extra_unrelated_quote":
                spans = evidence + [{"doc_id": context["doc_id"], "start": 0, "end": 1}]
            elif kind == "all_assigned_windows":
                spans = [{"doc_id": context["doc_id"], **w} for w in context["windows"]]
            elif kind == "negated_reference":
                text = "Not " + answer
            else:
                text = "15.00 sentences"  # Numerically different, same upstream normalization.
            record["action"] = submit(text, citations[0], spans)
            rows.append({"record": record, "role": row["role"], "kind": kind,
                         "origin": row["origin"]})
    output, groups = [], defaultdict(list)
    for row in rows:
        record = row["record"]
        cid = record["context_id"]
        result = {k: row[k] for k in ("role", "kind", "origin")}
        result.update(record_sha256=digest(record), record=record,
                      finite_reference_reward=assess(record, contexts[cid], cards[cid])["reward"],
                      candidate=assess_f1(record, contexts[cid], cards[cid]))
        output.append(result)
        groups[row["kind"]].append(result)
    means = {kind: {"count": len(rs), "finite_mean":
             sum(r["finite_reference_reward"] for r in rs) / len(rs), "candidate_mean":
             sum(r["candidate"]["reward"] for r in rs) / len(rs)}
             for kind, rs in sorted(groups.items())}
    checks = {
        "reference_controls_full_credit": all(r["candidate"]["reward"] == 1
                                               for r in groups["reference"]),
        "proposed_paraphrases_nonzero": all(r["candidate"]["reward"] > 0
                                            for r in groups["paraphrase"]),
        "invalid_evidence_zero": all(r["candidate"]["reward"] == 0 for kind in (
            "real_but_insufficient_quote", "uninspected", "extra_unrelated_quote",
            "all_assigned_windows") for r in groups[kind]),
        "authored_wrong_answers_less_than_full_credit": all(r["candidate"]["reward"] < 1
            for kind in ("wrong_answer", "negated_reference", "decimal_normalization_collision")
            for r in groups[kind]),
    }
    return {"version": VERSION, "source_manifest_sha256": pack["source_manifest_sha256"],
            "source_revision": pack["source_revision"], "evaluator_sha256": EVALUATOR_SHA256,
            "implementation_sha256": {p: file_hash(Path(p)) for p in (
                __file__, "src/training/reader_f1_reward.py",
                "src/eval/reader_rl_validation.py", "scripts/prepare_reader_semantic_reward.py")},
            "rows": output, "by_kind": means, "checks": checks,
            "development_gate_passed": all(checks.values()), "training_ready": False,
            "active_reward_changed": False, "independent_human_review": None,
            "new_model_calls": 0, "gpu_activations": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("data/rl/reader-facts-v1"))
    parser.add_argument("--saved-rollouts", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.bundle, args.saved_rollouts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as out:
        out.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("by_kind", "checks", "development_gate_passed")},
                     indent=2))
