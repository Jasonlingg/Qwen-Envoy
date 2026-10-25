#!/usr/bin/env python3
"""Prepare or score an offline QASPER-reference judge audit; never call an API."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.passage_pilot_replay import PROMPT_SHA  # noqa: E402
from src.eval.reader_improvement import digest, file_hash, load, write  # noqa: E402
from src.eval.reader_rl_audit import submit  # noqa: E402
from src.eval.reader_rl_validation import assess  # noqa: E402
from src.product.packet_verifier import verify_investigation_packet  # noqa: E402
from src.product.passage_scope import PASSAGES_V2  # noqa: E402
from src.research.agent import load_snapshot  # noqa: E402
from src.training.reader_data import load_bundle  # noqa: E402
from src.training.reader_semantic_reward import (  # noqa: E402
    AXES,
    PROMPT_SHA256,
    VERSION,
    calibration,
    prepare,
    request,
    score,
)

PARAPHRASES = [
    "The UM Inventory dataset.",
    "A parallel corpus containing 1,500 sentences.",
    "CCR (correct classification rate).",
    "1,873 Twitter threads, comprising about 14,000 tweets.",
    "Garg et al. (2012)'s Bayesian monolingual model.",
    "A bidirectional LSTM with a CRF (Bi-LSTM-CRF).",
    "They give Russian as an example.",
    "Using the Zero Resource Toolkit, ZRTools.",
]
WRONG = ["MNIST", "15,000 sentences", "BLEU", "187 threads and 1,400 tweets",
         "A transformer decoder only", "A unidirectional LSTM without a CRF", "Japanese",
         "By manual annotation only"]


def build(bundle, smoke=None, *, include_records=False):
    manifest, roles, contexts, cards = load_bundle(bundle)
    rows = []

    def add(record, role, origin, kind):
        cid = record["context_id"]
        context, card = contexts[cid], cards[cid]
        case = prepare(record, context, context_id=cid,
                       context_sha256=card["context_sha256"], role=role)
        rows.append({"case": case, "role": role, "origin": origin, "kind": kind,
                     "finite_reference_reward": assess(record, context, card)["reward"]})
        if include_records:
            rows[-1]["record"] = record

    if len(roles["reward_validation"]) != len(PARAPHRASES):
        raise ValueError("fixed reward-validation inventory changed")
    for c, paraphrase, wrong in zip(roles["reward_validation"], PARAPHRASES, WRONG):
        cid = c["case_id"]
        card, context = cards[cid], contexts[cid]
        spans = [{"doc_id": c["doc_id"], "start": s["start"], "end": s["end"]}
                 for claim in card["claims"] for s in claim["support"]]
        answer = context["annotations"][0]["answer_text"]
        for kind, text, evidence in [
            ("reference", answer, spans), ("paraphrase", paraphrase, spans),
            ("wrong_answer", wrong, spans),
            ("real_but_insufficient_quote", answer,
             [{"doc_id": c["doc_id"], "start": spans[0]["start"],
               "end": spans[0]["start"] + 1}]),
        ]:
            add({"context_id": cid, "action": submit(text, c["doc_id"], evidence),
                 "inspected_windows": context["windows"]}, "reward_validation",
                "authored_fixture_assumed_inspection_not_a_model_rollout", kind)
    if smoke is not None:
        if not smoke.is_dir():
            raise ValueError("saved rollout directory does not exist")
        for path in sorted(smoke.glob("*/rollout.json")):
            episode = load(path)
            cid = episode["case_id"]
            c = next((c for c in roles["train"] if c["case_id"] == cid), None)
            if (c is None or episode.get("infrastructure_ok") is not True
                    or episode["question_id"] != c["question_id"]):
                raise ValueError("only healthy saved training-role episodes are allowed")
            packet = episode["packet"]
            manifest_source, docs = load_snapshot(bundle / c["snapshot"])
            verified = verify_investigation_packet(packet, manifest_source, docs,
                allowed_prompt_sha256s=frozenset({PROMPT_SHA}), expected_reader_mode="assigned",
                expected_reader_guidance=PASSAGES_V2,
                expected_passage_windows=c["scope"]["windows"])
            if verified != episode["verified"] or packet["question"] != c["question"]:
                raise ValueError("saved packet/source verification differs")
            add({"context_id": cid, "action": submit(verified["candidate_answer"],
                c["doc_id"], packet["evidence"]),
                "inspected_windows": packet["inspected_passage_windows"]},
                "train", "real_saved_docker_rollout", "exposed_diagnostic_only")
    rows.sort(key=lambda r: r["case"]["case_sha256"])
    return {"version": VERSION, "source_manifest_sha256": file_hash(bundle / "manifest.json"),
            "source_dataset": manifest["source_dataset"],
            "source_revision": manifest["source_revision"],
            "prompt_sha256": PROMPT_SHA256, "cases": rows,
            "training_ready": False, "human_review": None,
            "model_requests": 0, "paid_budget_usd": 0, "gpu_activations": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("data/rl/reader-facts-v1"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--saved-rollouts", type=Path)
    parser.add_argument("--score", action="store_true")
    args = parser.parse_args()
    if args.score:
        pack = load(args.output / "cases.json")
        if digest(pack) != load(args.output / "manifest.json")["cases_sha256"]:
            raise ValueError("calibration pack changed")
        if (digest(load(args.output / "requests.json"))
                != load(args.output / "manifest.json")["requests_sha256"]):
            raise ValueError("judge request templates changed")
        raw = load(args.output / "judgments.json")
        cases = {r["case"]["case_sha256"]: r["case"] for r in pack["cases"]}
        scored, errors, seen, duplicates = {}, [], set(), set()
        for row in raw:
            key = row.get("case_sha256") if isinstance(row, dict) else None
            try:
                if not isinstance(key, str) or key not in cases:
                    raise ValueError("unknown candidate")
                if key in seen:
                    duplicates.add(key)
                    raise ValueError("duplicate judgment")
                seen.add(key)
                scored[key] = score(cases[key], row)
            except (ValueError, TypeError) as exc:
                errors.append({"case_sha256": key, "error": type(exc).__name__})
        for key in duplicates:
            scored.pop(key, None)
        report = calibration(pack["cases"], list(scored.values()),
                             load(args.output / "human-review.json"))
        report["judgment_errors"] = errors
        report["live_judge_run_verified"] = False
        if errors:
            report["development_gate_passed"] = False
        with (args.output / "scored-comparison.json").open("x") as out:
            out.write(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
        return
    pack = build(args.bundle, args.saved_rollouts)
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "cases.json", pack)
    requests = [request(r["case"], "SELECT_JUDGE_BEFORE_LIVE_USE") for r in pack["cases"]]
    write(args.output / "manifest.json", {"version": VERSION, "cases_sha256": digest(pack),
                                          "requests_sha256": digest(requests),
                                          "training_ready": False})
    write(args.output / "requests.json", requests)
    human = {"kind": "independent_human", "reviewer": None, "reviewed_at": None,
             "rows": [{"case_sha256": r["case"]["case_sha256"], **dict.fromkeys(AXES),
                       "notes": None} for r in pack["cases"] if r["role"] == "reward_validation"]}
    write(args.output / "human-review.json", human)
    write(args.output / "judgments.json", [])
    write(args.output / "comparison.json", calibration(pack["cases"], [], human))
    print(json.dumps({"cases": len(pack["cases"]), "model_requests": 0,
                      "training_ready": False, "status": "prepared_not_judged"}))


if __name__ == "__main__":
    main()
