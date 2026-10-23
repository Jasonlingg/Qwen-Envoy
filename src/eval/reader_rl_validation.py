"""Offline, finite reviewed-answer reward experiment. Never a semantic judge.

No generation, Python execution, optimizer, or training launcher lives here.
Keys are private evaluator references, not content to inject into policy inputs.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import platform
import unicodedata
from collections import Counter
from pathlib import Path

from src.env.tools import TOOL_PREAMBLE
from src.env.tools_v2 import V1_TOOL_PREAMBLE_SHA256
from src.eval.passage_pilot_replay import BASELINE, PROMPT_SHA
from src.eval.qasper_reward import parse_strict
from src.eval.reader_improvement import digest, file_hash, inside, load, write
from src.eval.reader_rl_audit import ROOT, SCORERS, checked_bundle, trainer_observations
from src.product.packet_verifier import MAX_SPAN_CHARS

VERSION = "reviewed-answer-support-diagnostic-v1"
AUDIT = ROOT / "examples/reader-rl-investigation"


def canonical(answer):
    """Only normalize whitespace/NFC: even letter case can change scientific units."""
    return " ".join(unicodedata.normalize("NFC", answer).split())


def interval(span, limit):
    return (isinstance(span, dict) and set(span) == {"start", "end"}
            and type(span["start"]) is int and type(span["end"]) is int
            and 0 <= span["start"] < span["end"] <= limit)


def covered(span, windows):
    return any(w["start"] <= span["start"] < span["end"] <= w["end"] for w in windows)


def check_keys(keys, inputs):
    """Validate bindings/shape, not whether the reviewer's semantics are correct."""
    if (keys.get("schema_version") != VERSION or keys.get("usage") != "exposed_diagnostic_only"
            or keys.get("reviewer_kind") != "model_assisted" or not keys.get("reviewer_identity")
            or not keys.get("reviewed_at") or keys.get("human_review") is not None
            or keys.get("inputs_sha256") != digest(inputs)):
        raise ValueError("attributed diagnostic review and unchanged input binding required")
    contexts = inputs["contexts"]
    expected = {cid for cid, c in contexts.items() if c["scope_answerability"] == "present"}
    cards = keys.get("cards")
    if not isinstance(cards, list) or len(cards) != len(expected):
        raise ValueError("review must cover each answerable context exactly once")
    by_id = {}
    records = {r["id"]: r for r in inputs["records"]}
    for card in cards:
        cid = card["context_id"]
        if cid not in expected or cid in by_id:
            raise ValueError("duplicate or unexpected reviewed context")
        context = contexts[cid]
        if card.get("context_sha256") != digest(context):
            raise ValueError("review source/question/scope binding changed")
        text = context["docs"][context["doc_id"]]["text"]
        windows = context["windows"]
        if not windows or not all(interval(w, len(text)) for w in windows):
            raise ValueError("invalid source scope")
        control = records.get(card.get("source_control_id"), {})
        if (control.get("context_id") != cid or control.get("expected") != "acceptable"
                or card.get("original_control_review") not in {
                    "supported", "needs_support_revision"}
                or not card.get("review_notes")):
            raise ValueError("original source control and explicit claim review required")
        claims = card.get("claims", [])
        if not isinstance(claims, list) or not 1 <= len(claims) <= 12:
            raise ValueError("bounded claim-level review required")
        ids = set()
        for claim in claims:
            if (not isinstance(claim.get("id"), str) or not claim["id"]
                    or claim["id"] in ids or not claim.get("claim")):
                raise ValueError("unique named claims required")
            ids.add(claim["id"])
            spans = claim.get("support", [])
            if not isinstance(spans, list) or not 1 <= len(spans) <= 5:
                raise ValueError("each claim needs bounded reviewed support")
            for span in spans:
                bounds = {k: span.get(k) for k in ("start", "end")}
                if (set(span) != {"start", "end", "quote"} or not interval(bounds, len(text))
                        or not covered(bounds, windows)
                        or span["end"] - span["start"] > MAX_SPAN_CHARS
                        or span["quote"] != text[span["start"]:span["end"]]):
                    raise ValueError("reviewed support is not an exact bounded source quote")
        variants = card.get("accepted_answers", [])
        if not isinstance(variants, list) or not 1 <= len(variants) <= 16:
            raise ValueError("explicit finite complete-answer variants required")
        seen = set()
        for variant in variants:
            answer = variant.get("answer")
            if (not isinstance(answer, str) or not 0 < len(answer) <= 2000
                    or not canonical(answer) or canonical(answer) in seen
                    or set(variant.get("claim_ids", [])) != ids
                    or len(variant["claim_ids"]) != len(ids)):
                raise ValueError("variants must explicitly cover every reviewed claim")
            seen.add(canonical(answer))
        by_id[cid] = card
    return by_id


def assess(record, context, card):
    """Score against prevalidated finite keys, using evaluator-owned read receipts.

    Unknown answer wording gets zero, not a claim of falsehood. Missing-scope
    cases return None, not zero: exclude them from this objective's RL groups.
    This function is not an API for trusting model-supplied inspection windows.
    """
    if context["scope_answerability"] == "missing":
        return {"eligible": False, "reward": None, "reason": "gap_quality_not_scored"}
    if (card is None or card["context_sha256"] != digest(context)
            or record["context_id"] != card["context_id"]):
        raise ValueError("missing or stale reviewed answer/support key")

    def result(reason, reward=0.0):
        return {"eligible": True, "reward": reward, "reason": reason,
                "support_authority": "attributed_reference_review_not_automatic_entailment"}

    action = record.get("action")
    if not isinstance(action, str) or len(action) > 8000:
        return result("invalid_terminal")
    try:
        parsed = parse_strict(action)
    except (RecursionError, ValueError):
        parsed = None
    if parsed is None:
        return result("invalid_or_gap_terminal_on_answerable_task")
    answer, citations, evidence = parsed
    doc_id = context["doc_id"]
    text = context["docs"][doc_id]["text"]
    inspected = record.get("inspected_windows")
    if (not isinstance(inspected, list) or not 1 <= len(inspected) <= 4
            or any(not interval(w, len(text)) or not covered(w, context["windows"])
                   for w in inspected)):
        return result("invalid_inspection_receipt")
    if citations != [doc_id] or not 1 <= len(evidence) <= 5:
        return result("invalid_citations")
    clipped = []
    for span in evidence:
        bounds = {k: span.get(k) for k in ("start", "end")}
        if (set(span) != {"doc_id", "start", "end"} or span["doc_id"] != doc_id
                or not interval(bounds, len(text)) or not covered(bounds, inspected)):
            return result("evidence_not_inspected_in_scope")
        clipped.append({"start": span["start"],
                        "end": min(span["end"], span["start"] + MAX_SPAN_CHARS)})
    variant = next((v for v in card["accepted_answers"]
                    if canonical(answer) == canonical(v["answer"])), None)
    if variant is None:
        return result("answer_not_in_reviewed_variants")
    supports = [s for c in card["claims"] for s in c["support"]]
    if any(not covered(s, clipped) for s in supports):
        return result("reviewed_support_missing_after_quote_clipping")
    if any(not any(covered(s, [w]) for s in supports) for w in clipped):
        return result("additional_citation_without_reviewed_support")
    return result("reviewed_answer_and_support_matched", 1.0)


def generation_fix(path):
    """Static evidence of the historical fix, not a live Transformers test."""
    tree = ast.parse(path.read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name == "generation_kwargs"]
    if len(functions) != 1:
        return False
    for node in ast.walk(functions[0]):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            for key, value in zip(node.value.keys, node.value.values):
                if (isinstance(key, ast.Constant) and key.value == "use_model_defaults"
                        and isinstance(value, ast.Constant) and value.value is False):
                    return True
    return False


def preflight():
    """A non-launching readiness report, not an authorization/launch wrapper."""
    plan = load(BASELINE)
    frozen = plan["implementation_sha256"] | plan["additional_implementation_sha256"]
    unchanged = {p: file_hash(inside(ROOT, p)) == sha for p, sha in frozen.items()}
    observations = trainer_observations()
    hold_path = ROOT / "data/sft/passage-reader-pilot-v1/HOLD.json"
    hold = load(hold_path)
    return {"status": "blocked", "training_ready": False,
        "report_only_no_launch_guard": True,
        "target_model_identity_from_frozen_evaluation": plan["model_identity"],
        "source_revision": plan["source_revision"], "sandbox_image": plan["sandbox_image"],
        "training_sampling_settings": None,
        "paid_budget": {"requests": 0, "tokens": 0, "dollars": 0, "gpu_seconds": 0},
        "local_validation_hardware": {"system": os.uname().sysname,
            "release": os.uname().release, "machine": os.uname().machine,
            "python": platform.python_version()},
        "effective_prompt_sha256": PROMPT_SHA,
        "frozen_runtime": unchanged,
        "v1_preamble_unchanged": hashlib.sha256(TOOL_PREAMBLE.encode()).hexdigest()
                                 == V1_TOOL_PREAMBLE_SHA256,
        "legacy_trainer": {**observations,
            "generation_defaults_override_fix_present": generation_fix(
                ROOT / "scripts/train_qasper_grpo.py")},
        "data_hold": {"path": str(hold_path.relative_to(ROOT)), "sha256": file_hash(hold_path),
                      "status": hold["status"]},
        "blockers": [
            "Exposed finite-answer keys are not independent reward-validation evidence.",
            "No eligible reviewed training pool; the previous export remains on hold.",
            "No current V2 Docker-only raw-token rollout adapter or loss-mask parity proof.",
            "Historical host-execution/old-prompt trainer is incompatible with this task.",
            "No current-checkpoint GPU update, frozen-reference and save/reload proof.",
            "No authorized enforceable paid request/token/time/dollar budget."],
        "new_model_calls": 0, "gpu_activations": 0, "optimizer_updates": 0}


def validate(keys_path: Path, output: Path):
    if (output.resolve().is_relative_to(AUDIT.resolve())
            or AUDIT.resolve().is_relative_to(output.resolve())):
        raise ValueError("output must be separate from saved audit inputs")
    if output.exists():
        raise FileExistsError(output)
    checked_bundle(AUDIT)
    inputs = load(AUDIT / "inputs.json")
    keys = load(keys_path)
    cards = check_keys(keys, inputs)
    old = {r["id"]: r["scores"] for r in load(AUDIT / "results.json")}
    results = [{"record_id": r["id"], "context_id": r["context_id"],
                "original_label": r["expected"], "variant": r["variant"],
                "previous_proxy_rewards": {s: old[r["id"]][s] for s in SCORERS},
                **assess(r, inputs["contexts"][r["context_id"]], cards.get(r["context_id"]))}
               for r in inputs["records"]]
    repairs = []
    for cid, card in cards.items():
        control = next(r for r in inputs["records"] if r["id"] == card["source_control_id"])
        _, citations, spans = parse_strict(control["action"])
        for i, variant in enumerate(card["accepted_answers"]):
            record = {**control, "action": "SUBMIT: " + variant["answer"] + " CITATIONS: " +
                      json.dumps(citations) + " EVIDENCE: " + json.dumps(spans)}
            repairs.append({"context_id": cid, "variant_index": i, "action": record["action"],
                            **assess(record, inputs["contexts"][cid], card)})
    negatives = [r for r in results if r["eligible"] and r["original_label"] == "unacceptable"]
    positives = [r for r in results if r["eligible"] and r["original_label"] == "acceptable"]
    supported = {c["source_control_id"] for c in cards.values()
                 if c["original_control_review"] == "supported"}
    disagreements = [r for r in positives if r["reward"] != 1.0]
    gate = (all(r["reward"] == 0 for r in negatives)
            and all(r["reward"] == 1 for r in results if r["record_id"] in supported)
            and all(r["reward"] == 1 for r in repairs))
    summary = {"schema_version": VERSION, "records": len(results),
        "eligible_answerable_records": sum(r["eligible"] for r in results),
        "excluded_missing_scope_records": sum(not r["eligible"] for r in results),
        "known_answerable_negatives": len(negatives),
        "known_answerable_negatives_positive": sum(r["reward"] > 0 for r in negatives),
        "prior_proxies_on_same_answerable_negatives": {s: {
            "positive_reward": sum(r["previous_proxy_rewards"][s] > 0 for r in negatives),
            "full_reward": sum(r["previous_proxy_rewards"][s] >= 1 - 1e-12 for r in negatives)}
            for s in SCORERS},
        "original_answerable_positive_controls": len(positives),
        "original_positive_controls_rewarded": sum(r["reward"] == 1 for r in positives),
        "source_review_disagreements": disagreements,
        "reviewed_complete_variants": len(repairs),
        "reviewed_variants_rewarded": sum(r["reward"] == 1 for r in repairs),
        "reasons": dict(Counter(r["reason"] for r in results)),
        "exposed_development_gate_passed": gate, "training_reward_ready": False,
        "independent_reward_validation": None, "human_review": None,
        "new_model_calls": 0, "gpu_activations": 0, "optimizer_updates": 0,
        "limitation": "Finite reviewed whole-answer variants on already exposed cases; "
            "unlisted correct paraphrases score zero. Source support rests on attributed review, "
            "not executable entailment. No gap-quality reward or generalization measurement."}
    readiness = preflight()
    output.mkdir(parents=True, exist_ok=False)
    for name, value in {"results.json": results, "reviewed-variant-checks.json": repairs,
                        "summary.json": summary, "preflight.json": readiness}.items():
        write(output / name, value)
    write(output / "manifest.json", {"artifact_sha256": {p.name: file_hash(p)
        for p in sorted(output.iterdir())}, "inputs_sha256": file_hash(AUDIT / "inputs.json"),
        "reviewed_keys_sha256": file_hash(keys_path), "source_sha256": {
            p: file_hash(ROOT / p) for p in ("src/eval/reader_rl_validation.py",
                "scripts/validate_reader_rl.py", "scripts/train_qasper_grpo.py",
                "src/eval/qasper_reward.py", "src/env/reward.py")}})
    return summary
