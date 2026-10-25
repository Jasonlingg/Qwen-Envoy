"""Offline Answer-F1 candidate; no changes to the active reader reward."""

import json

from benchmarks.envoybench.qasper_official import official_evaluator
from src.eval.qasper_reward import parse_strict
from src.eval.reader_improvement import digest
from src.eval.reader_rl_validation import assess

VERSION = "reader-answer-f1-reference-support-candidate-v1"


def assess_f1(record, context, card):
    """Use trusted evaluator receipts and a prevalidated frozen reference card.

    Reuse the old assessor's inspection/clipping/support checks by substituting
    one known reference answer in a PRIVATE copy, solely for the evidence gate.
    Score the ORIGINAL answer against every original text QASPER reference.
    No substituted answer is executed, shown to the policy, or saved as its output.
    This gate matches attributed support ranges; it does not judge entailment.
    """
    if context["scope_answerability"] == "missing":
        return {"eligible": False, "reward": None, "reason": "gap_quality_not_scored"}
    if (card is None or card["context_sha256"] != digest(context)
            or record["context_id"] != card["context_id"]
            or card["role"] not in {"train", "reward_validation"}):
        raise ValueError("missing, stale or forbidden reward reference")

    def result(reward, reason, answer_f1=None):
        return {"eligible": True, "reward": reward, "reason": reason,
                "answer_f1": answer_f1, "reward_version": VERSION,
                "training_ready": False,
                "support_authority": "attributed_reference_review_not_automatic_entailment"}

    action = record.get("action")
    try:
        parsed = parse_strict(action) if isinstance(action, str) and len(action) <= 8000 else None
    except (RecursionError, ValueError):
        parsed = None
    if parsed is None:
        return result(0.0, "invalid_terminal")
    answer, citations, evidence = parsed
    references = [a["answer_text"] for a in context["annotations"]
                  if not a["unanswerable"] and not a["requires_nontext_evidence"]]
    if not references:
        raise ValueError("no answerable original text references; reward unavailable")
    evaluator = official_evaluator()
    f1 = max(evaluator.token_f1_score(answer, ref) for ref in references)
    gate_action = ("SUBMIT: " + card["accepted_answers"][0]["answer"]
                   + " CITATIONS: " + json.dumps(citations)
                   + " EVIDENCE: " + json.dumps(evidence))
    gate = assess({**record, "action": gate_action}, context, card)
    if gate["reward"] != 1.0:
        return result(0.0, gate["reason"], f1)
    return result(f1, "answer_f1_with_reference_support", f1)
