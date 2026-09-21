"""QASPER annotation reward for code-execution experiments.

This is a training proxy, not a semantic-support judge. MuSiQue's outcome-v1
reward is deliberately unchanged. Gold annotations never enter the agent prompt.
"""
from __future__ import annotations

from collections import Counter
import json
import re
import string

REWARD_VERSION = "qasper-answer-evidence-v1"


def normalize(text: str) -> str:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def answer_f1(predicted: str, reference: str) -> float:
    """Official QASPER/SQuAD normalization and multiset token F1."""
    p, r = normalize(predicted).split(), normalize(reference).split()
    if not p or not r:
        return float(p == r)
    common = sum((Counter(p) & Counter(r)).values())
    return 2.0 * common / (len(p) + len(r))


def parse_strict(action: str) -> tuple[str, list[str], list[dict]] | None:
    match = re.fullmatch(r"SUBMIT:\s*(.*?)\s+CITATIONS:\s*(.*)", action.strip(), re.S)
    if not match or not match[1].strip():
        return None
    try:
        citations, end = json.JSONDecoder().raw_decode(match[2])
        rest = match[2][end:].strip()
        if not isinstance(citations, list) or any(not isinstance(c, str) for c in citations):
            return None
        if len(citations) != len(set(citations)):
            return None
        if not rest.startswith("EVIDENCE:"):
            return None
        evidence = json.loads(rest[len("EVIDENCE:"):].strip())
        if not isinstance(evidence, list) or any(not isinstance(e, dict) for e in evidence):
            return None
        return match[1].strip(), citations, evidence
    except (ValueError, TypeError):
        return None


def _merge(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def evidence_scores(predicted: list[dict], gold: list[dict]) -> tuple[float, float]:
    """Character coverage F1 and diagnostic paragraph recall.

    Coverage prevents one-character citations or whole-paper citations from
    receiving full credit. Adjacent/duplicate spans cannot inflate recall.
    """
    by_doc: dict[str, list[tuple[int, int]]] = {}
    for span in predicted:
        by_doc.setdefault(span["doc_id"], []).append((span["start"], span["end"]))
    predicted_count = sum(b-a for spans in by_doc.values() for a,b in _merge(spans))
    gold_by_doc: dict[str, list[tuple[int, int]]] = {}
    for span in gold:
        gold_by_doc.setdefault(span["doc_id"], []).append((span["start"], span["end"]))
    gold_count = sum(b-a for spans in gold_by_doc.values() for a,b in _merge(spans))
    overlap = sum(
        max(0, min(b, d)-max(a, c))
        for doc, spans in by_doc.items() for a,b in _merge(spans)
        for c,d in _merge(gold_by_doc.get(doc, []))
    )
    coverage = 2*overlap/(predicted_count+gold_count) if predicted_count+gold_count else 1.0
    # Paragraph recall is useful diagnostically; reward uses coverage instead.
    hits = sum(any(p["doc_id"] == g["doc_id"] and min(p["end"],g["end"]) >
                   max(p["start"],g["start"]) for p in predicted) for g in gold)
    paragraph_recall = hits/len(gold) if gold else float(not predicted)
    return coverage, paragraph_recall


def score_submission(action: str, question: dict, documents: dict[str, dict],
                     *, investigated: bool) -> dict:
    result = {"reward_version": REWARD_VERSION, "reward": 0.0, "answer_f1": 0.0,
              "evidence_f1": 0.0, "paragraph_recall": 0.0, "valid": False,
              "abstained": False, "reason": "invalid_submission"}
    parsed = parse_strict(action)
    if parsed is None:
        return result
    answer, citations, evidence = parsed
    result["abstained"] = normalize(answer) == "unanswerable"
    if not investigated:
        result["reason"] = "no_successful_document_action"
        return result
    for span in evidence:
        if not isinstance(span.get("doc_id"), str):
            result["reason"] = "invalid_span"
            return result
        doc = documents.get(span.get("doc_id"))
        start, end = span.get("start"), span.get("end")
        if (doc is None or type(start) is not int or type(end) is not int
                or not 0 <= start < end <= len(doc["text"])):
            result["reason"] = "invalid_span"
            return result
    if set(citations) != {e["doc_id"] for e in evidence}:
        result["reason"] = "citations_evidence_disagree"
        return result
    result["valid"] = True
    annotations = question["answer_annotations"]
    if all(a["unanswerable"] for a in annotations):
        correct = result["abstained"] and not citations and not evidence
        result.update(reward=float(correct), answer_f1=float(correct),
                      reason="correct_abstention" if correct else "unsupported_answer")
        return result
    if result["abstained"]:
        result["reason"] = "false_refusal"
        return result
    if not evidence:
        result["reason"] = "missing_evidence"
        return result
    candidates = []
    for ref in annotations:
        if ref["unanswerable"]:
            continue
        a = answer_f1(answer, ref["answer_text"])
        if ref["answer_type"] == "boolean":
            a = float(normalize(answer) == normalize(ref["answer_text"]))
        e, paragraph = evidence_scores(evidence, ref["evidence"])
        # Both components must describe the same annotation. Correctness remains
        # primary; valid but irrelevant spans cannot exceed half credit.
        candidates.append((a*(0.5+0.5*e), a, e, paragraph))
    reward, answer_score, evidence_score, paragraph = max(candidates, default=(0,0,0,0))
    result.update(reward=reward, answer_f1=answer_score, evidence_f1=evidence_score,
                  paragraph_recall=paragraph, reason="scored")
    return result
