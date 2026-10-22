"""Score frozen retrieval outputs; annotations never enter a retrieval model."""
from __future__ import annotations

import math

from src.research.hybrid_passages import ARMS


def merge(ranges):
    merged = []
    for start, end in sorted(ranges):
        if type(start) is not int or type(end) is not int or not 0 <= start < end:
            raise ValueError("invalid annotation/selection offsets")
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def coverage(windows, gold, doc_id, text_length):
    """Paragraph completeness and gold-character coverage, not semantic support."""
    selected = merge((w["start"], w["end"]) for w in windows)
    if any(end > text_length for _, end in selected):
        raise ValueError("selected window exceeds source")
    spans = []
    for span in gold:
        if span.get("doc_id") != doc_id or span["end"] > text_length:
            raise ValueError("gold span differs from target paper")
        spans.append((span["start"], span["end"]))
    merged_gold = merge(spans)

    def overlap(start, end):
        return sum(max(0, min(end, e) - max(start, s)) for s, e in selected)

    full = [overlap(s, e) == e - s for s, e in spans]
    total = sum(e - s for s, e in merged_gold)
    return {"gold_paragraph_count": len(spans), "any_overlap": any(overlap(s, e) for s, e in spans),
        "any_full_paragraph": any(full), "all_full_paragraphs": all(full) if full else None,
        "gold_character_fraction": sum(overlap(s, e) for s, e in merged_gold) / total
        if total else None}


def percentile95(values):
    return sorted(values)[math.ceil(0.95 * len(values)) - 1] if values else None


def summarize(rows):
    summaries = {}
    for arm in ARMS:
        all_rows = [r for r in rows if r["method"] == arm]
        scored = [r for r in all_rows if r["reference_answerability"] == "sufficient"
                  and r["coverage"]["gold_paragraph_count"]]
        summaries[arm] = {"question_count": len(all_rows), "scored_answerable_count": len(scored),
            "answerable_without_gold_ids": [r["question_id"] for r in all_rows
                if r["reference_answerability"] == "sufficient"
                and not r["coverage"]["gold_paragraph_count"]],
            "unanswerable_not_scored": sum(r["reference_answerability"] == "insufficient"
                                           for r in all_rows),
            "selection_failures": sum(r["status"] != "selected" for r in all_rows),
            "any_overlap_count": sum(r["coverage"]["any_overlap"] for r in scored),
            "any_full_paragraph_count": sum(r["coverage"]["any_full_paragraph"] for r in scored),
            "all_full_paragraphs_count": sum(r["coverage"]["all_full_paragraphs"] for r in scored),
            "mean_gold_character_fraction": sum(r["coverage"]["gold_character_fraction"]
                for r in scored) / len(scored) if scored else None,
            "query_seconds_p95": percentile95([r["query_seconds"] for r in all_rows]),
            "full_paragraph_wins_vs_bm25": [], "full_paragraph_losses_vs_bm25": []}
        baseline = {r["question_id"]: r for r in rows if r["method"] == "bm25"}
        for row in scored:
            before = baseline[row["question_id"]]["coverage"]["any_full_paragraph"]
            after = row["coverage"]["any_full_paragraph"]
            if before != after:
                key = "full_paragraph_wins_vs_bm25" if after else "full_paragraph_losses_vs_bm25"
                summaries[arm][key].append(row["question_id"])
    return summaries


def decide(reports):
    dev, test = (reports[name]["methods"] for name in ("dev", "test_candidate"))
    baseline, candidate = dev["bm25"], dev["hybrid_rerank"]
    gates = {
        "candidate_has_no_selection_failures": all(
            methods["hybrid_rerank"]["selection_failures"] == 0 for methods in (dev, test)),
        "dev_full_paragraph_gain_at_least_two": candidate["any_full_paragraph_count"]
            >= baseline["any_full_paragraph_count"] + 2,
        "dev_any_overlap_not_reduced": candidate["any_overlap_count"]
            >= baseline["any_overlap_count"],
        "secondary_full_paragraph_count_not_reduced": test["hybrid_rerank"][
            "any_full_paragraph_count"] >= test["bm25"]["any_full_paragraph_count"],
        "query_p95_within_two_seconds": all(0 <= methods["hybrid_rerank"][
            "query_seconds_p95"] <= 2 for methods in (dev, test)),
    }
    return {"candidate": "hybrid_rerank", "gates": gates, "passed": all(gates.values()),
            "production_default_changed": False, "qwen_answer_improvement": "not_measured"}
