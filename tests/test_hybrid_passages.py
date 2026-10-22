"""Retrieval is annotation-blind, exact-offset, bounded, and distinct from support."""
import copy

import numpy as np
import pytest

from src.eval.passage_retrieval_comparison import coverage, decide, summarize
from src.eval.qasper_passage_smoke import retrieve as legacy_retrieve
from src.research.hybrid_passages import (
    ARMS,
    build_index,
    fuse,
    retrieve,
    select_disjoint,
    windows_for,
)


class FakeModels:
    def __init__(self):
        self.calls = []

    def encode(self, texts):
        assert isinstance(texts, list) and all(isinstance(t, str) for t in texts)
        self.calls.append(("encode", texts))
        return np.asarray([[0., 1.] if "target" in text else [1., 0.] for text in texts])

    def dense_lengths(self, texts):
        return [300 for _ in texts]

    def pair_lengths(self, query, passages):
        return [600 for _ in passages]

    def rerank(self, query, passages):
        self.calls.append(("rerank", query, passages))
        return [(i, float(i)) for i in reversed(range(len(passages)))]


TEXT = "background text " * 900 + "target evidence " * 100


def test_bm25_is_byte_for_byte_compatible_in_windows_and_scores():
    models = FakeModels()
    index = build_index(TEXT, models)
    result = retrieve(index, "target evidence", "bm25", models)
    windows, ranking = legacy_retrieve(TEXT, "target evidence")
    assert result["windows"] == windows and result["ranking"] == ranking
    assert result["counts_as_source_inspection"] is False


def test_fixed_grid_includes_end_aligned_window_and_rejects_tiny_source():
    windows = windows_for(TEXT)
    assert windows[-1]["end"] == len(TEXT)
    assert all(w["end"] - w["start"] == 1600 for w in windows)
    with pytest.raises(ValueError):
        windows_for("tiny")


@pytest.mark.parametrize("arm", ARMS)
def test_each_arm_returns_two_exact_disjoint_source_windows(arm):
    models = FakeModels()
    index = build_index(TEXT, models)
    result = retrieve(index, "target evidence", arm, models)
    a, b = result["windows"]
    assert 0 <= a["start"] < a["end"] <= b["start"] < b["end"] <= len(TEXT)
    assert a["end"] - a["start"] + b["end"] - b["start"] == 3200
    assert result["truncated_reranker_pairs"] == (10 if arm == "hybrid_rerank" else 0)
    assert index.truncated_dense_windows == len(index.windows)


def test_reranking_is_confined_to_ten_hybrid_candidates_and_no_annotations():
    models = FakeModels()
    index = build_index(TEXT, models)
    before = retrieve(index, "target evidence", "hybrid", models)
    after = retrieve(index, "target evidence", "hybrid_rerank", models)
    assert after["candidate_pool"] == [{k: row[k] for k in ("start", "end")}
                                      for row in before["ranking"][:10]]
    _, query, passages = next(call for call in models.calls if call[0] == "rerank")
    assert query == "target evidence" and len(passages) == 10
    assert all(len(p) == 1600 for p in passages)
    assert all(set(w) == {"start", "end"} for w in after["windows"])
    assert retrieve(index, query, "hybrid_rerank", models)["windows"] == after["windows"]


@pytest.mark.parametrize("a,b", [([0, 0], [0, 1]), ([0], [0, 1]), ([0, 1], [0, 2])])
def test_fusion_rejects_missing_or_duplicate_candidates(a, b):
    with pytest.raises(ValueError):
        fuse(a, b)


def test_fusion_uses_fixed_rrf_and_source_tie_order():
    ranking, scores = fuse([1, 0, 2], [0, 1, 2])
    assert ranking == [0, 1, 2]
    assert scores[0] == scores[1] == 1 / 61 + 1 / 62


@pytest.mark.parametrize("bad", [[(0, 1.)], [(i, float('nan')) for i in range(10)],
                                 [(0, 1.)] * 10])
def test_malformed_reranker_is_an_error_not_silent_baseline_fallback(bad):
    models = FakeModels()
    index = build_index(TEXT, models)
    models.rerank = lambda *_: bad
    with pytest.raises(ValueError):
        retrieve(index, "target", "hybrid_rerank", models)


def test_disjoint_selection_failure_does_not_return_one_window_as_success():
    windows = [{"start": i, "end": i + 1600} for i in [0, 800, 1600]]
    with pytest.raises(ValueError):
        select_disjoint(windows, [1, 0, 2])


def test_overlap_is_not_full_paragraph_coverage_and_overlaps_are_not_double_counted():
    gold = [{"doc_id": "p", "start": 100, "end": 200}]
    result = coverage([{"start": 150, "end": 180}, {"start": 160, "end": 190}], gold, "p", 1000)
    assert result["any_overlap"] and not result["any_full_paragraph"]
    assert result["gold_character_fraction"] == 0.4
    complete = coverage([{"start": 100, "end": 150}, {"start": 150, "end": 200}], gold, "p", 1000)
    assert complete["any_full_paragraph"] and complete["gold_character_fraction"] == 1
    with pytest.raises(ValueError):
        coverage([], gold, "wrong", 1000)


def rows():
    out = []
    for arm in ARMS:
        for name, answerability, selected in [
            ("success", "sufficient", [{"start": 0, "end": 100}]),
            ("failure", "sufficient", []), ("unanswerable", "insufficient", [])]:
            out.append({"method": arm, "question_id": name,
                "reference_answerability": answerability,
                "status": "error" if name == "failure" else "selected",
                "query_seconds": 0.1, "coverage": coverage(selected,
                 [{"doc_id": "p", "start": 10, "end": 50}] if name != "unanswerable" else [],
                 "p", 100)})
    return out


def test_selection_errors_stay_in_denominator_and_unanswerable_is_not_a_hit():
    result = summarize(rows())
    for method in result.values():
        assert method["scored_answerable_count"] == 2
        assert method["any_full_paragraph_count"] == 1
        assert method["selection_failures"] == method["unanswerable_not_scored"] == 1


def test_decision_uses_preselected_method_and_all_gates_not_best_test_arm():
    summary = summarize(rows())
    reports = {split: {"methods": copy.deepcopy(summary)} for split in ("dev", "test_candidate")}
    assert not decide(reports)["passed"]
    reports["dev"]["methods"]["hybrid_rerank"]["any_full_paragraph_count"] += 2
    for report in reports.values():
        report["methods"]["hybrid_rerank"]["selection_failures"] = 0
    assert decide(reports)["passed"]
    reports["test_candidate"]["methods"]["hybrid_rerank"]["query_seconds_p95"] = 2.1
    assert not decide(reports)["passed"]
    assert decide(reports)["production_default_changed"] is False


def test_failed_neural_arm_preserves_baseline_and_other_arm_outputs():
    from scripts.compare_passage_retrieval import evaluate_arm

    models = FakeModels()
    index = build_index(TEXT, models)
    baseline = evaluate_arm(TEXT, "target", "bm25", models, index)
    dense = evaluate_arm(TEXT, "target", "dense", models, index)
    models.rerank = lambda *_: [(0, float("nan"))]
    failed = evaluate_arm(TEXT, "target", "hybrid_rerank", models, index)
    assert failed["status"] == "error" and failed["windows"] == []
    assert baseline["status"] == dense["status"] == "selected"
    fallback = evaluate_arm(TEXT, "target", "bm25", models, index_error="neural index failed")
    assert fallback["windows"] == baseline["windows"]
    assert evaluate_arm(TEXT, "target", "dense", models,
                        index_error="neural index failed")["status"] == "error"
