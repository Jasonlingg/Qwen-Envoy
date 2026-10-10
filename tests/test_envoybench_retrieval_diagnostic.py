"""Offline checks for fixed retrieval ranking and token-bounded packet selection."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.envoybench import retrieval_baseline as bm25
from benchmarks.envoybench import retrieval_diagnostic as diagnostic


class CharacterCounter:
    def __call__(self, messages: list[dict]) -> int:
        return sum(len(message["content"]) for message in messages)


def _question(**gold_fields) -> dict:
    return {
        "id": "q1",
        "question": 'Use the known paper "Test" (doc_id: "paper") to answer: What happened?',
        **gold_fields,
    }


def test_dense_and_rrf_rank_deterministically_without_gold_fields() -> None:
    question = _question(answer="secret", gold_evidence=[{"text": "secret"}])
    corrupted = _question(answer="opposite", expected_answerability="insufficient",
                          gold_evidence=[{"text": "opposite"}])
    windows = [
        {"start": 0, "end": 100, "text": "first"},
        {"start": 80, "end": 180, "text": "second"},
        {"start": 160, "end": 260, "text": "third"},
    ]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    query_vector = np.array([1.0, 0.0])

    def rank(q: dict) -> list[dict]:
        _, visible_query = bm25.question_target(q)
        assert visible_query == "What happened?"
        dense = diagnostic.rank_dense(windows, vectors, query_vector)
        sparse = [windows[1], windows[0], windows[2]]
        return diagnostic.rank_hybrid(sparse, dense)

    assert rank(question) == rank(corrupted)
    assert [row["start"] for row in diagnostic.rank_dense(
        windows, vectors, query_vector,
    )] == [0, 160, 80]
    assert [row["start"] for row in rank(question)] == [0, 80, 160]
    assert rank(question) == rank(question)
    counter = CharacterCounter()
    packet = diagnostic.select_packet(
        question["id"], question["question"], "paper", rank(question), counter,
        context_window=10_000,
    )
    mutated_packet = diagnostic.select_packet(
        corrupted["id"], corrupted["question"], "paper", rank(corrupted), counter,
        context_window=10_000,
    )
    assert packet == mutated_packet


def test_packet_uses_full_chat_token_budget_and_exact_windows() -> None:
    question = _question()
    ranked = [
        {"start": index * 150, "end": index * 150 + 160,
         "text": "x" * 160, "dense_score": 1 / (index + 1)}
        for index in range(20)
    ]
    counter = CharacterCounter()
    empty = counter(bm25.packet_messages(question["question"], []))
    one = counter(bm25.packet_messages(question["question"], [{
        **ranked[0], "doc_id": "paper", "passage_id": "paper:0:160",
    }]))
    budget = one + 5
    packet = diagnostic.select_packet(
        question["id"], question["question"], "paper", ranked, counter,
        context_window=budget + 100 + 20, output_reserve=100, safety_margin=20,
    )
    assert len(packet["passages"]) == 1
    assert packet["passages"][0]["text"] == ranked[0]["text"]
    assert packet["prompt_tokens_local"] <= budget
    assert packet["prompt_tokens_local"] > empty
    assert packet["input_token_budget"] == budget
    with pytest.raises(ValueError, match="differs"):
        diagnostic.select_packet(question["id"], question["question"], "other",
                                 ranked, counter, context_window=10_000)


def test_span_coverage_counts_overlaps_once() -> None:
    span = {"doc_id": "paper", "start": 10, "end": 110}
    passages = [
        {"doc_id": "paper", "start": 0, "end": 60},
        {"doc_id": "paper", "start": 40, "end": 100},
        {"doc_id": "other", "start": 100, "end": 110},
    ]
    assert diagnostic._covered_fraction(span, passages) == 0.9


def test_hybrid_refuses_different_candidate_sets() -> None:
    with pytest.raises(ValueError, match="identical windows"):
        diagnostic.rank_hybrid(
            [{"start": 0, "end": 10}], [{"start": 1, "end": 11}],
        )
