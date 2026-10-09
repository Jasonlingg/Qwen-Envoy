from __future__ import annotations

import json

from rank_bm25 import BM25Okapi

from src.env.corpus import Corpus
from src.policies.qwen_rag import QwenRAGPolicy, _extract_question, _tokenize


def _corpus(tmp_path):
    documents = [
        {
            "doc_id": "alpha",
            "title": "Adaptive Retrieval",
            "text": "routing chooses no retrieval, one retrieval, or iterative retrieval. " * 20,
        },
        {
            "doc_id": "beta",
            "title": "Corrective Retrieval",
            "text": (
                "a retrieval evaluator refines correct results and replaces incorrect ones. " * 20
            ),
        },
        {
            "doc_id": "noise",
            "title": "Unrelated Vision Work",
            "text": "pixels images convolution objects. " * 20,
        },
    ]
    for document in documents:
        (tmp_path / f"{document['doc_id']}.json").write_text(json.dumps(document))
    corpus = Corpus(corpus_path=str(tmp_path), chunk_size=180, chunk_overlap=20)
    corpus.load(build_index=False)
    return corpus


def test_question_extraction_ignores_environment_preamble():
    observation = "Tools available\nQuestion: Compare adaptive and corrective retrieval.\n"
    assert _extract_question(observation) == "Compare adaptive and corrective retrieval."


def test_sparse_retrieval_is_document_diverse_without_loading_model(tmp_path):
    policy = object.__new__(QwenRAGPolicy)
    policy.corpus = _corpus(tmp_path)
    policy.top_docs = 2
    policy.chunks_per_doc = 2
    policy.max_context_chars = 4_000
    policy._passages = policy._build_passages()
    policy._bm25 = BM25Okapi([
        _tokenize(f"{item['title']} {item['title']} {item['text']}")
        for item in policy._passages
    ])

    results = policy._retrieve("Compare adaptive routing with corrective retrieval")

    assert {item["doc_id"] for item in results} == {"alpha", "beta"}
    assert len(results) == 4
    assert all(item["start"] < item["end"] for item in results)
    assert sum(len(item["text"]) for item in results) <= policy.max_context_chars
