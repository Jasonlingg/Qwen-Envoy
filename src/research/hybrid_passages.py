"""Local-only, annotation-blind retrieval under the existing two-passage budget."""
from __future__ import annotations

import hashlib
import math
import re
import time
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from src.research.reranker import (
    CrossEncoderReranker,
)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
ARMS = ("bm25", "dense", "hybrid", "hybrid_rerank")
CONFIG = {"window_chars": 1600, "stride_chars": 800, "selected_windows": 2,
          "max_selected_chars": 3200, "rrf_k": 60, "rerank_candidates": 10,
          "dense_max_tokens": 256, "reranker_max_tokens": 512, "batch_size": 16,
          "device": "cpu", "selection": "greedy disjoint; source-order output",
          "tie_break": "source start", "query": "source_question only"}


def tokens(text):
    return re.findall(r"\w+", text.lower())


def windows_for(text):
    if not isinstance(text, str) or not 3200 <= len(text) <= 1_000_000:
        raise ValueError("paper needs 3200..1000000 characters for two disjoint windows")
    starts = list(range(0, len(text) - 1599, 800))
    if starts[-1] + 1600 < len(text):
        starts.append(len(text) - 1600)
    return [{"start": start, "end": start + 1600} for start in starts]


def ranked_scores(scores):
    if not all(math.isfinite(float(score)) for score in scores):
        raise ValueError("nonfinite retrieval score")
    return sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))


def fuse(sparse, dense, k=60):
    if (type(k) is not int or k < 1 or len(sparse) != len(dense)
            or sorted(sparse) != list(range(len(sparse)))
            or sorted(dense) != sorted(sparse)):
        raise ValueError("fusion requires two complete candidate permutations and positive k")
    scores = np.zeros(len(sparse))
    for ranking in (sparse, dense):
        for rank, index in enumerate(ranking, 1):
            scores[index] += 1 / (k + rank)
    return ranked_scores(scores), scores


def select_disjoint(windows, ranking):
    selected = []
    for index in ranking:
        candidate = windows[index]
        if all(candidate["end"] <= w["start"] or candidate["start"] >= w["end"]
               for w in selected):
            selected.append(dict(candidate))
            if len(selected) == 2:
                return sorted(selected, key=lambda w: w["start"])
    raise ValueError("could not select two disjoint windows; question is not replaced")


class LocalModels:
    """Pinned CPU models: no automatic downloads, remote code or external inference."""

    def __init__(self):
        from sentence_transformers import SentenceTransformer

        self.encoder = SentenceTransformer(EMBEDDING_MODEL, revision=EMBEDDING_REVISION,
            device="cpu", local_files_only=True, trust_remote_code=False)
        self.encoder.max_seq_length = CONFIG["dense_max_tokens"]
        self.reranker = CrossEncoderReranker(device="cpu", local_files_only=True,
            batch_size=CONFIG["batch_size"], max_length=CONFIG["reranker_max_tokens"])

    def encode(self, texts):
        vectors = self.encoder.encode(texts, batch_size=CONFIG["batch_size"],
            show_progress_bar=False, normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vectors)

    def dense_lengths(self, texts):
        return [len(row) for row in self.encoder.tokenizer(texts, truncation=False,
                    padding=False, verbose=False)["input_ids"]]

    def rerank(self, query, passages):
        return self.reranker.rank(query, passages)

    def pair_lengths(self, query, passages):
        return [len(row) for row in self.reranker._model.tokenizer(
            [query] * len(passages), passages, truncation=False, padding=False,
            verbose=False)["input_ids"]]


@dataclass
class PaperIndex:
    source_sha256: str
    text: str
    windows: list
    sparse: object
    vectors: np.ndarray
    timing: dict
    truncated_dense_windows: int


def build_index(text, models):
    windows = windows_for(text)
    passages = [text[w["start"]:w["end"]] for w in windows]
    start = time.perf_counter()
    corpus = [tokens(p) for p in passages]
    if not all(corpus):
        raise ValueError("empty retrieval window")
    sparse = BM25Okapi(corpus)
    lexical_seconds = time.perf_counter() - start
    start = time.perf_counter()
    vectors = models.encode(passages)
    dense_seconds = time.perf_counter() - start
    if (vectors.ndim != 2 or len(vectors) != len(windows)
            or not np.isfinite(vectors).all()):
        raise ValueError("invalid dense index")
    lengths = models.dense_lengths(passages)
    return PaperIndex(hashlib.sha256(text.encode()).hexdigest(), text, windows, sparse,
        vectors, {"lexical_index_seconds": lexical_seconds, "dense_index_seconds": dense_seconds},
        sum(n > CONFIG["dense_max_tokens"] for n in lengths))


def retrieve(index, query, arm, models):
    """Only source text/query reach models. Offsets are metadata, not inspection."""
    if arm not in ARMS or not isinstance(query, str) or not query.strip() or len(query) > 4000:
        raise ValueError("unknown method or invalid query")
    started = time.perf_counter()
    sparse_scores = index.sparse.get_scores(tokens(query))
    sparse = ranked_scores(sparse_scores)
    ranking, scores, pool = sparse, sparse_scores, []
    dense_query_truncated = False
    if arm != "bm25":
        vector = models.encode([query])
        if vector.shape != (1, index.vectors.shape[1]) or not np.isfinite(vector).all():
            raise ValueError("invalid dense query vector")
        dense_scores = index.vectors @ vector[0]
        dense = ranked_scores(dense_scores)
        ranking, scores = ((dense, dense_scores) if arm == "dense" else fuse(sparse, dense))
    if arm == "hybrid_rerank":
        pool = ranking[:CONFIG["rerank_candidates"]]
        passages = [index.text[index.windows[i]["start"]:index.windows[i]["end"]] for i in pool]
        scored = models.rerank(query, passages)
        if (len(scored) != len(pool) or sorted(i for i, _ in scored) != list(range(len(pool)))
                or any(not math.isfinite(float(score)) for _, score in scored)):
            raise ValueError("reranker must return exactly one finite score per candidate")
        scored = sorted(scored, key=lambda row: (-float(row[1]), pool[row[0]]))
        ranking = [pool[i] for i, _ in scored]
        scores = {pool[i]: float(score) for i, score in scored}
    error = None
    try:
        selected = select_disjoint(index.windows, ranking)
    except ValueError as exc:
        selected, error = [], str(exc)
    seconds = time.perf_counter() - started
    # Diagnostics are outside retrieval timing; they do not affect ranking/selection.
    if arm != "bm25":
        dense_query_truncated = models.dense_lengths([query])[0] > CONFIG["dense_max_tokens"]
    truncated_pairs = 0
    if pool:
        truncated_pairs = sum(n > CONFIG["reranker_max_tokens"]
                              for n in models.pair_lengths(query, passages))
    return {"method": arm, "status": "error" if error else "selected", "error": error,
        "source_sha256": index.source_sha256, "windows": selected,
        "ranking": [{**index.windows[i], "score": float(scores[i])} for i in ranking],
        "candidate_pool": [dict(index.windows[i]) for i in pool],
        "query_seconds": seconds, "dense_query_truncated": dense_query_truncated,
        "truncated_reranker_pairs": truncated_pairs,
        "counts_as_source_inspection": False}
