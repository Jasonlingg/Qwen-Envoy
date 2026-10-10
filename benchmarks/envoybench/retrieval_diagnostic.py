"""Offline BM25/dense/hybrid retrieval diagnostic on the frozen QASPER questions.

This measures whether supplied *passages* overlap unreviewed QASPER evidence.
It sends no model requests and does not measure answer correctness or support.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

from benchmarks.envoybench import retrieval_baseline as baseline
from benchmarks.envoybench.run import DEFAULT_DATASET, _file_sha256
from src.eval.artifacts import configuration_hash
from src.eval.research_review import load_corpus

EMBEDDING_REPOSITORY = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
RRF_K = 60
EVIDENCE_THRESHOLD = 0.8


def _snapshot_path() -> Path:
    """Resolve the pinned local snapshot without contacting Hugging Face."""
    cache = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
    path = (cache / "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots"
            / EMBEDDING_REVISION)
    required = ("modules.json", "model.safetensors", "tokenizer.json")
    if not all((path / item).is_file() for item in required):
        raise ValueError(f"Pinned MiniLM snapshot is not cached: {path}")
    return path


def _passage_key(row: dict) -> tuple[int, int]:
    return row["start"], row["end"]


def rank_dense(windows: list[dict], vectors: np.ndarray, query_vector: np.ndarray) -> list[dict]:
    """Cosine rank of the same windows used by BM25, with offset tie-breaks."""
    if len(windows) != len(vectors):
        raise ValueError("one embedding is required for each window")
    if vectors.ndim != 2 or query_vector.ndim != 1 or vectors.shape[1] != len(query_vector):
        raise ValueError("embedding dimensions differ")
    scores = vectors @ query_vector
    return sorted(
        ({**row, "dense_score": float(score)} for row, score in zip(windows, scores)),
        key=lambda row: (-row["dense_score"], row["start"]),
    )


def rank_hybrid(bm25: list[dict], dense: list[dict], *, rrf_k: int = RRF_K) -> list[dict]:
    """Fixed reciprocal-rank fusion; no weights or parameters fitted to gold."""
    if rrf_k <= 0 or {_passage_key(row) for row in bm25} != {
        _passage_key(row) for row in dense
    }:
        raise ValueError("hybrid rankers must cover identical windows and positive k")
    bm25_ranks = {_passage_key(row): rank for rank, row in enumerate(bm25, 1)}
    dense_ranks = {_passage_key(row): rank for rank, row in enumerate(dense, 1)}
    return sorted(
        (
            {**row, "rrf_score": 1 / (rrf_k + bm25_ranks[_passage_key(row)])
             + 1 / (rrf_k + dense_ranks[_passage_key(row)])}
            for row in bm25
        ),
        key=lambda row: (-row["rrf_score"], row["start"]),
    )


def select_packet(question_id: str, question_text: str, doc_id: str,
                  ranked: list[dict], count_tokens, *, context_window: int = 8192,
                  output_reserve: int = 1024, safety_margin: int = 128,
                  max_passages: int = 12) -> dict:
    """Select full ranked windows under the exact baseline rendered-prompt cap."""
    target, _ = baseline.question_target({"id": question_id, "question": question_text})
    if target != doc_id:
        raise ValueError("ranked document differs from the named paper")
    input_budget = context_window - output_reserve - safety_margin
    if input_budget <= 0 or not 1 <= max_passages <= 32:
        raise ValueError("invalid prompt budget or passage limit")
    if count_tokens(baseline.packet_messages(question_text, [])) > input_budget:
        raise ValueError("question and instructions alone exceed prompt budget")
    selected = []
    for row in ranked:
        passage = {**row, "doc_id": doc_id,
                   "passage_id": f"{doc_id}:{row['start']}:{row['end']}"}
        proposed = selected + [passage]
        if count_tokens(baseline.packet_messages(question_text, proposed)) <= input_budget:
            selected = proposed
        if len(selected) == max_passages:
            break
    if not selected:
        raise ValueError(f"{question_id}: no complete passage fits the prompt budget")
    messages = baseline.packet_messages(question_text, selected)
    return {"question_id": question_id, "doc_id": doc_id, "passages": selected,
            "prompt_tokens_local": count_tokens(messages), "input_token_budget": input_budget,
            "packet_hash": configuration_hash({"messages": messages})}


def _covered_fraction(span: dict, passages: list[dict]) -> float:
    """Fraction of an annotated span contained in the union of selected windows."""
    start, end = span["start"], span["end"]
    if not isinstance(start, int) or not isinstance(end, int) or end <= start:
        raise ValueError("malformed annotated evidence span")
    intervals = sorted(
        (max(start, p["start"]), min(end, p["end"]))
        for p in passages if p["doc_id"] == span["doc_id"]
        and p["start"] < end and start < p["end"]
    )
    covered = 0
    cursor = start
    for left, right in intervals:
        if right > cursor:
            covered += right - max(left, cursor)
            cursor = right
    return covered / (end - start)


def _coverage_summary(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["expected_answerability"] == "sufficient"]
    spans = [fraction for row in answerable for fraction in row["span_coverage"]]
    return {
        "answerable_questions": len(answerable),
        "questions_with_any_annotated_span_at_least_80pct": sum(
            any(frac >= EVIDENCE_THRESHOLD for frac in row["span_coverage"])
            for row in answerable
        ),
        "annotated_spans": len(spans),
        "spans_at_least_80pct": sum(frac >= EVIDENCE_THRESHOLD for frac in spans),
        "mean_selected_passages": sum(len(r["selected_passages"]) for r in rows) / len(rows),
        "mean_prompt_tokens_local": sum(r["prompt_tokens_local"] for r in rows) / len(rows),
    }


def _metadata(row: dict) -> dict:
    return {key: row[key] for key in ("doc_id", "start", "end", "passage_id")}


def run(prepared: Path, source_run: Path, dataset: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    start_time = time.monotonic()
    plan = json.loads((prepared / "plan.json").read_text())
    packets_doc = json.loads((prepared / "packets.json").read_text())
    if (plan.get("status") != "prepared_not_run"
            or plan.get("schema_version") != baseline.PREPARED_SCHEMA):
        raise ValueError("expected a prepared, unexecuted BM25 comparator")
    config = plan["retrieval_config"]
    expected_config = {"context_window": 8192, "safety_margin": 128,
                       "chunk_chars": 1400, "overlap_chars": 200, "max_passages": 12}
    if config != expected_config:
        raise ValueError("prepared comparator does not have the frozen retrieval settings")
    # Regeneration checks source/benchmark/corpus/tokenizer/implementation bindings.
    regenerated_plan, bm25_packets = baseline.build_preparation(dataset, source_run, **config)
    if regenerated_plan != plan or packets_doc != {"packets": bm25_packets}:
        raise ValueError("prepared BM25 packets or frozen inputs changed")
    source, benchmark, corpus = baseline._source(dataset, source_run)
    if len(benchmark["questions"]) != 40 or plan["question_ids"] != [
        q["id"] for q in benchmark["questions"]
    ]:
        raise ValueError("diagnostic requires the same frozen 40 questions")
    documents = load_corpus(corpus)
    snapshot = _snapshot_path()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(snapshot), device="cpu", local_files_only=True)
    counter = baseline.LocalTokenCounter(plan["tokenizer"]["revision"],
                                         plan["tokenizer"]["chat_template_kwargs"])
    if counter.metadata != plan["tokenizer"]:
        raise ValueError("cached tokenizer or chat template differs from preparation")
    initialization_seconds = time.monotonic() - start_time
    queries = [{"id": q["id"], "text": q["question"],
                "doc_id": baseline.question_target(q)[0],
                "query": baseline.question_target(q)[1]}
               for q in benchmark["questions"]]
    doc_vectors = {}
    methods = {name: [] for name in ("bm25", "dense", "hybrid_rrf")}
    retrieval_start = time.monotonic()
    for query, bm25_packet in zip(queries, bm25_packets):
        doc_id = query["doc_id"]
        document = documents[doc_id]
        bm25_ranked = baseline.rank_passages(
            document["text"], query["query"], chunk_chars=config["chunk_chars"],
            overlap_chars=config["overlap_chars"],
        )
        if doc_id not in doc_vectors:
            # The source text, not QASPER labels or reference evidence, is encoded.
            original_order = sorted(bm25_ranked, key=lambda row: row["start"])
            vectors = model.encode(
                [row["text"] for row in original_order], batch_size=32,
                convert_to_numpy=True, normalize_embeddings=True,
                show_progress_bar=False,
            )
            doc_vectors[doc_id] = (original_order, vectors)
        windows, vectors = doc_vectors[doc_id]
        q_vector = model.encode(
            [query["query"]], convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )[0]
        dense_ranked = rank_dense(windows, vectors, q_vector)
        rankings = {"bm25": bm25_ranked, "dense": dense_ranked,
                    "hybrid_rrf": rank_hybrid(bm25_ranked, dense_ranked)}
        for name, ranked in rankings.items():
            packet = select_packet(
                query["id"], query["text"], doc_id, ranked, counter,
                context_window=config["context_window"], output_reserve=1024,
                safety_margin=config["safety_margin"],
                max_passages=config["max_passages"],
            )
            if name == "bm25" and packet["packet_hash"] != configuration_hash(
                {"messages": bm25_packet["messages"]}
            ):
                raise ValueError("BM25 diagnostic differs from the prepared packet")
            methods[name].append({"question_id": query["id"],
                                  "selected_passages": [_metadata(p) for p in packet["passages"]],
                                  "prompt_tokens_local": packet["prompt_tokens_local"],
                                  "packet_hash": packet["packet_hash"]})
    retrieval_seconds = time.monotonic() - retrieval_start

    # Gold is accessed only after every ranking and packet has been finalized.
    by_id = {q["id"]: q for q in benchmark["questions"]}
    summaries = {}
    for name, rows in methods.items():
        for row in rows:
            q = by_id[row["question_id"]]
            row["expected_answerability"] = q["expected_answerability"]
            row["span_coverage"] = [
                _covered_fraction(span, row["selected_passages"])
                for span in q["gold_evidence"]
            ]
        summaries[name] = _coverage_summary(rows)
    report = {
        "schema_version": "envoybench-retrieval-diagnostic-v1",
        "status": "complete_offline_no_model_requests",
        "scope": "same 40 selected QASPER known-paper questions; unreviewed annotations",
        "meaning": "annotated evidence-window overlap, not answer correctness or semantic support",
        "predeclared_hypothesis": (
            "Dense or fixed RRF retrieval may recover evidence missed by BM25."
        ),
        "decision_rule": (
            "Report all fixed methods; do not tune on this known set or infer answer quality."
        ),
        "question_count": len(queries), "model_api_requests": 0,
        "settings": {**config, "output_reserve": 1024, "rrf_k": RRF_K,
                     "evidence_overlap_threshold": EVIDENCE_THRESHOLD,
                     "embedding_repository": EMBEDDING_REPOSITORY,
                     "embedding_revision": EMBEDDING_REVISION,
                     "embedding_max_seq_length": model.max_seq_length,
                     "normalization": "unit_l2_cosine", "device": "cpu",
                     "tie_break": "source_start_ascending"},
        "inputs": {"prepared_plan_sha256": _file_sha256(prepared / "plan.json"),
                   "prepared_packets_sha256": _file_sha256(prepared / "packets.json"),
                   "source_manifest_sha256": _file_sha256(source_run / "manifest.json"),
                   "benchmark_hash": configuration_hash(benchmark),
                   "corpus_hash": benchmark["corpus_hash"],
                   "tokenizer": counter.metadata,
                   "embedding_weights_sha256": _file_sha256(snapshot / "model.safetensors"),
                   "script_sha256": _file_sha256(Path(__file__))},
        "runtime": {"python": sys.version.split()[0], "platform": platform.platform(),
                    "numpy": np.__version__, "sentence_transformers":
                    __import__("sentence_transformers").__version__,
                    "embedding_and_tokenizer_load_seconds": round(initialization_seconds, 3),
                    "retrieval_and_budget_seconds": round(retrieval_seconds, 3),
                    "total_seconds": round(time.monotonic() - start_time, 3)},
        "summary": summaries, "questions": methods,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = run(args.prepared, args.source_run, args.dataset, args.output)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Retrieval diagnostic: {exc}\n")
    print(json.dumps({"status": report["status"], "summary": report["summary"],
                      "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
