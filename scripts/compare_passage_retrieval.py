"""Freeze/run a CPU-only retrieval comparison; never dispatch Qwen or change defaults."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.envoybench.frozen_dataset import DEFAULT_DATASET, load_split
from src.eval.passage_retrieval_comparison import coverage, decide, summarize
from src.eval.qasper_passage_smoke import retrieve as legacy_retrieve
from src.research.hybrid_passages import (
    ARMS,
    CONFIG,
    EMBEDDING_MODEL,
    EMBEDDING_REVISION,
    LocalModels,
    build_index,
    retrieve,
)
from src.research.reranker import MS_MARCO_MINILM_MODEL, MS_MARCO_MINILM_REVISION

FILES = ["scripts/compare_passage_retrieval.py", "src/research/hybrid_passages.py",
    "src/eval/passage_retrieval_comparison.py", "src/eval/qasper_passage_smoke.py",
    "src/research/reranker.py", "benchmarks/envoybench/frozen_dataset.py"]
SPLITS = ("dev", "test_candidate")
MODELS = {"embedding": {"repository": EMBEDDING_MODEL, "revision": EMBEDDING_REVISION},
          "reranker": {"repository": MS_MARCO_MINILM_MODEL,
                       "revision": MS_MARCO_MINILM_REVISION}}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False) + "\n")
    temporary.replace(path)


def implementation():
    return {name: sha(ROOT / name) for name in FILES}


def model_files():
    from huggingface_hub import snapshot_download

    result = {}
    for name, model in MODELS.items():
        snapshot = Path(snapshot_download(repo_id=model["repository"],
            revision=model["revision"], local_files_only=True, token=False))
        files = {str(f.relative_to(snapshot)): sha(f) for f in snapshot.rglob("*")
                 if f.is_file()}
        required = ({"modules.json", "model.safetensors", "tokenizer.json"}
                    if name == "embedding" else {"pytorch_model.bin", "config.json", "vocab.txt"})
        if not required <= files.keys():
            raise ValueError("pinned model files are incomplete")
        result[name] = files
    return result


def dataset_identity():
    result = {}
    for split in SPLITS:
        questions, manifest, path, _, _ = load_split(DEFAULT_DATASET, split)
        result[split] = {"benchmark_sha256": sha(path), "corpus_hash": manifest["corpus_hash"],
            "source_revision": manifest["source_revision"],
            "question_ids": [q["id"] for q in questions["questions"]]}
    return result


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    plan = {"schema_version": "two-passage-retrieval-comparison-v1", "arms": ARMS,
        "config": CONFIG, "models": MODELS, "model_file_sha256": model_files(),
        "implementation_sha256": implementation(), "datasets": dataset_identity(),
        "seed": 42, "cpu_threads": 4, "max_paid_requests": 0, "gpu_launches": 0,
        "gold_usage": "scoring only; never retrieval input",
        "decision_rule": {"candidate": "hybrid_rerank", "dev_full_paragraph_gain": 2,
            "dev_any_overlap_loss_allowed": 0, "secondary_full_paragraph_loss_allowed": 0,
            "warm_cpu_query_p95_seconds_max": 2, "candidate_selection_failures_allowed": 0},
        "data_status": "both splits already exposed; not held out",
        "production_default_changed": False}
    save(output / "plan.json", plan)
    return plan


def check(output):
    plan = json.loads((output / "plan.json").read_text())
    if (plan["implementation_sha256"] != implementation()
            or plan["model_file_sha256"] != model_files()
            or plan["datasets"] != dataset_identity()
            or plan["config"] != CONFIG or plan["models"] != MODELS):
        raise ValueError("prepared implementation, models or dataset changed")
    return plan


def evaluate_arm(text, query, arm, models, index=None, index_error=None):
    started = time.perf_counter()
    try:
        if index_error:
            if arm != "bm25":
                raise ValueError(index_error)
            windows, ranking = legacy_retrieve(text, query)
            return {"method": arm, "status": "selected", "error": None,
                "windows": windows, "ranking": ranking, "candidate_pool": [],
                "query_seconds": time.perf_counter() - started}
        return retrieve(index, query, arm, models)
    except (ValueError, RuntimeError) as exc:
        return {"method": arm, "status": "error", "error": f"{type(exc).__name__}: {exc}",
            "windows": [], "ranking": [], "candidate_pool": [],
            "query_seconds": time.perf_counter() - started}


def run(output):
    plan = check(output)
    with (output / "RUN_STARTED").open("x"):
        pass
    import numpy as np
    import torch

    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    models = LocalModels()
    load_seconds = time.perf_counter() - started
    environment = {"python": sys.version, "platform": platform.platform(),
        "machine": platform.machine(), "processor": platform.processor(),
        "torch_threads": torch.get_num_threads(), "seed": 42, "device": "cpu",
        "model_load_seconds": load_seconds,
        "packages": {name: importlib.metadata.version(name) for name in (
            "torch", "sentence-transformers", "transformers", "huggingface-hub",
            "rank-bm25", "numpy")},
        "implementation_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                        cwd=ROOT, text=True).strip(),
        "paid_requests": 0, "qwen_requests": 0, "gpu_launches": 0}
    save(output / "environment.json", environment)
    reports, assignments = {}, []
    cache = {}
    for split in SPLITS:
        benchmark, _, _, corpus, _ = load_split(DEFAULT_DATASET, split)
        rows, indexing = [], []
        for ordinal, question in enumerate(benchmark["questions"], 1):
            doc_id = "qasper_" + re.sub(r"[^A-Za-z0-9_-]+", "_", question["source_paper_id"])
            document = json.loads((corpus / (doc_id + ".json")).read_text())
            if document["metadata"]["qasper_paper_id"] != question["source_paper_id"]:
                raise ValueError("source paper identity differs")
            text, query = document["text"], question["source_question"]
            digest = hashlib.sha256(text.encode()).hexdigest()
            index_error = None
            reused = digest in cache
            try:
                if not reused:
                    cache[digest] = build_index(text, models)
                index = cache[digest]
                indexing.append({"question_id": question["id"], "source_sha256": digest,
                    "reused": reused, "windows": len(index.windows), **index.timing,
                    "truncated_dense_windows": index.truncated_dense_windows})
            except (ValueError, RuntimeError) as exc:
                index_error = str(exc)
                indexing.append({"question_id": question["id"], "error": index_error})
            # The backend receives source text and query only. Scoring labels stay here.
            for arm in ARMS:
                result = evaluate_arm(text, query, arm, models,
                    index=None if index_error else index, index_error=index_error)
                if arm == "bm25":
                    try:
                        old_windows, old_ranking = legacy_retrieve(text, query)
                        assert old_windows == result["windows"]
                        assert old_ranking == result["ranking"]
                    except ValueError:
                        if result["status"] != "error":
                            raise ValueError("legacy BM25 compatibility differs") from None
                row = {"question_id": question["id"], "doc_id": doc_id,
                    "source_sha256": digest,
                    "question": query,
                    "reference_answerability": question["expected_answerability"],
                    **result, "coverage": coverage(result["windows"], question["gold_evidence"],
                                                   doc_id, len(text)),
                    "candidate_pool_coverage": coverage(result["candidate_pool"],
                        question["gold_evidence"], doc_id, len(text))}
                rows.append(row)
                assignments.append({"split": split, "question_id": question["id"],
                    "doc_id": doc_id, "question": query, "method": arm,
                    "source_sha256": digest, "status": result["status"],
                    "windows": result["windows"], "counts_as_source_inspection": False})
            if ordinal % 10 == 0:
                print(json.dumps({"split": split, "questions_completed": ordinal}), flush=True)
        report = {"split": split, "data_status": plan["data_status"],
            "methods": summarize(rows), "indexing": indexing, "questions": rows,
            "semantic_support": "not_measured", "human_review": None}
        reports[split] = report
        save(output / (split + ".json"), report)
    decision = decide(reports)
    save(output / "decision.json", decision)
    save(output / "reader-assignments.json", assignments)
    print(json.dumps(decision), flush=True)
    return reports, decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      TOKENIZERS_PARALLELISM="false")
    if args.phase == "prepare":
        prepare(args.output)
        print("Frozen local-only plan; no retrieval scores computed")
    else:
        run(args.output)


if __name__ == "__main__":
    main()
