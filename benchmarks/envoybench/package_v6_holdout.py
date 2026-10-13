"""Package the locked 24-question QASPER v6 holdout for EnvoyBench.

The source benchmark and corpus are frozen before model evaluation. This
command verifies their hashes, adds pinned QASPER annotation metadata, and
copies the corpus bytes unchanged. It runs no model and produces no scores.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from benchmarks.envoybench import build_data
from benchmarks.envoybench.frozen_dataset import load_split
from src.eval.artifacts import content_hash
from src.research.benchmark import validate_benchmark
from src.research.qasper import (
    QASPER_CONFIG,
    QASPER_DATASET,
    QASPER_REVISION,
    _to_code_exec_question,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCK = ROOT / "data/research/qasper-v6-holdout-lock.json"
DEFAULT_SOURCE_SPLITS = ROOT / "benchmarks/envoybench/data/splits.json"
SPLIT = "holdout"
STATUS = "prospective_unreviewed_holdout"


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _source_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _check_hash(path: Path, expected: str) -> None:
    if build_data.sha256_file(path) != expected:
        raise ValueError(f"Frozen file hash differs: {path}")


def _verify_lock(lock: dict) -> None:
    if lock.get("schema_version") != "qasper-v6-holdout-lock-v1":
        raise ValueError("Expected qasper-v6-holdout-lock-v1")
    if lock.get("source_split") != "test" or lock.get("source_revision") != QASPER_REVISION:
        raise ValueError("Holdout source is not the pinned QASPER test split")
    ids = lock.get("question_ids")
    if not isinstance(ids, list) or len(ids) != 24 or ids != sorted(set(ids)):
        raise ValueError("Holdout lock must contain exactly 24 sorted, unique question IDs")
    docs = lock.get("reserved_doc_ids")
    if not isinstance(docs, list) or len(docs) != 23 or docs != sorted(set(docs)):
        raise ValueError("Holdout lock must contain exactly 23 sorted, unique target documents")
    if lock.get("answerability_counts") != {"sufficient": 17, "insufficient": 7}:
        raise ValueError("Holdout answerability counts differ from the frozen 17/7 split")
    if lock.get("overlap_with_prior_benchmarks") != 0:
        raise ValueError("Holdout has previously used benchmark questions")
    if lock.get("overlap_with_reviewed_sft_candidates") != 0:
        raise ValueError("Holdout has reviewed SFT candidates")


def materialize(
    lock_path: Path,
    output: Path,
    *,
    source_splits_path: Path = DEFAULT_SOURCE_SPLITS,
    arrow_dir: Path | None = None,
) -> dict:
    """Create a new immutable dataset directory from the verified lock."""
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    lock = _read_json(lock_path)
    _verify_lock(lock)
    source_benchmark_path = _source_path(lock["benchmark_path"])
    selection_path = _source_path(lock["selection_path"])
    source_corpus = _source_path(lock["corpus_path"])
    _check_hash(source_benchmark_path, lock["benchmark_sha256"])
    _check_hash(selection_path, lock["selection_sha256"])
    if content_hash(source_corpus) != lock["corpus_hash"]:
        raise ValueError("Frozen source corpus hash differs from holdout lock")

    selection = _read_json(selection_path)
    if (
        selection.get("source_split") != "test"
        or selection.get("source_revision") != QASPER_REVISION
        or selection.get("seed") != lock["selection_seed"]
        or selection.get("question_ids") != lock["question_ids"]
        or selection.get("benchmark_sha256") != lock["benchmark_sha256"]
        or selection.get("corpus_hash") != lock["corpus_hash"]
        or selection.get("exclusions") != lock.get("prior_exclusion_benchmarks")
    ):
        raise ValueError("Frozen selection differs from holdout lock")
    for excluded in lock.get("prior_exclusion_benchmarks", []):
        _check_hash(_source_path(excluded["path"]), excluded["sha256"])

    source_benchmark = _read_json(source_benchmark_path)
    validate_benchmark(source_benchmark)
    source_questions = source_benchmark["questions"]
    if (
        [question["id"] for question in source_questions] != lock["question_ids"]
        or source_benchmark["corpus_hash"] != lock["corpus_hash"]
        or len(source_benchmark["reserved_doc_ids"]) != len(list(source_corpus.glob("*.json")))
        or set(source_benchmark["reserved_doc_ids"])
        != {path.stem for path in source_corpus.glob("*.json")}
        or dict(Counter(q["expected_answerability"] for q in source_questions))
        != lock["answerability_counts"]
    ):
        raise ValueError("Frozen benchmark differs from holdout selection or corpus")

    pinned_splits = _read_json(source_splits_path)
    source = pinned_splits["source"]
    if source["dataset"] != QASPER_DATASET or source["revision"] != QASPER_REVISION:
        raise ValueError("EnvoyBench source pin differs from holdout lock")
    rows, source_audit = build_data._load_source(
        "test", source["splits"]["test"], arrow_dir
    )
    catalogue = build_data._question_catalogue(rows, "test")
    if set(lock["question_ids"]) - catalogue.keys():
        raise ValueError("Holdout question absent from pinned QASPER source")
    questions = []
    target_doc_ids = set()
    for source_question in source_questions:
        converted, _ = catalogue[source_question["id"]]
        original = _to_code_exec_question(converted)
        if original != source_question:
            raise ValueError(f"Frozen question differs from pinned QASPER: {source_question['id']}")
        enriched = build_data._prepare_question(converted)
        target_doc_ids.add(converted["target_doc_ids"][0])
        questions.append(enriched)
    if sorted(target_doc_ids) != lock["reserved_doc_ids"]:
        raise ValueError("Frozen target papers differ from holdout lock")
    if target_doc_ids & set(selection.get("excluded_doc_ids", [])):
        raise ValueError("Holdout target paper appears in frozen prior-use exclusions")
    prior_question_ids = {
        question["id"]
        for excluded in lock.get("prior_exclusion_benchmarks", [])
        for question in _read_json(_source_path(excluded["path"]))["questions"]
    }
    prior_target_docs = {
        catalogue[qid][0]["target_doc_ids"][0]
        for qid in prior_question_ids & catalogue.keys()
    }
    if set(lock["question_ids"]) & prior_question_ids or target_doc_ids & prior_target_docs:
        raise ValueError("Holdout questions or target papers overlap a prior benchmark")

    split_dir = output / SPLIT
    corpus_dir = split_dir / "corpus"
    corpus_dir.mkdir(parents=True)
    for source_file in sorted(source_corpus.glob("*.json")):
        shutil.copyfile(source_file, corpus_dir / source_file.name)
    if content_hash(corpus_dir) != lock["corpus_hash"]:
        raise ValueError("Packaged corpus differs from frozen source")

    benchmark = {
        **{key: value for key, value in source_benchmark.items() if key != "questions"},
        "benchmark_id": "envoybench-qasper-v6-holdout-v1",
        "status": STATUS,
        "source_dataset": QASPER_DATASET,
        "source_revision": QASPER_REVISION,
        "source_split": "test",
        "source_benchmark_sha256": lock["benchmark_sha256"],
        "holdout_lock_sha256": build_data.sha256_file(lock_path),
        "questions": questions,
    }
    validate_benchmark(benchmark)
    benchmark_path = split_dir / "benchmark.json"
    build_data._write_json(benchmark_path, benchmark)
    paper_manifest = []
    for path in sorted(corpus_dir.glob("*.json")):
        document = _read_json(path)
        paper_manifest.append({
            "doc_id": document["doc_id"],
            "source_paper_id": document["metadata"]["qasper_paper_id"],
            "sha256": build_data.sha256_file(path),
        })
    benchmark_hash = build_data.sha256_file(benchmark_path)
    snapshot = {
        "schema_version": "research-snapshot-v1",
        "benchmark_id": benchmark["benchmark_id"],
        "source_dataset": QASPER_DATASET,
        "source_config": QASPER_CONFIG,
        "source_revision": QASPER_REVISION,
        **source_audit,
        "split_status": STATUS,
        "human_review_status": "pending",
        "papers": paper_manifest,
        "paper_count": len(paper_manifest),
        "selected_target_paper_ids": sorted({q["source_paper_id"] for q in questions}),
        "selected_question_ids": lock["question_ids"],
        "corpus_hash": lock["corpus_hash"],
        "benchmark_sha256": benchmark_hash,
        "source_benchmark_sha256": lock["benchmark_sha256"],
        "holdout_lock_sha256": build_data.sha256_file(lock_path),
        "status": "complete",
    }
    build_data._write_json(split_dir / "manifest.json", snapshot)
    validate_benchmark(benchmark, snapshot)
    split_spec = {
        "source_split": "test",
        "status": STATUS,
        "selection_seed": lock["selection_seed"],
        "question_ids": lock["question_ids"],
        "reserved_doc_ids": lock["reserved_doc_ids"],
        "answerability_counts": lock["answerability_counts"],
        "lock_sha256": build_data.sha256_file(lock_path),
    }
    build_data._write_json(output / "splits.json", {
        "schema_version": "envoybench-splits-v1",
        "source": source,
        "splits": {SPLIT: split_spec},
    })
    entry = {
        "benchmark": f"{SPLIT}/benchmark.json",
        "corpus": f"{SPLIT}/corpus",
        "manifest": f"{SPLIT}/manifest.json",
        "benchmark_sha256": benchmark_hash,
        "corpus_hash": lock["corpus_hash"],
        "source_rows_sha256": source["splits"]["test"]["rows_sha256"],
        "question_count": len(questions),
        "target_paper_count": len(target_doc_ids),
        "corpus_paper_count": len(paper_manifest),
        "answerability": lock["answerability_counts"],
        "status": STATUS,
        "human_review_status": "pending",
    }
    index = {
        "schema_version": "envoybench-data-manifest-v1",
        "source_dataset": QASPER_DATASET,
        "source_config": QASPER_CONFIG,
        "source_revision": QASPER_REVISION,
        "source_url": f"https://huggingface.co/datasets/{QASPER_DATASET}",
        "source_license": "CC BY 4.0",
        "frozen_splits_sha256": build_data.sha256_file(output / "splits.json"),
        "holdout_lock_sha256": build_data.sha256_file(lock_path),
        "independent_human_review": "pending; no benchmark score is claimed",
        "splits": {SPLIT: entry},
    }
    build_data._write_json(output / "manifest.json", index)
    loaded, _, _, loaded_corpus, _ = load_split(output, SPLIT)
    if [q["id"] for q in loaded["questions"]] != lock["question_ids"] or content_hash(
        loaded_corpus
    ) != lock["corpus_hash"]:
        raise ValueError("Packaged split does not reproduce the frozen holdout")
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--source-splits", type=Path, default=DEFAULT_SOURCE_SPLITS)
    parser.add_argument("--arrow-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    index = materialize(
        args.lock, args.output,
        source_splits_path=args.source_splits,
        arrow_dir=args.arrow_dir,
    )
    print(json.dumps({"output": str(args.output), "holdout": index["splits"][SPLIT]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
