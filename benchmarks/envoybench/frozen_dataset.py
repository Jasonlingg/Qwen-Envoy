"""Frozen EnvoyBench dataset validation shared by inference, scoring, and Studio."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.eval.artifacts import content_hash
from src.research.benchmark import load_benchmark

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "benchmarks/envoybench/data"


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"missing or invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dataset_path(dataset: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("dataset split entry is missing a path")
    base = dataset.resolve(strict=True)
    path = (base / relative).resolve()
    if not path.is_relative_to(base):
        raise ValueError(f"dataset path escapes the frozen dataset: {relative}")
    if not path.exists():
        raise ValueError(
            f"frozen dataset file missing: {path}; run "
            "python -m benchmarks.envoybench.build_data --output benchmarks/envoybench/data"
        )
    return path


def load_split(dataset: Path, split: str) -> tuple[dict, dict, Path, Path, dict]:
    """Load and hash-check a materialized split before any model request."""
    index = _read_json(dataset / "manifest.json")
    if index.get("schema_version") != "envoybench-data-manifest-v1":
        raise ValueError("unsupported frozen dataset manifest")
    splits_path = _dataset_path(dataset, "splits.json")
    if _file_sha256(splits_path) != index.get("frozen_splits_sha256"):
        raise ValueError("frozen split specification hash differs from dataset manifest")
    split_spec = (_read_json(splits_path).get("splits") or {}).get(split)
    if not isinstance(split_spec, dict):
        raise ValueError(f"{split}: absent from frozen split specification")
    entry = (index.get("splits") or {}).get(split)
    if not isinstance(entry, dict):
        available = sorted(index.get("splits") or {})
        raise ValueError(f"unknown split {split!r}; available: {available}")
    benchmark_path = _dataset_path(dataset, entry.get("benchmark"))
    corpus_path = _dataset_path(dataset, entry.get("corpus"))
    snapshot_path = _dataset_path(dataset, entry.get("manifest"))
    if not corpus_path.is_dir():
        raise ValueError(f"corpus is not a directory: {corpus_path}")
    snapshot = _read_json(snapshot_path)
    benchmark = load_benchmark(benchmark_path, snapshot)
    benchmark_file_hash = _file_sha256(benchmark_path)
    if benchmark_file_hash != entry.get("benchmark_sha256") or benchmark_file_hash != snapshot.get(
        "benchmark_sha256"
    ):
        raise ValueError("generated benchmark bytes differ from frozen manifests")
    selected_ids = [question["id"] for question in benchmark["questions"]]
    if (
        selected_ids != split_spec.get("question_ids")
        or selected_ids != snapshot.get("selected_question_ids")
        or len(selected_ids) != entry.get("question_count")
    ):
        raise ValueError("generated benchmark question IDs differ from frozen selection")
    target_papers = sorted({question.get("source_paper_id") for question in benchmark["questions"]})
    if target_papers != snapshot.get("selected_target_paper_ids"):
        raise ValueError("generated benchmark target papers differ from frozen selection")
    if snapshot.get("split_status") != entry.get("status"):
        raise ValueError("snapshot split status differs from dataset manifest")
    actual_hash = content_hash(corpus_path)
    if (
        actual_hash != benchmark["corpus_hash"]
        or actual_hash != snapshot.get("corpus_hash")
        or actual_hash != entry.get("corpus_hash")
    ):
        raise ValueError("frozen corpus contents differ from benchmark corpus_hash")
    return benchmark, snapshot, benchmark_path, corpus_path, entry
