"""Source-only, human-operated reference review for EnvoyBench Studio.

This module never infers a decision from model output or from an automated
screen. Completion is a separate reviewer action after every question has an
explicit decision. The scorer remains the authority on completed reviews.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Iterator

from benchmarks.envoybench.score import _validated_references, reference_review_template
from src.eval.artifacts import content_hash
from src.research.benchmark import validate_benchmark


def _validate_corpus(benchmark: dict, corpus_dir: Path) -> None:
    validate_benchmark(benchmark)
    if not corpus_dir.is_dir():
        raise ValueError("frozen corpus directory does not exist")
    if content_hash(corpus_dir) != benchmark["corpus_hash"]:
        raise ValueError("benchmark corpus hash differs from frozen corpus bytes")


def _read_sheet(path: Path, benchmark: dict) -> tuple[dict, str]:
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        sheet = reference_review_template(benchmark)
        return sheet, hashlib.sha256(_serialize(sheet)).hexdigest()
    sheet = json.loads(raw)
    if not isinstance(sheet, dict):
        raise ValueError("reference review must be a JSON object")
    template = reference_review_template(benchmark)
    if set(sheet) != set(template):
        raise ValueError("reference review schema fields differ from template")
    for field in ("schema_version", "benchmark_id", "benchmark_hash", "corpus_hash",
                  "instructions"):
        if sheet[field] != template[field]:
            raise ValueError(f"reference review {field} mismatch")
    items = sheet["questions"]
    expected = template["questions"]
    if not isinstance(items, list) or len(items) != len(expected):
        raise ValueError("reference review questions differ from frozen benchmark")
    for item, source in zip(items, expected, strict=True):
        if not isinstance(item, dict) or set(item) != set(source):
            raise ValueError("reference review question schema differs from template")
        for field, value in source.items():
            if field not in {"decision", "reason"} and item[field] != value:
                raise ValueError(f"reference review {field} differs from frozen benchmark")
        decision, reason = item["decision"], item["reason"]
        if decision not in (None, "reference_valid", "unscorable"):
            raise ValueError(f"{item['question_id']} has an invalid decision")
        if not isinstance(reason, str):
            raise ValueError(f"{item['question_id']} needs a string reason")
        if decision == "unscorable" and not reason.strip():
            raise ValueError(f"{item['question_id']} needs an unscorable reason")
    reviewer_id = sheet["reviewer_id"]
    if not isinstance(reviewer_id, str):
        raise ValueError("reference review needs a string reviewer_id")
    if any(item["decision"] is not None for item in items) and not reviewer_id.strip():
        raise ValueError("decided references need a reviewer_id")
    if sheet["status"] == "complete":
        _validated_references(benchmark, sheet)
    elif sheet["status"] == "incomplete":
        if sheet["reviewed_at"] != "":
            raise ValueError("incomplete review cannot have reviewed_at")
    else:
        raise ValueError("reference review status must be incomplete or complete")
    return sheet, hashlib.sha256(raw).hexdigest()


def _payload(sheet: dict, sha256: str) -> dict:
    return {
        "review": sheet,
        "sha256": sha256,
        "decided_count": sum(item["decision"] is not None for item in sheet["questions"]),
        "total_count": len(sheet["questions"]),
    }


def _serialize(sheet: dict) -> bytes:
    return (json.dumps(sheet, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def load_reference_review(path: Path, benchmark: dict, corpus_dir: Path) -> dict:
    """Read a source-only review after validating its benchmark and corpus binding."""
    _validate_corpus(benchmark, corpus_dir)
    sheet, sha256 = _read_sheet(path, benchmark)
    return _payload(sheet, sha256)


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    lock_path = path.with_name(f".{path.name}.lock")
    with lock_path.open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _write_atomic(path: Path, sheet: dict) -> str:
    data = _serialize(sheet)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
            delete=False,
        ) as stream:
            temp_name = stream.name
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if temp_name is not None and os.path.exists(temp_name):
            os.unlink(temp_name)
    return hashlib.sha256(data).hexdigest()


def _check_revision(actual: str, expected: str | None) -> None:
    if expected is not None and actual != expected:
        raise ValueError("reference review changed since it was loaded")


def save_reference_decision(
    path: Path,
    benchmark: dict,
    corpus_dir: Path,
    question_id: str,
    decision: str,
    reason: str,
    reviewer_id: str,
    *,
    expected_sha256: str | None = None,
) -> dict:
    """Record one explicit human review decision; never mark the sheet complete."""
    if decision not in {"reference_valid", "unscorable"}:
        raise ValueError("decision must be reference_valid or unscorable")
    if not isinstance(reason, str):
        raise ValueError("reason must be a string")
    if decision == "unscorable" and not reason.strip():
        raise ValueError("unscorable decisions need a reason")
    if not isinstance(reviewer_id, str) or not reviewer_id.strip():
        raise ValueError("reviewer_id must name the human reviewer")
    _validate_corpus(benchmark, corpus_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _locked(path):
        sheet, sha256 = _read_sheet(path, benchmark)
        _check_revision(sha256, expected_sha256)
        if sheet["status"] != "incomplete":
            raise ValueError("completed reference review is locked")
        current_reviewer = sheet["reviewer_id"].strip()
        if current_reviewer and current_reviewer != reviewer_id.strip():
            raise ValueError("reviewer_id differs from the existing reviewer")
        row = next(
            (item for item in sheet["questions"] if item["question_id"] == question_id),
            None,
        )
        if row is None:
            raise ValueError("unknown benchmark question_id")
        row["decision"] = decision
        row["reason"] = reason.strip()
        sheet["reviewer_id"] = reviewer_id.strip()
        return _payload(sheet, _write_atomic(path, sheet))


def finalize_reference_review(
    path: Path,
    benchmark: dict,
    corpus_dir: Path,
    reviewer_id: str,
    *,
    expected_sha256: str | None = None,
) -> dict:
    """Deliberately lock a fully decided source review for the scorer."""
    if not isinstance(reviewer_id, str) or not reviewer_id.strip():
        raise ValueError("reviewer_id must name the human reviewer")
    _validate_corpus(benchmark, corpus_dir)
    with _locked(path):
        sheet, sha256 = _read_sheet(path, benchmark)
        _check_revision(sha256, expected_sha256)
        if sheet["status"] != "incomplete":
            raise ValueError("completed reference review is locked")
        if sheet["reviewer_id"].strip() != reviewer_id.strip():
            raise ValueError("reviewer_id differs from the existing reviewer")
        sheet["status"] = "complete"
        sheet["reviewed_at"] = date.today().isoformat()
        _validated_references(benchmark, sheet)
        return _payload(sheet, _write_atomic(path, sheet))
