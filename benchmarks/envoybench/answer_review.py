"""Human-operated, identity-blind answer review for EnvoyBench Studio.

This editor opens only the prepared ``review.json`` and frozen benchmark/source
corpus. It never opens the blind key or automatic scores. The blind key's
per-row content hashes are checked by ``score_completed_review`` only after
the human reviewer has locked all verdicts.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import tempfile
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Iterator

from benchmarks.envoybench.score import REVIEW_VERSION, RUBRIC_VERSION, _review_content
from src.eval.artifacts import configuration_hash, content_hash
from src.eval.research_review import load_corpus
from src.research.benchmark import validate_benchmark

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ROW_FIELDS = {
    "blind_id", "question_id", "question", "expected_answerability",
    "reference_answer", "grader_notes", "status", "answer", "citations",
    "evidence", "verdict", "notes", "relevant_source_passage",
}
_REVIEW_FIELDS = {
    "schema_version", "benchmark_id", "benchmark_hash", "corpus_hash",
    "results_hash", "reference_review_hash", "status", "reviewer_kind",
    "reviewer_id", "reviewed_at", "rubric_version", "adjudications",
    "excluded_questions", "instructions", "rows",
}
_VERDICTS = {"pass", "partial", "fail"}


def _validate_frozen_inputs(benchmark: dict, corpus_dir: Path) -> None:
    validate_benchmark(benchmark)
    if not corpus_dir.is_dir():
        raise ValueError("frozen corpus directory does not exist")
    if content_hash(corpus_dir) != benchmark["corpus_hash"]:
        raise ValueError("benchmark corpus hash differs from frozen corpus bytes")
    documents = load_corpus(corpus_dir)
    if set(documents) != set(benchmark["reserved_doc_ids"]):
        raise ValueError("frozen corpus document IDs differ from benchmark")


def _validate_review(review: dict, benchmark: dict) -> None:
    if not isinstance(review, dict) or set(review) != _REVIEW_FIELDS:
        raise ValueError("blind answer review schema fields are invalid")
    if review["schema_version"] != REVIEW_VERSION:
        raise ValueError("unsupported blind answer review schema")
    for field, expected in (
        ("benchmark_id", benchmark["benchmark_id"]),
        ("benchmark_hash", configuration_hash(benchmark)),
        ("corpus_hash", benchmark["corpus_hash"]),
    ):
        if review[field] != expected:
            raise ValueError(f"blind answer review {field} mismatch")
    for field in ("results_hash", "reference_review_hash"):
        if not isinstance(review[field], str) or not _SHA256.fullmatch(review[field]):
            raise ValueError(f"blind answer review {field} is invalid")
    if review["rubric_version"] != RUBRIC_VERSION:
        raise ValueError("blind answer review rubric version is unknown")
    if not isinstance(review["instructions"], dict):
        raise ValueError("blind answer review instructions must be an object")
    if not isinstance(review["adjudications"], list):
        raise ValueError("blind answer review adjudications must be a list")
    if not isinstance(review["reviewer_id"], str):
        raise ValueError("blind answer review reviewer_id must be a string")

    questions = {question["id"]: question for question in benchmark["questions"]}
    excluded = review["excluded_questions"]
    if not isinstance(excluded, list):
        raise ValueError("blind answer review exclusions must be a list")
    excluded_ids: set[str] = set()
    for item in excluded:
        if not isinstance(item, dict) or set(item) != {"question_id", "reason"}:
            raise ValueError("blind answer review exclusion is invalid")
        question_id = item["question_id"]
        if (not isinstance(question_id, str) or question_id not in questions
                or question_id in excluded_ids
                or not isinstance(item["reason"], str) or not item["reason"].strip()):
            raise ValueError("blind answer review exclusion is invalid")
        excluded_ids.add(question_id)

    rows = review["rows"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("blind answer review needs rows")
    blind_ids: set[str] = set()
    per_question = {question_id: 0 for question_id in questions if question_id not in excluded_ids}
    for row in rows:
        if not isinstance(row, dict) or set(row) != _ROW_FIELDS:
            raise ValueError("blind answer review row schema is invalid")
        blind_id, question_id = row["blind_id"], row["question_id"]
        if not isinstance(blind_id, str) or not blind_id or blind_id in blind_ids:
            raise ValueError("blind answer review has duplicate or invalid blind IDs")
        blind_ids.add(blind_id)
        if not isinstance(question_id, str) or question_id not in per_question:
            raise ValueError(f"{blind_id} has an excluded or unknown question ID")
        per_question[question_id] += 1
        question = questions[question_id]
        for field, expected in (
            ("question", question["question"]),
            ("expected_answerability", question["expected_answerability"]),
            ("reference_answer", question.get("answer")),
            ("grader_notes", question.get("grader_notes", [])),
        ):
            if row[field] != expected:
                raise ValueError(f"{blind_id} {field} differs from frozen benchmark")
        if (not isinstance(row["status"], str)
                or row["status"] not in {"submitted", "no_submission", "error", "escalated"}):
            raise ValueError(f"{blind_id} has an invalid run status")
        if (not isinstance(row["answer"], str)
                or not isinstance(row["citations"], list)
                or not isinstance(row["evidence"], list)):
            raise ValueError(f"{blind_id} has invalid answer or evidence fields")
        if row["verdict"] is not None and (
            not isinstance(row["verdict"], str) or row["verdict"] not in _VERDICTS
        ):
            raise ValueError(f"{blind_id} has an invalid verdict")
        if (not isinstance(row["notes"], str)
                or not isinstance(row["relevant_source_passage"], str)):
            raise ValueError(f"{blind_id} needs string review notes and passage")
        if row["verdict"] in {"partial", "fail"} and (
            not row["notes"].strip() or not row["relevant_source_passage"].strip()
        ):
            raise ValueError(f"{blind_id} needs a reason and relevant source passage")
    counts = set(per_question.values())
    if len(counts) != 1 or next(iter(counts), 0) < 2:
        raise ValueError("blind answer review is not a complete paired question matrix")

    decided = sum(row["verdict"] is not None for row in rows)
    if decided and not review["reviewer_id"].strip():
        raise ValueError("decided answers need a reviewer_id")
    if review["status"] == "incomplete":
        if review["reviewer_kind"] is not None or review["reviewed_at"] != "":
            raise ValueError("incomplete review cannot claim human completion")
    elif review["status"] == "complete":
        if decided != len(rows) or review["reviewer_kind"] != "human":
            raise ValueError("completed human review needs every verdict")
        try:
            date.fromisoformat(review["reviewed_at"])
        except (TypeError, ValueError) as exc:
            raise ValueError("completed review needs an ISO reviewed_at date") from exc
    else:
        raise ValueError("blind answer review status must be incomplete or complete")


def _read_review(path: Path, benchmark: dict) -> tuple[dict, str]:
    # A review path that is a symlink to blind-key.json must never be followed.
    if path.name != "review.json" or path.is_symlink():
        raise ValueError("answer review path must be a regular review.json file")
    try:
        raw = path.read_bytes()
    except FileNotFoundError as exc:
        raise ValueError("prepared blind review.json does not exist") from exc
    review = json.loads(raw)
    _validate_review(review, benchmark)
    return review, hashlib.sha256(raw).hexdigest()


def _payload(review: dict, revision: str) -> dict:
    return {
        **review,
        "revision": revision,
        "decided_count": sum(row["verdict"] is not None for row in review["rows"]),
        "total_count": len(review["rows"]),
    }


def load_answer_review(path: Path, benchmark: dict, corpus_dir: Path) -> dict:
    """Load a prepared blind review without reading the identity key or run results."""
    _validate_frozen_inputs(benchmark, corpus_dir)
    review, revision = _read_review(path, benchmark)
    return _payload(review, revision)


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    lock_path = path.with_name(f".{path.name}.lock")
    with lock_path.open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _write_atomic(path: Path, review: dict) -> str:
    data = (json.dumps(review, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = stream.name
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
    return hashlib.sha256(data).hexdigest()


def _check_revision(actual: str, expected: str | None) -> None:
    if not isinstance(expected, str) or not _SHA256.fullmatch(expected):
        raise ValueError("expected_sha256 revision is required")
    if actual != expected:
        raise ValueError("blind answer review changed since it was loaded")


def _check_reviewer(review: dict, reviewer_id: str) -> str:
    if not isinstance(reviewer_id, str) or not reviewer_id.strip():
        raise ValueError("reviewer_id must name the human reviewer")
    reviewer = reviewer_id.strip()
    existing = review["reviewer_id"].strip()
    if existing and existing != reviewer:
        raise ValueError("reviewer_id differs from the existing reviewer")
    return reviewer


def save_answer_verdict(
    path: Path,
    benchmark: dict,
    corpus_dir: Path,
    blind_id: str,
    verdict: str,
    notes: str,
    relevant_passage: str,
    reviewer_id: str,
    *,
    expected_sha256: str | None = None,
) -> dict:
    """Record one human verdict while preserving all frozen anonymous answers."""
    if verdict not in _VERDICTS:
        raise ValueError("verdict must be pass, partial, or fail")
    if not isinstance(notes, str) or not isinstance(relevant_passage, str):
        raise ValueError("notes and relevant_passage must be strings")
    if verdict in {"partial", "fail"} and (
        not notes.strip() or not relevant_passage.strip()
    ):
        raise ValueError("partial/fail needs a reason and relevant source passage")
    _validate_frozen_inputs(benchmark, corpus_dir)
    with _locked(path):
        review, revision = _read_review(path, benchmark)
        _check_revision(revision, expected_sha256)
        if review["status"] != "incomplete":
            raise ValueError("completed blind answer review is locked")
        reviewer = _check_reviewer(review, reviewer_id)
        row = next((item for item in review["rows"] if item["blind_id"] == blind_id), None)
        if row is None:
            raise ValueError("unknown blind_id")
        frozen_hashes = {
            item["blind_id"]: configuration_hash(_review_content(item))
            for item in review["rows"]
        }
        row["verdict"] = verdict
        row["notes"] = notes.strip()
        row["relevant_source_passage"] = relevant_passage.strip()
        review["reviewer_id"] = reviewer
        if any(configuration_hash(_review_content(item)) != frozen_hashes[item["blind_id"]]
               for item in review["rows"]):
            raise ValueError("immutable blind answer content changed during save")
        _validate_review(review, benchmark)
        return _payload(review, _write_atomic(path, review))


def finalize_answer_review(
    path: Path,
    benchmark: dict,
    corpus_dir: Path,
    reviewer_id: str,
    *,
    expected_sha256: str | None = None,
) -> dict:
    """Lock a fully judged, interactive human review for later keyed scoring."""
    _validate_frozen_inputs(benchmark, corpus_dir)
    with _locked(path):
        review, revision = _read_review(path, benchmark)
        _check_revision(revision, expected_sha256)
        if review["status"] != "incomplete":
            raise ValueError("completed blind answer review is locked")
        reviewer = _check_reviewer(review, reviewer_id)
        if any(row["verdict"] not in _VERDICTS for row in review["rows"]):
            raise ValueError("every blind answer needs a verdict before finalization")
        frozen_hashes = {
            item["blind_id"]: configuration_hash(_review_content(item))
            for item in review["rows"]
        }
        review["status"] = "complete"
        review["reviewer_kind"] = "human"
        review["reviewer_id"] = reviewer
        review["reviewed_at"] = date.today().isoformat()
        if any(configuration_hash(_review_content(item)) != frozen_hashes[item["blind_id"]]
               for item in review["rows"]):
            raise ValueError("immutable blind answer content changed during finalization")
        _validate_review(review, benchmark)
        return _payload(review, _write_atomic(path, review))
