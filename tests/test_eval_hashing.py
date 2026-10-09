from pathlib import Path

import pytest

from src.eval.hashing import known_doc_id, sha256


def test_sha256_matches_hashlib_for_small_file(tmp_path: Path):
    import hashlib

    path = tmp_path / "small.bin"
    path.write_bytes(b"hello world")
    assert sha256(path) == hashlib.sha256(b"hello world").hexdigest()


def test_sha256_matches_hashlib_across_a_chunk_boundary(tmp_path: Path):
    """Regression: the streaming implementation reads in 1MB chunks -- make sure
    a file that spans multiple chunks still hashes identically to a one-shot read."""
    import hashlib

    path = tmp_path / "large.bin"
    payload = b"x" * (1024 * 1024 + 1)
    path.write_bytes(payload)
    assert sha256(path) == hashlib.sha256(payload).hexdigest()


def test_known_doc_id_extracts_quoted_id():
    question = 'Known paper doc_id: "qasper_123_456" ... what is the F1 score?'
    assert known_doc_id(question) == "qasper_123_456"


def test_known_doc_id_raises_when_absent():
    with pytest.raises(ValueError):
        known_doc_id("What is the F1 score reported in the paper?")
