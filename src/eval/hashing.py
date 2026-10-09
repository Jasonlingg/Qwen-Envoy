"""Shared content-hashing helpers for QASPER data-pipeline scripts.

Before this module existed, `sha256(path)` and `known_doc_id(question)` were
each copy-pasted into several scripts independently. They're used for dedup
and provenance gating (e.g. deciding whether two files are "the same data"),
so a silent drift between copies — one gets fixed, the others don't — would
produce disagreeing hashes/IDs with no error, exactly the failure mode this
project already hit once with the training/inference tokenization mismatch.
One canonical implementation, imported everywhere, removes that risk.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

_DOC_ID_IN_QUESTION = re.compile(r'doc_id:\s*"([^"\s]+)"')


def sha256(path: Path) -> str:
    """Hex-digest a file's contents, streaming so large files don't load fully into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def known_doc_id(question: str) -> str:
    """Extract the doc_id a "known-paper" QASPER question was templated against."""
    match = _DOC_ID_IN_QUESTION.search(question)
    if match is None:
        raise ValueError("Known-paper question has no doc_id")
    return match.group(1)
