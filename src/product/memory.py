"""Shared frozen-vault import and record metadata for the Qwen investigator.

The historical personal-memory application is archived in Git; current callers
retain these small source helpers without its capture/approval product layer.
"""

import re
from pathlib import Path

import yaml

from src.research.vault import import_vault

REVIEW_VERSION = "personal-memory-review-v1"


def freeze_vault(vault: Path, output: Path, collection: str = ".") -> dict:
    """Reuse the opt-in importer; each call makes a new immutable snapshot."""
    return import_vault(vault, collection, output)


def _record_context(document: dict) -> dict:
    """Surface dated memory fields without changing the existing importer schema."""
    match = re.match(r"\A---\n(.*?)\n---(?:\n|$)", document["text"], re.S)
    if match is None:
        return {}
    try:
        frontmatter = yaml.safe_load(match[1]) or {}
    except yaml.YAMLError:
        return {}
    if not isinstance(frontmatter, dict):
        return {}
    fields = (
        "record_id", "memory_id", "kind", "captured_at", "effective_date",
        "status", "review_status", "supersedes", "supersedes_doc_id", "derived_from",
    )
    return {field: str(frontmatter[field]) for field in fields
            if field in frontmatter and frontmatter[field] is not None}
