"""Validate the supported modern arXiv identifier syntax before URL/path use."""

import re

MODERN_ARXIV_ID = re.compile(r"[0-9]{2}(?:0[1-9]|1[0-2])\.[0-9]{4,5}(?:v[1-9][0-9]*)?")


def validate_arxiv_id(value: str) -> str:
    """Return a modern ID unchanged; legacy category IDs and URLs are unsupported."""
    if not isinstance(value, str) or not MODERN_ARXIV_ID.fullmatch(value):
        raise ValueError("Supply a modern arXiv ID such as 2503.09516 or 2503.09516v1")
    return value
