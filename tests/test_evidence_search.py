import pytest

from src.product.evidence_search import find_evidence
from tests.arxiv_fakes import article_page, make_library


def _library(tmp_path):
    library, _ = make_library(tmp_path, {"2609.12345v2": {"html": article_page()},
                                         "2609.54321v1": {}})
    library.open("2609.12345v2")
    library.open("2609.54321v1")
    return library


def test_find_evidence_ranks_across_papers_with_exact_offsets(tmp_path):
    library = _library(tmp_path)
    result = find_evidence(library, "eviction memory", ["2609.12345v2", "2609.54321v1"], top_k=3)
    assert 0 < len(result["evidence"]) <= 3
    top = result["evidence"][0]
    assert top["arxiv_id"] == "2609.12345v2"
    assert library.document(top["arxiv_id"])["text"][top["start"]:top["end"]] == top["quote"]
    scores = [hit["score"] for hit in result["evidence"]]
    assert scores == sorted(scores, reverse=True)


def test_find_evidence_validates_input(tmp_path):
    library = _library(tmp_path)
    for bad in [{"query": ""}, {"arxiv_ids": []}, {"top_k": 0}, {"top_k": 21},
                {"arxiv_ids": ["2609.12345v2"] * 11}]:
        args = {"query": "eviction", "arxiv_ids": ["2609.12345v2"], "top_k": 8, **bad}
        with pytest.raises(ValueError):
            find_evidence(library, **args)
    with pytest.raises(ValueError, match="not open"):
        find_evidence(library, "eviction", ["2609.99999v1"])
