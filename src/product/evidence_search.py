"""Model-free passage search across several opened papers, with exact offsets."""

from __future__ import annotations

from src.research.tools_runtime import ResearchTools

MAX_PAPERS = 10
MAX_TOP_K = 20


def find_evidence(library, query: str, arxiv_ids: list[str], top_k: int = 8) -> dict:
    if not isinstance(query, str) or not query.strip() or len(query) > 4_000:
        raise ValueError("query must contain 1..4000 characters")
    if not isinstance(arxiv_ids, list) or not 1 <= len(arxiv_ids) <= MAX_PAPERS:
        raise ValueError(f"arxiv_ids must list 1..{MAX_PAPERS} opened papers")
    if type(top_k) is not int or not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k must be 1..{MAX_TOP_K}")
    tools = ResearchTools(library.view(arxiv_ids) / "corpus")
    hits, seen = [], set()
    for doc_id, document in tools.documents.items():
        for hit in tools.search_paper(doc_id, query, top_k=min(top_k, 10)):
            key = (doc_id, hit["start"], hit["end"])
            if key not in seen:
                seen.add(key)
                hits.append({**hit, "arxiv_id": document["metadata"]["arxiv_id"]})
    hits.sort(key=lambda hit: (-hit["score"], hit["arxiv_id"], hit["start"]))
    return {
        "query": query,
        "papers": sorted(doc["metadata"]["arxiv_id"] for doc in tools.documents.values()),
        "evidence": hits[:top_k],
        "retriever": "lexical_paragraph",
        "warning": "Lexical relevance and exact excerpts do not establish claim support.",
    }
