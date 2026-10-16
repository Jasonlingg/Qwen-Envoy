"""Failure-isolated orchestration for the MCP investigation tool."""
from __future__ import annotations

from src.product.arxiv_ids import validate_arxiv_id
from src.product.net import SourceUnavailable
from src.product.parallel_readers import WARNING, investigate_papers


def investigate(library, investigator, reasoner, question, arxiv_ids, synthesize=False):
    if not isinstance(question, str) or not question.strip() or len(question) > 4_000:
        raise ValueError("question must contain 1..4000 characters")
    if not isinstance(arxiv_ids, list) or not 1 <= len(arxiv_ids) <= 5:
        raise ValueError("investigate accepts 1..5 papers")
    if type(synthesize) is not bool:
        raise ValueError("synthesize must be a boolean")
    requested = list(dict.fromkeys(validate_arxiv_id(value) for value in arxiv_ids))
    if investigator is None:
        return {"status": "model_not_configured", "evidence": [], "papers": [],
                "model_requests_attempted": 0, "warning": WARNING,
                "detail": "Start the server with --reader nemotron or --reader qwen."}
    versions, order, seen = [], [], set()
    for value in requested:
        try:
            version = library.open(value)["arxiv_id"]
        except Exception as exc:
            failure = {"arxiv_id": value, "error": str(exc)[:500], "status": "error",
                       "model_requests_attempted": 0}
            if isinstance(exc, SourceUnavailable):
                failure.update(status="source_unavailable", source=exc.source)
            order.append((value, failure))
        else:
            if version not in seen:
                seen.add(version)
                versions.append(version)
                order.append((version, None))
    if versions:
        merged = investigate_papers(investigator, library, question, versions)
    else:
        merged = {"status": "source_unavailable", "question": question.strip(), "papers": [],
                  "evidence": [], "candidate_answers": [], "semantic_support": "not_reviewed",
                  "warning": WARNING}
    by_version = {paper["arxiv_id"]: paper for paper in merged["papers"]}
    merged["papers"] = [failure if failure is not None else by_version[version]
                        for version, failure in order]
    merged["paper_versions"] = versions
    if not merged["evidence"] and any(failure for _, failure in order):
        merged["status"] = ("source_unavailable" if all(
            paper["status"] == "source_unavailable" for paper in merged["papers"])
            else "incomplete")
    counts = [paper.get("model_requests_attempted") for paper in merged["papers"]]
    merged["model_requests_observed"] = sum(n for n in counts if type(n) is int)
    merged["model_requests_complete"] = all(type(n) is int for n in counts)
    merged["model_requests_attempted"] = (merged["model_requests_observed"]
                                          if merged["model_requests_complete"] else None)
    if synthesize:
        merged["synthesis"], merged["synthesis_error"] = None, None
        merged["synthesis_model_requests_attempted"] = 0
        if reasoner is None:
            merged["synthesis_error"] = "no reasoner configured"
        elif not merged["evidence"]:
            merged["synthesis_error"] = "no verified evidence to synthesize"
        else:
            try:
                documents = [library.document(version) for version in versions]
                merged["synthesis_model_requests_attempted"] = 1
                merged["synthesis"] = reasoner.synthesize(
                    merged, {doc["doc_id"]: doc["text"] for doc in documents})
            except Exception as exc:
                merged["synthesis_error"] = f"{type(exc).__name__}: {exc}"[:500]
    return merged
