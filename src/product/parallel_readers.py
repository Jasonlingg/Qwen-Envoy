"""Run one reader per paper in parallel and merge only verified evidence."""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor

from src.product.arxiv_ids import validate_arxiv_id
from src.product.packet_verifier import verify_investigation_packet
from src.research.agent import load_snapshot

MAX_PAPERS = 5
WARNING = ("Exact excerpts prove provenance, not that they support any proposed answer; "
           "reason over the quotes, not the proposals.")


def investigate_papers(investigator, library, question: str, arxiv_ids: list[str], *,
                       max_workers: int = 5) -> dict:
    if not isinstance(question, str) or not question.strip() or len(question) > 4_000:
        raise ValueError("question must contain 1..4000 characters")
    if not isinstance(arxiv_ids, list) or not 1 <= len(arxiv_ids) <= MAX_PAPERS:
        raise ValueError(f"investigate accepts 1..{MAX_PAPERS} papers")
    if type(max_workers) is not int or not 1 <= max_workers <= MAX_PAPERS:
        raise ValueError("max_workers must be 1..5")
    # Resolve/build views inside each worker: an individual paper failure must
    # not prevent other readers from producing verified evidence.
    versions = list(dict.fromkeys(validate_arxiv_id(value) for value in arxiv_ids))
    allowed = frozenset({hashlib.sha256(investigator.system_prompt.encode()).hexdigest()})

    def run(version):
        try:
            version = library.summary(version)["arxiv_id"]
            view = library.view([version])
            packet = investigator.investigate(question.strip(), view)
            manifest, docs = load_snapshot(view)
        except Exception as exc:
            return version, None, "error", f"{type(exc).__name__}: {exc}"[:500]
        try:
            verified = verify_investigation_packet(
                packet, manifest, docs, allowed_prompt_sha256s=allowed)
        except Exception as exc:
            return version, None, "unverified", f"{type(exc).__name__}: {exc}"[:500]
        if verified["status"] in {"error", "unavailable"}:
            verified["error"] = str(packet.get("error", "reader did not complete"))[:500]
        verified["harness_shim_sha256"] = packet.get("harness_shim_sha256")
        return version, verified, None, None

    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(versions)))) as pool:
        results = list(pool.map(run, versions))  # map preserves input order

    papers, evidence, proposals = [], [], []
    for version, verified, failure_status, error in results:
        if verified is None:
            papers.append({"arxiv_id": version, "status": failure_status, "error": error,
                           "model_requests_attempted": None})
            continue
        papers.append({"arxiv_id": version, "status": verified["status"],
                       **{key: verified.get(key) for key in (
                           "candidate_answer", "model_requests_attempted", "error",
                           "harness_version", "harness_shim_sha256", "prompt_hash",
                           "tool_preamble_sha256", "model_identity")}})
        if verified.get("reader_mode") == "assigned":
            papers[-1].update({key: verified[key] for key in (
                "reader_mode", "paper_assignment", "assignment_sha256")})
        if verified["candidate_answer"]:
            proposals.append({"arxiv_id": version, "text": verified["candidate_answer"]})
        evidence.extend(verified["evidence"])
    for index, item in enumerate(evidence, 1):
        item["evidence_id"] = f"E{index}"

    statuses = {paper["status"] for paper in papers}
    status = ("evidence_found" if evidence else
              "no_evidence" if statuses == {"no_evidence"} else
              "unavailable" if statuses == {"unavailable"} else "incomplete")
    return {"status": status, "question": question.strip(), "papers": papers,
            "evidence": evidence, "candidate_answers": proposals,
            "semantic_support": "not_reviewed", "warning": WARNING}
