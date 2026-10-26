"""Source-bound comparison of papers for one proposed experiment decision."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from src.product.ml_experiment_types import Key, RecordModel, Text


class ComparedFinding(RecordModel):
    reference_id: Key
    relation: Literal["supports", "qualifies", "challenges", "different_setting"]
    reported_finding: Text
    intervention: Text
    comparator: Text
    setting: Text
    transfer_limit: Text
    independence_note: Text


class ResearchJustification(RecordModel):
    candidate_method: Text
    simpler_baseline: Text
    compared_findings: list[ComparedFinding] = Field(min_length=2, max_length=8)
    transfer_gaps: list[Text] = Field(min_length=1, max_length=8)
    cheapest_matched_test: Text
    decision_rule: Text
    user_reported_context: Text | None = None
    semantic_review: Literal["not_reviewed"] = "not_reviewed"

    @model_validator(mode="after")
    def non_redundant_comparison(self):
        ids = [item.reference_id for item in self.compared_findings]
        if len(ids) != len(set(ids)):
            raise ValueError("Each compared finding needs a different paper span")
        if all(item.relation == "supports" for item in self.compared_findings):
            raise ValueError("Compare at least one qualifying, challenging, or different setting")
        return self


def validate_paper_comparison(justification: ResearchJustification, references) -> None:
    """Check source identities; the meaning of a quotation still needs review."""
    by_id = {ref.reference_id: ref for ref in references}
    papers = set()
    for item in justification.compared_findings:
        ref = by_id.get(item.reference_id)
        if ref is None or ref.kind != "paper_span":
            raise ValueError("Every compared finding needs a saved paper_span reference")
        papers.add(ref.arxiv_id.rsplit("v", 1)[0])
    if len(papers) < 2:
        raise ValueError("Compare at least two distinct papers, not versions or spans of one")


def justification_packet(selected_research_note: dict) -> dict:
    """Render an existing immutable note without fetching sources or scoring claims."""
    note = selected_research_note["note"]
    justification = note.get("justification")
    if justification is None:
        raise ValueError("This research note has no structured justification")
    refs = {ref["reference_id"]: ref for ref in note["references"]}
    receipts = {receipt["reference_id"]: receipt
                for receipt in selected_research_note["evidence_receipts"]}
    findings = []
    for item in justification["compared_findings"]:
        ref = refs[item["reference_id"]]
        receipt = receipts[item["reference_id"]]
        if ref["kind"] != "paper_span" or receipt["verification"] != "paper_span_verified":
            raise ValueError("Compared paper is missing a verified source receipt")
        findings.append({**item, "arxiv_id": ref["arxiv_id"],
                         "source_url": receipt["metadata"]["source_url"],
                         "span_start": ref["start"], "span_end": ref["end"],
                         "quote": receipt["value"],
                         "source_sha256": receipt["source_sha256"],
                         "coverage": receipt["metadata"]["coverage"]})
    return {"experiment_id": selected_research_note["experiment_id"],
            "research_note_id": note["note_id"],
            "research_note_sha256": selected_research_note["sha256"],
            "question": note["question"],
            "candidate_method": justification["candidate_method"],
            "simpler_baseline": justification["simpler_baseline"],
            "compared_findings": findings,
            "transfer_gaps": justification["transfer_gaps"],
            "cheapest_matched_test": justification["cheapest_matched_test"],
            "decision_rule": justification["decision_rule"],
            "user_reported_context": justification["user_reported_context"],
            "semantic_review": "not_reviewed",
            "warning": "Source hashes and exact quotes establish provenance only. "
                       "Relations, reported effects, paper independence, transfer, "
                       "and the proposed test are caller-authored and need claim-level review. "
                       "This packet is not a consensus score or an experiment result."}
