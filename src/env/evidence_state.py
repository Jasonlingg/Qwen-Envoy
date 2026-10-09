"""Episode-local verification for document-research trajectories.

The verifier deliberately checks only properties that the harness can establish
without knowing the gold answer: whether the agent investigated, whether an
action repeats an earlier document query, and whether a submission's citations
and spans are structurally grounded in documents the agent observed.

Semantic support remains the policy's responsibility.  This is the document
research analogue of a small action model, not a semantic grader.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from src.eval.abstention import is_abstention as is_protocol_abstention

DOCUMENT_TOOLS = {
    "search",
    "read",
    "passage",
    "extract",
    "scan",
    "search_within",
    "verify",
    "list_docs",
}
QUERY_TOOLS = {"search", "extract", "scan", "search_within"}
DOC_ID_TOOLS = {"read", "passage", "extract", "scan", "search_within", "verify"}
FAILED_OBSERVATION_MARKERS = (
    "traceback (most recent call last)",
    "syntaxerror",
    "timed out",
    "error: document",
    "'error':",
    '"error":',
)
RESEARCH_REFUSAL = re.compile(
    r"\b(?:i|we) cannot (?:answer|confirm|determine|identify|establish|provide)\b|"
    r"\b(?:the )?(?:inspected |retrieved )?(?:paper|passages?) "
    r"(?:do|does) not (?:state|specify|mention|provide|contain|identify|report|"
    r"address|describe|name|give)\b",
    re.I,
)
REFUSAL_THEN_ANSWER = re.compile(
    r"\b(?:however|but|although|that said|based on)\b[^.]*\b(?:is|are|was|were)\b",
    re.I,
)


class VerifierEvent(BaseModel):
    """One action or submission rejected by the evidence-state verifier."""

    step: int
    kind: Literal["duplicate_action", "submission"]
    reasons: list[str]
    feedback: str
    state: dict[str, object] = Field(default_factory=dict)


class EscalationEvent(BaseModel):
    """A verifier failure handed back to the calling host model."""

    step: int
    kind: Literal["duplicate_action", "submission"]
    reasons: list[str]
    candidate_action: str
    message: str
    state: dict[str, object] = Field(default_factory=dict)


@dataclass(frozen=True)
class ParsedToolCall:
    name: str
    key: str
    doc_id: str | None = None


def is_research_abstention(answer: str) -> bool:
    """Detect the protocol token and common evidence-limited refusals.

    The established evaluation metric intentionally keeps its preregistered
    definition.  The interactive verifier needs to recognize the actual prose
    used by the policy (for example, ``I cannot confirm ...``) so it can offer a
    recovery turn.  A hedge followed by a confident answer is not a refusal.
    """
    if is_protocol_abstention(answer):
        return True
    return bool(RESEARCH_REFUSAL.search(answer) and not REFUSAL_THEN_ANSWER.search(answer))


def _literal_string(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _call_key(node: ast.Call) -> str:
    """Return a stable structural key while preserving useful argument changes."""
    return ast.dump(node, annotate_fields=True, include_attributes=False)


def parse_document_calls(action: str) -> list[ParsedToolCall]:
    """Extract document-tool calls without executing model-generated code."""
    try:
        tree = ast.parse(action)
    except SyntaxError:
        return []

    calls: list[ParsedToolCall] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        name = node.func.id
        if name not in DOCUMENT_TOOLS:
            continue
        doc_id = None
        if name in DOC_ID_TOOLS and node.args:
            doc_id = _literal_string(node.args[0])
        calls.append(ParsedToolCall(name=name, key=_call_key(node), doc_id=doc_id))
    return calls


@dataclass
class EvidenceState:
    """Compact record of what one policy has actually investigated."""

    document_lengths: dict[str, int]
    successful_tool_actions: int = 0
    seen_call_keys: set[str] = field(default_factory=set)
    distinct_query_keys: set[str] = field(default_factory=set)
    inspected_doc_ids: set[str] = field(default_factory=set)
    full_document_reads: set[str] = field(default_factory=set)
    feedback_used: int = 0
    abstention_recovery_active: bool = False

    def snapshot(self) -> dict[str, object]:
        return {
            "successful_tool_actions": self.successful_tool_actions,
            "distinct_queries": len(self.distinct_query_keys),
            "inspected_doc_ids": sorted(self.inspected_doc_ids),
            "full_document_reads": sorted(self.full_document_reads),
            "feedback_used": self.feedback_used,
            "abstention_recovery_active": self.abstention_recovery_active,
        }

    def duplicate_reason(self, action: str) -> str | None:
        """Reject an action only when every document call repeats prior work."""
        calls = parse_document_calls(action)
        if not calls or any(call.key not in self.seen_call_keys for call in calls):
            return None
        names = ", ".join(sorted({call.name for call in calls}))
        return (
            f"This repeats document-tool work already attempted ({names}). "
            "Change the query or inspection method instead of repeating the same call."
        )

    def observe(self, action: str, observation: str) -> None:
        """Record a successful tool action and documents visible in its output."""
        lowered = observation.lower()
        if any(marker in lowered for marker in FAILED_OBSERVATION_MARKERS):
            return
        calls = parse_document_calls(action)
        if not calls:
            return

        self.successful_tool_actions += 1
        for call in calls:
            self.seen_call_keys.add(call.key)
            if call.name in QUERY_TOOLS:
                self.distinct_query_keys.add(call.key)
            if call.doc_id in self.document_lengths:
                self.inspected_doc_ids.add(call.doc_id)
                if call.name == "read":
                    self.full_document_reads.add(call.doc_id)

        # General search and list_docs reveal document IDs through stdout rather
        # than through a doc_id argument. Record only IDs actually visible to the
        # policy, using boundaries to avoid matching one ID inside another.
        for doc_id in self.document_lengths:
            if re.search(rf"(?<![\w.-]){re.escape(doc_id)}(?![\w.-])", observation):
                self.inspected_doc_ids.add(doc_id)

    def submission_reasons(
        self,
        *,
        answer: str,
        citations: list[str],
        evidence: list[dict],
        require_evidence: bool,
        minimum_investigation_actions: int = 2,
    ) -> list[str]:
        """Return mechanically verifiable problems with a proposed submission."""
        reasons: list[str] = []
        abstained = is_research_abstention(answer)

        if abstained:
            if self.successful_tool_actions < minimum_investigation_actions:
                reasons.append(
                    "The refusal is premature: fewer than two successful document "
                    "investigation actions were observed."
                )
            elif self.abstention_recovery_active:
                reasons.append(
                    "The policy still abstains after a recovery investigation. "
                    "Try another evidence-gathering strategy before escalating."
                )
            return reasons

        if not answer.strip():
            reasons.append("The answer is empty.")
        if not citations:
            reasons.append("A non-abstaining answer needs at least one citation.")

        unknown_citations = sorted(set(citations) - self.document_lengths.keys())
        if unknown_citations:
            reasons.append(f"Unknown citation document IDs: {unknown_citations}.")
        unseen_citations = sorted(set(citations) - self.inspected_doc_ids)
        if unseen_citations:
            reasons.append(
                f"Cited documents were not observed during this episode: {unseen_citations}."
            )

        if require_evidence and not evidence:
            reasons.append("This evaluation requires at least one exact evidence span.")

        evidence_doc_ids: set[str] = set()
        for index, span in enumerate(evidence):
            doc_id = span.get("doc_id")
            start, end = span.get("start"), span.get("end")
            if not isinstance(doc_id, str):
                reasons.append(f"Evidence span {index} has no valid doc_id.")
                continue
            evidence_doc_ids.add(doc_id)
            length = self.document_lengths.get(doc_id)
            if (
                length is None
                or type(start) is not int
                or type(end) is not int
                or not 0 <= start < end <= length
            ):
                reasons.append(f"Evidence span {index} has invalid offsets for {doc_id!r}.")
            if doc_id not in self.inspected_doc_ids:
                reasons.append(f"Evidence document {doc_id!r} was not observed in this episode.")

        if evidence and set(citations) != evidence_doc_ids:
            reasons.append("Citation IDs and evidence-span document IDs must match.")
        return reasons


def format_feedback(
    *,
    reasons: list[str],
    state: EvidenceState,
    feedback_number: int,
    feedback_budget: int,
    kind: Literal["duplicate_action", "submission"] = "submission",
) -> str:
    """Build concise, structured feedback for the policy's next turn."""
    snapshot = state.snapshot()
    bullets = "\n".join(f"- {reason}" for reason in reasons)
    docs = snapshot["inspected_doc_ids"] or "none"
    recovery = ""
    if kind == "duplicate_action" or any(
        marker in reason.lower()
        for reason in reasons
        for marker in ("premature", "still abstains")
    ):
        if snapshot["inspected_doc_ids"]:
            doc_id = repr(snapshot["inspected_doc_ids"][0])
            recovery = (
                "\nYour next response must be one DIFFERENT Python investigation action, "
                "not SUBMIT. Choose a useful option such as:\n"
                f"- print(search_within({doc_id}, '<different keywords>', top_k=8))\n"
                f"- print(scan({doc_id}, r'<target term|number>', max_hits=6))\n"
                f"- text = read({doc_id}); print(text[:6000])\n"
                "Replace the placeholders with terms from the question. Do not repeat the "
                "previous call."
            )
        else:
            recovery = (
                "\nYour next response must be one Python investigation action, not SUBMIT. "
                "Search for the question's key entities before answering."
            )
    else:
        recovery = (
            "\nCorrect the cited document IDs or evidence spans, then submit again. "
            "Do not invent a source you did not inspect."
        )
    return (
        f"[Evidence verifier — recovery {feedback_number}/{feedback_budget}]\n"
        "The proposed action was not accepted:\n"
        f"{bullets}\n"
        "Current evidence state:\n"
        f"- successful document actions: {snapshot['successful_tool_actions']}\n"
        f"- distinct searches: {snapshot['distinct_queries']}\n"
        f"- inspected document IDs: {docs}\n"
        f"{recovery}\n"
        "The verifier checks structure and search coverage; you must judge semantic support."
    )


def format_escalation(
    *,
    reasons: list[str],
    state: EvidenceState,
    feedback_budget: int,
) -> str:
    """Return a machine-readable handoff message after bounded recovery fails."""
    snapshot = state.snapshot()
    bullets = "\n".join(f"- {reason}" for reason in reasons)
    return (
        "[ESCALATE_TO_HOST]\n"
        f"The small-model researcher exhausted {feedback_budget} recovery attempts.\n"
        f"{bullets}\n"
        "Evidence state:\n"
        f"- successful document actions: {snapshot['successful_tool_actions']}\n"
        f"- distinct searches: {snapshot['distinct_queries']}\n"
        f"- inspected document IDs: {snapshot['inspected_doc_ids']}\n"
        "The host model should inspect the recorded trajectory and continue the research."
    )
