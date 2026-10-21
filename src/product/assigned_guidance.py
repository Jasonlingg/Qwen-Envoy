"""Opt-in guidance candidate; baseline prompt bytes stay reproducible."""
from __future__ import annotations

import ast

from src.product.assigned_actions import displayed_tool_call
from src.product.passage_scope import PASSAGE_PROTOCOL, PASSAGES
from src.product.reader_gaps import GAP_PROTOCOL

BASELINE = "baseline"
FOCUSED = "focused-v1"
FOCUSED_PROTOCOL = """
Focused assigned-reader guidance v1:
Start with a displayed search_within() using words from the user's question.
When a hit appears relevant, inspect a short passage() around its returned
offset before deciding that the requested fact is missing. Opening successive
abstract windows is not a substitute for inspecting relevant methods/results.
Read what the displayed text actually states. If it supplies the requested
fact, answer that fact and cite its smallest supporting span. Check that your
answer agrees with the quoted text; a real quote can contradict your claim.
Preserve the reported hardware, workload and measurement conditions. Do not
transfer a result to a different setup without supporting measurements.
Prefer print(search_within("assigned_id", "question keywords")) or
print(passage("assigned_id", 0, 500)), using the controller's actual ID and real
offsets. A bare hits = tool(...) assignment is not displayed and is rejected.
If the inspected text still does not answer the question, report what remains
unknown and a useful next search with report_gap(), then explicitly abstain.
Reserve an action for the gap report and another for the final submission.
Do not disguise a missing-answer claim as a supported factual answer. Gap
reports and source access do not establish semantic support or prove absence.
"""


def validate_guidance(version: str, reader_mode: str) -> None:
    if version not in {BASELINE, FOCUSED, PASSAGES}:
        raise ValueError("reader_guidance must be baseline, focused-v1 or passages-v1")
    if version != BASELINE and reader_mode != "assigned":
        raise ValueError("focused-v1 guidance requires the assigned reader")


def guidance_prompt(base: str, version: str) -> str:
    if version == BASELINE:
        return base
    # Replace the optional rule rather than leaving conflicting instructions.
    if base.count(GAP_PROTOCOL) != 1:
        raise ValueError("focused guidance requires the known assigned gap protocol")
    required = GAP_PROTOCOL.replace("you may\nuse", "you must\nuse")
    return base.replace(GAP_PROTOCOL, required) + (
        PASSAGE_PROTOCOL if version == PASSAGES else FOCUSED_PROTOCOL)


def action_recovery(action: str, reason: str, remaining: int) -> str:
    """Suggest a displayed retry only after the same conservative AST validation."""
    suggestion = 'print(search_within("assigned_id", "question keywords"))'
    try:
        tree = ast.parse(action)
        if (len(tree.body) == 1 and isinstance(tree.body[0], ast.Assign)
                and len(tree.body[0].targets) == 1
                and isinstance(tree.body[0].targets[0], ast.Name)
                and tree.body[0].targets[0].id in {"hits", "result"}):
            call = displayed_tool_call(ast.unparse(tree.body[0].value))
            suggestion = f"print({ast.unparse(call)})"
    except (SyntaxError, ValueError, TypeError, RecursionError):
        pass
    return (f"Action rejected: {reason}. Nothing ran; no source text was inspected. "
            f"Use displayed literal tool code, for example {suggestion}. "
            "Use the controller's actual assigned ID and a query for your question. "
            f"[{remaining} steps left.]")


def missing_gap_feedback(remaining: int) -> str:
    return ("Submission rejected: focused-v1 abstention requires a gap report first. "
            'Use report_gap(missing="What you have not established", '
            'follow_up_query="A useful next search"), then explicitly abstain. '
            "A gap report is an unreviewed host suggestion, not evidence or a search. "
            f"[{remaining} steps left.]")


def inspection_feedback(call: ast.Call) -> str:
    if call.func.id in {"search", "search_within", "extract", "scan"}:
        return ("\nController reminder: matching source text is not semantic support. "
                "If a hit appears relevant, inspect a short passage around its returned "
                "offset before claiming the requested fact was not found. Reserve the "
                "last two actions for report_gap and abstention if evidence remains missing.")
    return ("\nController reminder: check that your answer agrees with the displayed "
            "source. Cite its exact supporting span; preserve its measurement conditions. "
            "If evidence remains missing, report_gap before abstaining.")
