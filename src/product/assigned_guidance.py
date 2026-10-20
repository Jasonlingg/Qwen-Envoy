"""Opt-in guidance candidate; baseline prompt bytes stay reproducible."""
from __future__ import annotations

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
    if version not in {BASELINE, FOCUSED}:
        raise ValueError("reader_guidance must be baseline or focused-v1")
    if version != BASELINE and reader_mode != "assigned":
        raise ValueError("focused-v1 guidance requires the assigned reader")


def guidance_prompt(base: str, version: str) -> str:
    if version == BASELINE:
        return base
    # Replace the optional rule rather than leaving conflicting instructions.
    if base.count(GAP_PROTOCOL) != 1:
        raise ValueError("focused guidance requires the known assigned gap protocol")
    required = GAP_PROTOCOL.replace("you may\nuse", "you must\nuse")
    return base.replace(GAP_PROTOCOL, required) + FOCUSED_PROTOCOL
