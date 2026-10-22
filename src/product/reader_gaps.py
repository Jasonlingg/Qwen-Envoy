"""Bounded reader gap reports: untrusted suggestions, never evidence or searches."""
from __future__ import annotations

import ast

MAX_GAPS = 3
_LIMITS = {"missing": 600, "follow_up_query": 300}
GAP_PROTOCOL = """
If important evidence is still missing after inspecting source text, you may
use one standalone control action with literal strings:
report_gap(missing="What you have not established", follow_up_query="A useful next search")
This records an unreviewed suggestion for the host. It does not run a search,
count as source inspection, or end your investigation. At most three distinct
reports are allowed and each uses one action. Report gaps before your final
submission so verified findings and gaps can both reach the host. Continue
using document tools, then submit supported findings with exact spans or use
the existing abstention protocol. Say you have not found evidence, not that
no evidence exists. Do not invent supporting passages or search results.
"""

TERMINAL_GAP_PROTOCOL = """
If the inspected passages do not establish the requested answer, finish with
one standalone control action using literal strings:
submit_gap(missing="What these passages do not establish", follow_up_query="A useful next search")
This records the gap and ends with no accepted answer in the same action.
First complete two distinct successful passage reads that display source text.
Metadata, failed reads and the gap action never count as inspection. Do not
use report_gap() or a separate Unanswerable submission in this protocol.
The report is an unreviewed suggestion for the host. No search is launched.
Say what you have not established in the selected inspected passages; do not
claim absence from the full paper or literature. Never invent a measurement.
"""


def _texts(values: dict) -> dict:
    if set(values) != set(_LIMITS):
        raise ValueError("report_gap() needs missing and follow_up_query only")
    result = {}
    for field, limit in _LIMITS.items():
        value = values[field]
        if not isinstance(value, str) or not value.strip() or len(value) > limit:
            raise ValueError(f"{field} must contain 1..{limit} characters")
        result[field] = " ".join(value.split())
    return result


def parse_gap_call(action: str) -> dict | None:
    """Parse a lone literal report_gap call; never evaluate model Python."""
    return _parse_gap_action(action, "report_gap")


def parse_gap_submission(action: str) -> dict | None:
    """Parse a terminal gap as bounded literal AST data, never executable code."""
    if not isinstance(action, str) or len(action) > 8_000:
        raise ValueError("terminal gap action must be bounded text")
    return _parse_gap_action(action, "submit_gap")


def _parse_gap_action(action: str, name: str) -> dict | None:
    try:
        tree = ast.parse(action)
    except SyntaxError:
        return None
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == name]
    if not calls:
        return None
    if (len(calls) != 1 or len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr)
            or tree.body[0].value is not calls[0]):
        raise ValueError(f"{name}() must be the only statement in its action")
    call = calls[0]
    fields = tuple(_LIMITS)
    if len(call.args) > len(fields):
        raise ValueError(f"{name}() takes missing and follow_up_query")
    nodes = dict(zip(fields, call.args))
    for keyword in call.keywords:
        if keyword.arg not in fields or keyword.arg in nodes:
            raise ValueError(f"{name}() received an unexpected or repeated argument")
        nodes[keyword.arg] = keyword.value
    try:
        values = {name: ast.literal_eval(node) for name, node in nodes.items()}
    except (ValueError, TypeError, SyntaxError, RecursionError) as exc:
        raise ValueError(f"{name}() arguments must be literal strings") from exc
    try:
        return _texts(values)
    except ValueError as exc:
        raise ValueError(str(exc).replace("report_gap()", f"{name}()")) from exc


def record_gap(result: dict, fields: dict, step: int, inspection_steps: int) -> dict:
    if not inspection_steps:
        raise ValueError("inspect source text before reporting a research gap")
    gaps = result["research_gaps"]
    if any(all(gap[key] == value for key, value in fields.items()) for gap in gaps):
        raise ValueError("this gap was already reported; inspect or submit")
    if len(gaps) >= MAX_GAPS:
        raise ValueError("at most three research gaps may be reported")
    report = {**fields, "reported_at_step": step}
    gaps.append(report)
    return report


def verify_gap_reports(packet: dict) -> tuple[list, str | None]:
    """Malformed ancillary reports cannot discard independently verified quotes."""
    try:
        gaps = packet.get("research_gaps", [])
        if not isinstance(gaps, list) or len(gaps) > MAX_GAPS:
            raise ValueError("invalid gap report count")
        checked, seen = [], set()
        trajectory = packet.get("trajectory", [])
        for gap in gaps:
            if not isinstance(gap, dict) or set(gap) != {*_LIMITS, "reported_at_step"}:
                raise ValueError("invalid gap report fields")
            fields = _texts({key: gap[key] for key in _LIMITS})
            step = gap["reported_at_step"]
            if type(step) is not int or not 1 <= step <= len(trajectory):
                raise ValueError("invalid gap report step")
            trace = trajectory[step - 1]
            action = trace.get("action") if isinstance(trace, dict) else None
            terminal = isinstance(trace, dict) and trace.get("gap_submission") is True
            if terminal and (packet.get("reader_guidance") != "passages-v2"
                    or packet.get("status") != "no_evidence" or step != len(trajectory)
                    or type(packet.get("inspection_steps")) is not int
                    or packet["inspection_steps"] < 2
                    or type(packet.get("document_tool_steps")) is not int
                    or packet["document_tool_steps"] < 2
                    or len({(t.get("action")) for t in trajectory[:step - 1]
                            if isinstance(t, dict) and t.get("source_inspection") is True}) < 2):
                raise ValueError("terminal gap lacks two prior distinct source inspections")
            parsed = ((parse_gap_submission(action) if terminal else parse_gap_call(action))
                      if isinstance(action, str) and len(action) <= 8_000 else None)
            if (not isinstance(trace, dict) or trace.get("research_gap") != gap
                    or not isinstance(action, str) or len(action) > 8_000
                    or parsed != fields
                    or not any(isinstance(t, dict) and t.get("source_inspection") is True
                               for t in trajectory[:step - 1])
                    or type(packet.get("inspection_steps")) is not int
                    or packet["inspection_steps"] < 1):
                raise ValueError("gap report lacks a recorded inspected-source action")
            key = tuple(fields.values())
            if key in seen:
                raise ValueError("repeated gap report")
            seen.add(key)
            checked.append({**fields, "reported_at_step": step,
                            "review_status": "unreviewed_model_report", "absence_proven": False})
        return checked, None
    except (ValueError, TypeError, SyntaxError, RecursionError):
        return [], "Malformed research gap reports were excluded; verified evidence was retained."
