"""Translate a lone literal submit(...) call using AST parsing, never execution."""

from __future__ import annotations

import ast
import json

_FIELDS = ("answer", "citations", "evidence")


def _literal(node: ast.AST):
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError, RecursionError) as exc:
        raise ValueError(
            "submit() arguments must be literal values, not variables or expressions"
        ) from exc


def submit_call_to_line(action: str) -> str | None:
    """Return a legacy submission line, None for other code, or a clear rejection."""
    try:
        tree = ast.parse(action.strip())
    except SyntaxError:
        return None
    if (len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr)
            or not isinstance(tree.body[0].value, ast.Call)):
        return None
    call = tree.body[0].value
    if not isinstance(call.func, ast.Name) or call.func.id not in {"submit", "SUBMIT"}:
        return None
    if len(call.args) > len(_FIELDS):
        raise ValueError("submit() takes answer, citations, and evidence")
    values = {name: _literal(node) for name, node in zip(_FIELDS, call.args)}
    for keyword in call.keywords:
        if keyword.arg not in _FIELDS or keyword.arg in values:
            raise ValueError(f"submit() got an unexpected or repeated argument {keyword.arg!r}")
        values[keyword.arg] = _literal(keyword.value)

    answer = values.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("submit() needs a non-empty answer string")
    answer = " ".join(answer.split())
    if "CITATIONS:" in answer.upper() or "EVIDENCE:" in answer.upper():
        raise ValueError("answer text must not contain CITATIONS: or EVIDENCE:")
    citations = values.get("citations", [])
    if not isinstance(citations, list) or not all(isinstance(c, str) for c in citations):
        raise ValueError("citations must be a list of document IDs")
    evidence = values.get("evidence", [])
    if not isinstance(evidence, list) or not all(isinstance(e, dict) for e in evidence):
        raise ValueError("evidence must be a list of objects with doc_id, start, end")
    try:
        suffix = (f"CITATIONS: {json.dumps(citations, allow_nan=False)} "
                  f"EVIDENCE: {json.dumps(evidence, allow_nan=False)}")
    except (TypeError, ValueError) as exc:
        raise ValueError("submit() evidence must contain JSON-compatible literal values") from exc
    return f"SUBMIT: {answer} {suffix}"
