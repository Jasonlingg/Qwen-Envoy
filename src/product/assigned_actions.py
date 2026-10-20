"""Conservative displayed-tool Python for the opt-in assigned-paper protocol."""
from __future__ import annotations

import ast

from src.product.harness_v2 import scoped_search

_TOOLS = {"search", "read", "passage", "extract", "scan", "search_within", "verify", "list_docs"}
_INSPECTION = {"read", "passage", "extract", "scan", "search_within"}
_USAGE = ("assigned readers use one literal document-tool call, optionally print(tool(...)), "
          "or result = tool(...); print(result). Other Python forms are rejected")
ACTION_PROTOCOL = """
Assigned-reader Python is deliberately restricted to a displayed document-tool
call with literal arguments: search_within("assigned_id", "query"),
print(passage("assigned_id", 0, 1000)), or
hits = search_within("assigned_id", "query"); print(hits).
The optional scratch variable must be named hits or result. Use one tool call
per action. Imports, functions, conditionals, loops, tool
redefinitions and arbitrary expressions are rejected. Variables cannot be used
as tool arguments. Only actually displayed source text counts as inspection.
Prefer short focused windows; truncated structured output may require a shorter read.
"""


def displayed_tool_call(action: str) -> ast.Call:
    """Validate structure without executing any model code on the host."""
    try:
        tree = ast.parse(action)
        if len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr):
            call = tree.body[0].value
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
                if call.func.id == "print" and len(call.args) == 1 and not call.keywords:
                    call = call.args[0]
        elif (len(tree.body) == 2 and isinstance(tree.body[0], ast.Assign)
              and len(tree.body[0].targets) == 1
              and isinstance(tree.body[0].targets[0], ast.Name)):
            name = tree.body[0].targets[0].id
            if name not in {"hits", "result"}:
                raise ValueError(_USAGE)
            call = tree.body[0].value
            display = tree.body[1]
            if (not isinstance(display, ast.Expr) or not isinstance(display.value, ast.Call)
                    or not isinstance(display.value.func, ast.Name)
                    or display.value.func.id != "print" or display.value.keywords
                    or len(display.value.args) != 1
                    or not isinstance(display.value.args[0], ast.Name)
                    or display.value.args[0].id != name):
                raise ValueError(_USAGE)
        else:
            raise ValueError(_USAGE)
        if (not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name)
                or call.func.id not in _TOOLS or any(kw.arg is None for kw in call.keywords)):
            raise ValueError(_USAGE)
        for value in [*call.args, *(kw.value for kw in call.keywords)]:
            literal = ast.literal_eval(value)
            if type(literal) not in {str, int, float, type(None)}:
                raise ValueError(_USAGE)
        return call
    except (ValueError, TypeError, SyntaxError, RecursionError) as exc:
        raise ValueError(_USAGE) from exc


def source_text_visible(call: ast.Call, output: str, docs: dict) -> bool:
    """Check that a permitted inspection returned text from the assigned source.

    This establishes displayed source access, not semantic reading or support.
    Every model code action was structurally constrained before Docker execution.
    """
    name = call.func.id
    if name not in _INSPECTION and not scoped_search(call, docs):
        return False
    source = next(iter(docs.values()))["text"]
    stripped = output.strip()
    if name == "read" and (stripped == source.strip() or (
            len(source) > 8_000 and stripped.startswith(source[:160])
            and stripped.endswith(source[-160:].rstrip()))):
        return True
    try:
        value = ast.literal_eval(stripped)
    except (ValueError, TypeError, SyntaxError, RecursionError):
        return False
    items = value if isinstance(value, list) else [value]
    for item in items:
        if name == "extract" and isinstance(item, str) and item.strip() and item in source:
            return True
        if isinstance(item, dict):
            text = item.get("text")
            start = item.get("start", item.get("offset"))
            if (isinstance(text, str) and text.strip() and type(start) is int and start >= 0
                    and source[start:start + len(text)] == text):
                return True
    return False
