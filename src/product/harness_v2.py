"""V2 prompt and static action handling; model code is never executed here."""
from __future__ import annotations

import ast

from src.env.tools_v2 import V2_PROMPT_ADDENDUM
from src.policies.code_execution import clean_action

MAX_FORMAT_RETRIES = 2
FORMAT_RETRY_MESSAGE = (
    "Format error: that reply was neither Python nor a submission, so nothing ran and "
    "no step was used. Reply with Python code only, or finish with "
    "submit(answer=..., citations=[...], evidence=[...])."
)


def next_action(policy, prompt: str, result: dict, harness: str, max_chars: int):
    retries = []
    while True:
        result['model_requests_attempted'] += 1
        raw = policy.act(prompt)
        if not isinstance(raw, str):
            raise ValueError('Qwen policy returned a non-text action')
        action = clean_action(raw)
        if (harness != 'v2' or len(retries) >= MAX_FORMAT_RETRIES
                or not action or len(action) > max_chars
                or action.upper().startswith('SUBMIT')):
            return action, retries
        try:
            ast.parse(action)
        except SyntaxError:
            retries.append({'rejected_action': action[:max_chars]})
            prompt = FORMAT_RETRY_MESSAGE
            continue
        return action, retries


def scoped_search(node: ast.Call, doc_ids) -> bool:
    """Classify ambiguous calls conservatively so they cannot unlock discovery."""
    if not isinstance(node.func, ast.Name) or node.func.id != 'search':
        return False
    keys = {kw.arg for kw in node.keywords}
    if 'doc_id' in keys or (keys & {'query', 'q'} and node.args):
        return True
    if len(node.args) >= 2:
        first, second = node.args[:2]
        if any(isinstance(x, ast.Constant) and isinstance(x.value, str)
               and x.value in doc_ids for x in (first, second)):
            return True
        # A second text/variable argument might be a paper ID. Only numeric
        # positional top_k is unambiguously discovery before IDs are known.
        return not (isinstance(second, ast.Constant)
                    and type(second.value) in {int, float})
    return False


def v2_discovery(action: str, doc_ids) -> bool:
    try:
        calls = [n for n in ast.walk(ast.parse(action)) if isinstance(n, ast.Call)]
    except SyntaxError:
        return False
    names = {n.func.id for n in calls if isinstance(n.func, ast.Name)}
    return (bool(names & {'search', 'list_docs'})
            and not names & {'read', 'passage', 'extract', 'search_within', 'scan', 'verify'}
            and not any(scoped_search(n, doc_ids) for n in calls))


def v2_inspection(action: str, doc_ids) -> bool:
    try:
        return any(scoped_search(n, doc_ids) for n in ast.walk(ast.parse(action))
                   if isinstance(n, ast.Call))
    except SyntaxError:
        return False


def effective_prompt(base: str, *, harness: str) -> str:
    return base + V2_PROMPT_ADDENDUM if harness == 'v2' else base
