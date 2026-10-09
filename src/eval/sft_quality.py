"""Conservative structural checks for known-paper code demonstrations.

These checks find bad demonstrations; passing them does not prove semantic support.
No generated Python is executed here. Replay and source review remain necessary.
"""

from __future__ import annotations

import ast
import hashlib
import json
import random
import re
from collections import defaultdict

from src.env.reward import parse_submission_details

DIRECT_TOOLS = {"read", "passage", "search_within", "extract"}


def known_paper_id(question: str) -> str:
    match = re.search(r'doc_id:\s*"([^"\s]+)"', question)
    if match is None:
        raise ValueError("Known-paper question must explicitly supply a doc_id")
    return match.group(1)


def directly_accesses_paper(action: str, doc_id: str) -> bool:
    """Static candidate check, allowing a literal or a simple doc_id assignment."""
    try:
        tree = ast.parse(action)
    except SyntaxError:
        return False
    constants = {
        target.id: node.value.value
        for node in tree.body if isinstance(node, ast.Assign)
        if isinstance(node.value, ast.Constant)
        for target in node.targets if isinstance(target, ast.Name)
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id not in DIRECT_TOOLS:
            continue
        arg = node.args[0] if node.args else next(
            (kw.value for kw in node.keywords if kw.arg == "doc_id"), None
        )
        value = arg.value if isinstance(arg, ast.Constant) else (
            constants.get(arg.id) if isinstance(arg, ast.Name) else None
        )
        if value == doc_id:
            return True
    return False


def trajectory_hash(row: dict) -> str:
    payload = {key: row[key] for key in ("question_id", "question", "trajectory")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def trajectory_issues(row: dict, *, max_actions: int = 6) -> list[str]:
    """Reject impossible episodes and poor candidates for the next focused pilot."""
    steps = row.get("trajectory") or []
    issues = []
    if len(steps) < 2:
        issues.append("no_investigation")
    if len(steps) > max_actions:
        issues.append("over_action_budget")
    if any(step.get("done") for step in steps[:-1]):
        issues.append("action_after_episode_done")
    if not steps:
        return issues + ["no_submit"]
    try:
        doc_id = known_paper_id(row["question"])
    except ValueError:
        return issues + ["missing_known_paper"]
    if not directly_accesses_paper(steps[0]["action"], doc_id):
        issues.append("no_direct_first_action")
    for step in steps[:-1]:
        try:
            ast.parse(step["action"])
        except SyntaxError:
            issues.append("invalid_code")
        observation = step.get("observation", "")
        if not observation.strip():
            issues.append("missing_observation")
        error_pattern = (
            r"Traceback \(most recent call last\)|\b(?:SyntaxError|TypeError|NameError):|^ERROR:"
        )
        if re.search(error_pattern, observation, re.M):
            issues.append("execution_error")
    actions = [step["action"].strip() for step in steps[:-1]]
    if len(set(actions)) < len(actions):
        issues.append("repeated_action")
    submission = parse_submission_details(steps[-1]["action"])
    if submission is None:
        issues.append("no_submit")
    elif not submission[0]:
        issues.append("empty_answer")
    elif row.get("expected_answerability") == "sufficient" and doc_id not in submission[1]:
        issues.append("missing_target_citation")
    return sorted(set(issues))


def split_by_paper(rows: list[dict], val_fraction: float, seed: int) -> tuple[list, list]:
    """Keep every question/turn from a paper in the same split."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between zero and one")
    groups = defaultdict(list)
    for row in rows:
        groups[known_paper_id(row["question"])].append(row)
    if len(groups) < 2:
        raise ValueError("Need at least two reviewed papers for train/validation isolation")
    paper_ids = sorted(groups)
    random.Random(seed).shuffle(paper_ids)
    target = max(1, round(len(rows) * val_fraction))
    val, train = [], []
    for index, paper in enumerate(paper_ids):
        (val if len(val) < target and index < len(paper_ids) - 1 else train).extend(groups[paper])
    return train, val
