"""CPU-only checks for the v6 adapter reload smoke protocol."""

from __future__ import annotations

import json

import pytest

from scripts.smoke_qasper_v6_adapter import classify_next_action, first_training_prompt, smoke
from src.env.document_env import initial_observation
from src.policies.code_execution import QASPER_SYSTEM_PROMPT


def _training_row() -> dict:
    return {"messages": [
        {"role": "system", "content": QASPER_SYSTEM_PROMPT},
        {"role": "user", "content": initial_observation(
            'Use the known paper "Example" (doc_id: "qasper_123") to answer: What changed?',
            require_evidence=True, include_preamble=True,
        )},
        {"role": "assistant", "content": 'print(passage("qasper_123", 0, 500))'},
    ]}


def test_uses_exact_first_training_prompt(tmp_path):
    path = tmp_path / "train.jsonl"
    row = _training_row()
    path.write_text(json.dumps(row) + "\n")
    assert first_training_prompt(path) == row["messages"][:2]

    row["messages"][1]["content"] = "Question: What changed?"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="initial observation"):
        first_training_prompt(path)


@pytest.mark.parametrize("raw,kind", [
    ('print(passage("qasper_123", 0, 500))', "python_tool"),
    ('SUBMIT: Unanswerable CITATIONS: [] EVIDENCE: []', "submit"),
    ('SUBMIT: Yes CITATIONS: ["qasper_123"] EVIDENCE: '
     '[{"doc_id":"qasper_123","start":0,"end":10}]', "submit"),
])
def test_accepts_executable_tool_or_strict_submission(raw, kind):
    assert classify_next_action(raw) == (kind, raw)


@pytest.mark.parametrize("raw", [
    "I think the paper says yes.",
    "x = 1",
    "print(search('",
    'SUBMIT: Yes CITATIONS: ["qasper_123"]',
    'SUBMIT: Yes CITATIONS: ["qasper_123"] EVIDENCE: []',
    'SUBMIT: Unanswerable CITATIONS: ["qasper_123"] EVIDENCE: '
    '[{"doc_id":"qasper_123","start":0,"end":10}]',
    'Here is the action:\nprint(search("paper"))',
])
def test_rejects_non_protocol_action(raw):
    with pytest.raises(ValueError):
        classify_next_action(raw)


def test_rejects_missing_adapter_before_importing_gpu_dependencies(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(_training_row()) + "\n")
    with pytest.raises(ValueError, match="Saved adapter config and safetensors"):
        smoke(tmp_path / "missing", path)
