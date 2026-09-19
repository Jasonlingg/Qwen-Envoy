from __future__ import annotations

import pytest

from src.policies.forced_first_read import ForcedFirstReadPolicy


class RecordingPolicy:
    def __init__(self) -> None:
        self.history: list[dict[str, str]] = []

    def reset(self) -> None:
        self.history = []

    def act(self, observation: str) -> str:
        self.history.append({"role": "user", "content": observation})
        action = 'SUBMIT: answer CITATIONS: ["paper"]'
        self.history.append({"role": "assistant", "content": action})
        return action


def test_forced_read_is_recorded_before_delegating() -> None:
    inner = RecordingPolicy()
    policy = ForcedFirstReadPolicy(inner)

    first = policy.act('Question: use the paper (doc_id: "paper")')
    second = policy.act("authoritative evidence")

    assert first == 'print(read("paper"))'
    assert second == 'SUBMIT: answer CITATIONS: ["paper"]'
    assert inner.history == [
        {"role": "user", "content": 'Question: use the paper (doc_id: "paper")'},
        {"role": "assistant", "content": 'print(read("paper"))'},
        {"role": "user", "content": "authoritative evidence"},
        {"role": "assistant", "content": 'SUBMIT: answer CITATIONS: ["paper"]'},
    ]


def test_forced_read_requires_a_doc_id() -> None:
    with pytest.raises(ValueError, match="quoted doc_id"):
        ForcedFirstReadPolicy(RecordingPolicy()).act("Question without a known paper")
