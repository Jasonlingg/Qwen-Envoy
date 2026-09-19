"""Diagnostic wrapper that removes retrieval choice from an agent episode."""

from __future__ import annotations

import json
import re

from src.policies.protocol import Policy


KNOWN_DOC_ID = re.compile(r"doc_id:\s*[\"']([^\"']+)[\"']")


class ForcedFirstReadPolicy:
    """Execute one deterministic ``read()`` before delegating to a chat policy.

    The forced turn is also inserted into the wrapped policy's chat history. This
    makes the next model forward pass identical to a normal agent turn in which
    the policy chose ``read()`` itself, while removing query wording and ranking
    from oracle-evidence diagnostics.
    """

    def __init__(self, policy: Policy) -> None:
        self.policy = policy
        self._forced = False

    def __getattr__(self, name: str):
        return getattr(self.policy, name)

    def reset(self) -> None:
        self.policy.reset()
        self._forced = False

    def act(self, observation: str) -> str:
        if self._forced:
            return self.policy.act(observation)

        match = KNOWN_DOC_ID.search(observation)
        if match is None:
            raise ValueError("Forced-first-read diagnostic requires a quoted doc_id")
        action = f"print(read({json.dumps(match.group(1))}))"

        history = getattr(self.policy, "history", None)
        if not isinstance(history, list):
            raise TypeError("Wrapped policy must expose mutable chat history")
        history.append({"role": "user", "content": observation})
        history.append({"role": "assistant", "content": action})
        self._forced = True
        return action
