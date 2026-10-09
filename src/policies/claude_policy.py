"""Reference policy: Claude decides what code to write at each step.

This validates that the environment rewards good exploration strategies.
Future work replaces this with an open-weight model trained via GRPO.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import anthropic
from loguru import logger

from src.policies.claude_action_cleaning import clean_action
from src.policies.claude_prompts import SYSTEM_PROMPT

DEFAULT_MODEL = "claude-haiku-4-5-20251001"


class ModelIdentityError(ValueError):
    """A pinned teacher request or provider response used a different model."""


class ClaudePolicy:
    """Reference policy using Claude API as the agent brain.

    Set CLAUDE_MODEL_PATH to use a different model (e.g. as a stronger teacher
    for trajectory distillation) without touching call sites, matching the
    BASE_MODEL_PATH override pattern used by the local Qwen policies.
    """

    def __init__(
        self,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float | None = 0.0,
        api_key: str | None = None,
        system_prompt: str = SYSTEM_PROMPT,
        expected_model: str | None = None,
        action_cleaner: Callable[[str], str] = clean_action,
    ) -> None:
        self.model = model or os.environ.get("CLAUDE_MODEL_PATH") or DEFAULT_MODEL
        self.expected_model = expected_model
        if expected_model is not None and self.model != expected_model:
            raise ModelIdentityError(
                f"Expected teacher {expected_model!r}, got {self.model!r}; refusing API calls"
            )
        self.client = anthropic.Anthropic(api_key=api_key)
        self._budget = None
        budget_path = os.environ.get("ENVOY_CLAUDE_BUDGET_PATH")
        if budget_path:
            from src.policies.teacher_budget import BudgetedMessages, TeacherBudget

            self.client = self.client.with_options(max_retries=0, timeout=120)
            self._budget = TeacherBudget(
                Path(budget_path), float(os.environ["ENVOY_CLAUDE_BUDGET_USD"])
            )
            self._messages = BudgetedMessages(
                self.client.messages, self._budget, "learning_loop_model_sweep"
            )
        else:
            self._messages = None
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.action_cleaner = action_cleaner
        self.temperature = temperature
        self._temperature_supported = temperature is not None
        self.history: list[dict] = []
        self.usage_records: list[dict] = []

    def act(self, observation: str) -> str:
        """Given an observation, return an action (code or SUBMIT)."""
        self.history.append({"role": "user", "content": observation})

        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": self.system_prompt,
            "messages": self.history,
        }
        if os.environ.get("ENVOY_CLAUDE_THINKING_DISABLED") == "1":
            kwargs["thinking"] = {"type": "disabled"}
        if self._temperature_supported:
            kwargs["temperature"] = self.temperature
        try:
            response = (self._messages or self.client.messages).create(**kwargs)
        except anthropic.BadRequestError as exc:
            if self._temperature_supported and "temperature" in str(exc):
                # Newer model generations (e.g. claude-sonnet-5) reject an
                # explicit temperature entirely rather than just clamping it.
                logger.warning(f"{self.model} rejected temperature; retrying without it")
                self._temperature_supported = False
                kwargs.pop("temperature", None)
                response = (self._messages or self.client.messages).create(**kwargs)
            else:
                raise

        usage = getattr(response, "usage", None)
        record = {
            "model": getattr(response, "model", self.model),
            "stop_reason": getattr(response, "stop_reason", None),
        }
        for key in ("input_tokens", "output_tokens", "cache_creation_input_tokens",
                    "cache_read_input_tokens"):
            value = getattr(usage, key, None)
            if isinstance(value, int):
                record[key] = value
        self.usage_records.append(record)

        if self.expected_model is not None and getattr(response, "model", None) != self.expected_model:
            raise ModelIdentityError(
                f"Requested {self.expected_model!r}, but provider returned "
                f"{getattr(response, 'model', None)!r}; refusing to use this action"
            )

        # Skip non-text blocks (e.g. ThinkingBlock, present when a model's
        # native reasoning mode is on by default) and take the final text
        # block, which is the model's actual response after any reasoning.
        text_blocks = [block.text for block in response.content if hasattr(block, "text")]
        if not text_blocks:
            raise ValueError(f"{self.model} response had no text content block")
        action = text_blocks[-1]
        self.history.append({"role": "assistant", "content": action})

        # Clean raw model output into executable Python
        action = self.action_cleaner(action)

        logger.debug(f"Claude action ({len(action)} chars): {action[:100]}...")
        return action

    def reset(self) -> None:
        """Clear history for a new episode."""
        self.history = []
        self.usage_records = []
