"""Conservative, single-worker request budget for a bounded hosted comparison.

This is a local estimate at declared catalog prices, not a provider billing cap.
Failed requests retain their reservation; they are never automatically retried.
"""

from __future__ import annotations

import math

from src.policies.openai_compatible import _post_json


class InferenceBudget:
    def __init__(self, config: dict, transport=_post_json):
        self.config = dict(config)
        for key in ("max_estimated_usd", "input_usd_per_million", "output_usd_per_million"):
            value = config.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"budget {key} must be positive and finite")
        for key in ("max_requests", "max_input_utf8_bytes"):
            if type(config.get(key)) is not int or config[key] < 1:
                raise ValueError(f"budget {key} must be a positive integer")
        self.transport = transport
        self.requests = 0
        self.estimated_usd = 0.0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.missing_usage_requests = 0
        self.halted_reason = None

    def _price(self, prompt: int, completion: int) -> float:
        return (prompt * self.config["input_usd_per_million"]
                + completion * self.config["output_usd_per_million"]) / 1_000_000

    def _halt(self, reason: str):
        self.halted_reason = reason
        raise RuntimeError(reason)

    def __call__(self, url, payload, headers, timeout):
        if self.halted_reason:
            self._halt(self.halted_reason)
        messages = payload["messages"]
        input_bytes = sum(len(message["content"].encode("utf-8")) for message in messages)
        if input_bytes > self.config["max_input_utf8_bytes"]:
            self._halt("request exceeds configured input byte ceiling")
        if self.requests >= self.config["max_requests"]:
            self._halt("request count budget exhausted")
        # One token per UTF-8 byte plus deliberately generous chat framing.
        # This bounds our estimate, not an undocumented provider tokenizer.
        reservation = self._price(input_bytes + 1024 * len(messages), payload["max_tokens"])
        if self.estimated_usd + reservation > self.config["max_estimated_usd"]:
            self._halt("estimated inference cost budget exhausted")
        self.requests += 1
        self.estimated_usd += reservation
        try:
            response = self.transport(url, payload, headers, timeout)
        except Exception:
            self.halted_reason = "endpoint request failed; no automatic retry"
            raise
        usage = response.get("usage") or {}
        prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
        if (type(prompt) is int and prompt >= 0 and type(completion) is int
                and completion >= 0):
            actual = self._price(prompt, completion)
            self.estimated_usd += actual - reservation
            self.prompt_tokens += prompt
            self.completion_tokens += completion
            if actual > reservation:
                self.halted_reason = "provider usage exceeded conservative reservation"
        else:
            self.missing_usage_requests += 1
        return response

    def snapshot(self):
        return {
            "config": self.config, "requests": self.requests,
            "estimated_usd": round(self.estimated_usd, 8),
            "prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
            "missing_usage_requests": self.missing_usage_requests,
            "halted_reason": self.halted_reason,
            "cost_note": "Catalog-rate estimate, not an invoice; no cache discounts applied. "
                         "Unknown usage and failed requests retain conservative reservations.",
        }
