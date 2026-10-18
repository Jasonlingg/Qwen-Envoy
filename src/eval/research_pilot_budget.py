"""Shared local request reservations; never a provider-side or dollar cap."""

from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
from collections.abc import Callable
from typing import Any


class BudgetStop(RuntimeError):  # noqa: N818 - follows the existing teacher-budget exception.
    """The local pilot must not dispatch another model request."""


def _integer(value: Any, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


class RequestBudget:
    """Reserve before dispatch, retain every reservation, and never retry.

    Token counts must include the complete rendered chat, using the endpoint's
    tokenizer/template. They are local measurements, not provider guarantees.
    An already dispatched concurrent request cannot be recalled when another
    request discovers missing usage. The observer receives detached snapshots
    while the state lock is held; it must be quick and must not dispatch work.
    """

    def __init__(self, max_requests: int = 96, max_input_tokens: int = 300_000,
                 max_output_tokens: int = 98_304, observer: Callable | None = None):
        limits = (max_requests, max_input_tokens, max_output_tokens)
        if any(not _integer(value, 1) for value in limits):
            raise ValueError("Local budget limits must be positive integers")
        if observer is not None and not callable(observer):
            raise ValueError("observer must be callable")
        self._caps = dict(zip(("max_requests", "max_input_tokens", "max_output_tokens"), limits))
        self._observer, self._lock = observer, threading.RLock()
        self._records: list[dict[str, Any]] = []
        self._blocked_reason: str | None = None

    def summary(self) -> dict[str, Any]:
        """Detached ledger; unknown usage is null, never silently zero."""
        with self._lock:
            known = {key: [row[key] for row in self._records if row[key] is not None]
                     for key in ("provider_prompt_tokens", "provider_completion_tokens")}
            observed = {key.removeprefix("provider_"): sum(values)
                        for key, values in known.items()}
            complete = all(len(values) == len(self._records) for values in known.values())
            actual = {key: value if complete else None for key, value in observed.items()}
            return copy.deepcopy({
                "schema_version": "research-pilot-local-budget-v1",
                "scope": "Local reservations only; no provider-side or USD cap",
                "caps": {**self._caps, "max_tokens_per_request": 1024,
                         "context_safety_tokens": 128},
                "reserved": {
                    "requests": len(self._records),
                    "input_tokens": sum(row["local_input_tokens"] for row in self._records),
                    "output_tokens": sum(row["reserved_output_tokens"] for row in self._records),
                },
                "observed_usage": observed, "actual_usage": actual,
                "usage_complete": complete,
                "blocked_reason": self._blocked_reason, "records": self._records,
            })

    def _notify(self) -> bool:
        if self._observer is not None:
            try:
                self._observer(self.summary())
            except Exception as exc:
                self._blocked_reason = f"Snapshot observer failed ({type(exc).__name__})"
                return False
        return True

    def transport(self, label: str, count_tokens: Callable, context_window: int,
                  upstream: Callable) -> Callable:
        """Wrap a nonretrying ``(url, payload, headers, timeout)`` transport."""
        if not isinstance(label, str) or not label.strip():
            raise ValueError("label must be nonempty text")
        if not _integer(context_window, 1) or not callable(count_tokens) or not callable(upstream):
            raise ValueError("A context window, chat token counter, and transport are required")

        def dispatch(url, payload, headers, timeout):
            # Snapshot before counting so callers cannot change history or token
            # limits between the reservation and the actual upstream request.
            body = copy.deepcopy(payload)
            if not isinstance(body, dict):
                raise ValueError("A chat request object is required")
            model, messages = body.get("model"), body.get("messages")
            output = body.get("max_tokens")
            if not isinstance(model, str) or not model.strip():
                raise ValueError("model must be nonempty text")
            if not isinstance(messages, list) or not messages or any(
                not isinstance(message, dict)
                or message.get("role") not in ("system", "user", "assistant", "tool")
                or not isinstance(message.get("content"), str) for message in messages
            ):
                raise ValueError("messages must be a nonempty text-only chat history")
            if not _integer(output, 1) or output > 1024:
                raise ValueError("max_tokens must be an integer between 1 and 1024")
            if (type(body.get("n", 1)) is not int or body.get("n", 1) != 1
                    or body.get("stream", False) is not False or "max_completion_tokens" in body):
                raise ValueError("Only one nonstreaming completion using max_tokens is budgeted")
            try:
                encoded = json.dumps(body, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()
            except (TypeError, ValueError):
                raise ValueError("Chat payload must contain finite JSON values") from None
            local = count_tokens(copy.deepcopy(messages))
            if not _integer(local, 1):
                raise ValueError("The complete chat token count must be a positive integer")
            if local + output + 128 > context_window:
                raise BudgetStop("Chat plus output and safety margin exceeds context window")
            with self._lock:
                if self._blocked_reason:
                    raise BudgetStop(self._blocked_reason)
                used = self.summary()["reserved"]
                additions = {"requests": 1, "input_tokens": local, "output_tokens": output}
                if any(used[key] + value > self._caps[f"max_{key}"]
                       for key, value in additions.items()):
                    raise BudgetStop("Local request/token reservation limit reached")
                row = {"request_id": len(self._records) + 1, "label": label,
                       "payload_sha256": hashlib.sha256(encoded).hexdigest(),
                       "local_input_tokens": local, "reserved_output_tokens": output,
                       "provider_prompt_tokens": None, "provider_completion_tokens": None,
                       "status": "in_flight", "elapsed_seconds": None, "exception_class": None}
                self._records.append(row)
                if not self._notify():
                    row.update(status="observer_error", elapsed_seconds=0.0)
                    raise BudgetStop(self._blocked_reason)
            started = time.monotonic()
            try:
                response = upstream(url, body, headers, timeout)
            except BaseException as exc:
                with self._lock:
                    row.update(status="error", exception_class=type(exc).__name__,
                               elapsed_seconds=time.monotonic() - started)
                    self._blocked_reason = f"Request failed ({type(exc).__name__}); usage unknown"
                    self._notify()
                raise
            with self._lock:
                usage = response.get("usage") if isinstance(response, dict) else None
                for source, target in (("prompt_tokens", "provider_prompt_tokens"),
                                       ("completion_tokens", "provider_completion_tokens")):
                    value = usage.get(source) if isinstance(usage, dict) else None
                    row[target] = value if _integer(value) else None
                prompt = row["provider_prompt_tokens"]
                completion = row["provider_completion_tokens"]
                if prompt is None or completion is None:
                    row["status"] = "usage_unknown"
                    self._blocked_reason = "Missing or invalid provider usage; reservation retained"
                elif prompt > local or completion > output:
                    row["status"] = "usage_exceeded"
                    self._blocked_reason = "Provider usage exceeded local reservation; inspect"
                else:
                    row["status"] = "ok"
                row["elapsed_seconds"] = time.monotonic() - started
                self._notify()
            return response

        return dispatch
