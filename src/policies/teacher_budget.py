"""Persistent, shared spend reservations for text-only Sonnet teacher requests.

Rates checked 2026-09-19: https://platform.claude.com/docs/en/about-claude/pricing
Input counting is an estimate; reserve 20% plus 1,024 input tokens of headroom
and the full output limit. SDK retries must be disabled. Unknown charges retain
their reservation and stop the batch. This ledger is not an account balance.
"""
from __future__ import annotations

import fcntl
import json
import math
import os
from pathlib import Path
import threading
import uuid

import anthropic


class BudgetStop(RuntimeError):
    pass


class TeacherBudget:
    def __init__(self, path: Path, cap_usd: float):
        if not math.isfinite(cap_usd) or cap_usd <= 0:
            raise ValueError("A finite positive teacher budget is required")
        self.path = path
        self.lock = threading.Lock()
        self._file_lock = path.with_suffix(path.suffix + ".lock").open("a")
        try:
            fcntl.flock(self._file_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._file_lock.close()
            raise BudgetStop("Another process owns this teacher budget") from None
        cap = math.floor(cap_usd * 1_000_000)
        self.state = {"version": 1, "cap_microusd": cap, "requests": {}}
        try:
            if path.exists():
                self.state = json.loads(path.read_text())
                if self.state["cap_microusd"] != cap:
                    raise ValueError("Refusing to change the cap of an existing ledger")
                if any(r["status"] != "settled" for r in self.state["requests"].values()):
                    raise BudgetStop("Unsettled API charges: inspect the ledger before resuming")
            self._save()
        except BaseException:
            self.close()
            raise

    def _save(self):
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w") as stream:
            json.dump(self.state, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(self.path)

    def reserve(self, input_tokens: int, max_output: int, question_id: str) -> str:
        # USD 2 / million input, USD 10 / million output; micro-USD integers.
        amount = (math.ceil(input_tokens * 1.2) + 1024) * 2 + max_output * 10
        with self.lock:
            used = sum(r["charged_microusd"] for r in self.state["requests"].values())
            if used + amount > self.state["cap_microusd"]:
                raise BudgetStop("Teacher budget cannot cover the next request's reservation")
            token = uuid.uuid4().hex
            self.state["requests"][token] = {
                "question_id": question_id, "status": "reserved",
                "reserved_microusd": amount, "charged_microusd": amount,
            }
            self._save()
            return token

    def settle(self, token: str, response=None, *, rejected: bool = False):
        with self.lock:
            row = self.state["requests"][token]
            if rejected:
                cost, usage = 0, {}
            else:
                usage_object = getattr(response, "usage", None)
                usage = {key: getattr(usage_object, key, None) for key in
                         ("input_tokens", "output_tokens")}
                if any(type(v) is not int or v < 0 for v in usage.values()):
                    raise BudgetStop("Missing usage: request reservation retained")
                if any(getattr(usage_object, key, 0) for key in
                       ("cache_creation_input_tokens", "cache_read_input_tokens")):
                    raise BudgetStop("Unexpected caching usage; reservation retained")
                cost = 2 * usage["input_tokens"] + 10 * usage["output_tokens"]
            row.update(charged_microusd=cost, status="settled", usage=usage,
                       model=getattr(response, "model", None),
                       request_id=getattr(response, "_request_id", None))
            if cost > row["reserved_microusd"]:
                row["status"] = "reservation_exceeded"
            self._save()
            if row["status"] != "settled":
                raise BudgetStop("Actual usage exceeded reserved cost; stopping for inspection")

    def summary(self):
        with self.lock:
            rows = list(self.state["requests"].values())
            return {"cap_usd": self.state["cap_microusd"] / 1e6,
                    "accounted_usd": sum(r["charged_microusd"] for r in rows) / 1e6,
                    "requests": len(rows),
                    "unsettled_requests": sum(r["status"] != "settled" for r in rows)}

    def close(self):
        self._file_lock.close()


class BudgetedMessages:
    """Wrap the real no-retry Messages client, including count-token preflight."""
    def __init__(self, messages, budget: TeacherBudget, question_id: str):
        self.messages, self.budget, self.question_id = messages, budget, question_id

    def create(self, **kwargs):
        if kwargs["model"] != "claude-sonnet-5":
            raise BudgetStop("Budget is priced for claude-sonnet-5 only")
        if set(kwargs) - {"model", "max_tokens", "system", "messages", "temperature", "thinking"}:
            raise BudgetStop("Unpriced request features are not allowed")
        if "thinking" in kwargs and kwargs["thinking"] != {"type": "disabled"}:
            raise BudgetStop("Only disabled thinking is budgeted")
        if not isinstance(kwargs["system"], str) or any(
            not isinstance(m["content"], str) for m in kwargs["messages"]
        ):
            raise BudgetStop("Only plain text requests are budgeted")
        count = self.messages.count_tokens(**{k: kwargs[k] for k in ("model", "system", "messages")})
        if type(count.input_tokens) is not int or count.input_tokens < 0:
            raise BudgetStop("Invalid input-token count")
        token = self.budget.reserve(count.input_tokens, kwargs["max_tokens"], self.question_id)
        try:
            response = self.messages.create(**kwargs)
        except anthropic.APIStatusError as exc:
            if 400 <= exc.status_code < 500:
                self.budget.settle(token, rejected=True)
                raise
            raise BudgetStop("Server error; unknown charge reservation retained") from exc
        except Exception as exc:
            raise BudgetStop("Request interrupted; unknown charge reservation retained") from exc
        self.budget.settle(token, response)
        return response
