"""One fail-closed inference budget shared by search control and paper reading."""
from __future__ import annotations

import threading


class BudgetExhaustedError(RuntimeError):
    """No further inference may be dispatched for this investigation."""


class ExperimentBudget:
    """Reserve worst-case tokens before dispatch, including failed requests.

    The configured local tokenizer must match the served Qwen chat template.
    Provider max_tokens enforcement is assumed; this is not a dollar ceiling.
    No retry or refund occurs after a transport failure or uncertain outcome.
    """

    INPUT_MAX = 7168
    OUTPUT_MAX = 1024

    def __init__(self, tokenizer, transport, *, requests_max=18, tokens_max=147456):
        if type(requests_max) is not int or not 1 <= requests_max <= 18:
            raise ValueError("requests_max must be 1..18")
        if type(tokens_max) is not int or not 8192 <= tokens_max <= 147456:
            raise ValueError("tokens_max must be 8192..147456")
        self.tokenizer, self.transport = tokenizer, transport
        self.requests_max, self.tokens_max = requests_max, tokens_max
        self.records = []
        self._lock = threading.Lock()

    @property
    def remaining(self):
        return min(self.requests_max, self.tokens_max // 8192) - len(self.records)

    def __call__(self, url, payload, headers, timeout):
        if (payload.get("max_tokens") != self.OUTPUT_MAX
                or payload.get("chat_template_kwargs") != {"enable_thinking": False}
                or payload.get("stream", False) is not False or payload.get("n", 1) != 1):
            raise ValueError("experiment requires bounded, nonstreaming Qwen requests")
        tokens = self.tokenizer.apply_chat_template(
            payload["messages"], tokenize=True, add_generation_prompt=True,
            enable_thinking=False, return_dict=False)
        if (not isinstance(tokens, list) or any(type(token) is not int for token in tokens)
                or len(tokens) > self.INPUT_MAX):
            raise ValueError("experiment prompt exceeds the 7168-token limit or tokenizer failed")
        with self._lock:
            if self.remaining <= 0:
                raise BudgetExhaustedError("shared experiment inference budget exhausted")
            record = {"request": len(self.records) + 1, "prompt_tokens_local": len(tokens),
                      "tokens_reserved": 8192, "status": "dispatched", "usage": None}
            self.records.append(record)
        try:
            response = self.transport(url, payload, headers, timeout)
            usage = response.get("usage") if isinstance(response, dict) else None
            if isinstance(usage, dict):
                record["usage"] = {key: usage.get(key) if type(usage.get(key)) is int
                                   and usage[key] >= 0 else None
                                   for key in ("prompt_tokens", "completion_tokens")}
            record["status"] = "returned"
            return response
        except Exception:
            record["status"] = "failed_or_uncertain"
            raise

    def summary(self):
        records = [dict(record) for record in self.records]
        complete = bool(records) and all(
            record["status"] == "returned" and record["usage"] is not None
            and all(value is not None for value in record["usage"].values())
            for record in records)
        return {"requests_max": self.requests_max, "tokens_max": self.tokens_max,
                "requests_dispatched": len(records), "requests_remaining": self.remaining,
                "tokens_reserved": len(records) * 8192, "usage_complete": complete,
                "records": records, "provider_dollar_cap": False}
