"""Policy adapter for any OpenAI-compatible chat-completions endpoint."""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable
from typing import Any
from urllib import error, request

from loguru import logger

from src.policies.code_execution import DEFAULT_MAX_TOKENS, SYSTEM_PROMPT, clean_action

Transport = Callable[[str, dict[str, Any], dict[str, str], float], dict[str, Any]]
_LOGPROB_PREFIX_LIMIT = 64
_TOP_LOGPROB_LIMIT = 20
_TOP_LOGPROB_TOKEN_LIMIT = 4
_REJECTED_CONTENT_LIMIT = 8_000


def _chat_completions_url(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("/chat/completions"):
        return endpoint
    return f"{endpoint}/chat/completions"


def _post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    body = json.dumps(payload).encode()
    req = request.Request(url, data=body, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read())
    except error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"model endpoint returned HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"could not reach model endpoint: {exc.reason}") from exc


def _text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"}
        ]
        if parts:
            return "".join(parts)
    raise ValueError("chat-completions response did not contain text content")


def _split_thinking(content: str) -> tuple[str | None, str]:
    """Separate Qwen's inline thinking before action parsing or execution.

    Some chat templates put ``<think>`` in the prompt, so generated text may
    begin with reasoning and only contain the closing tag. Ambiguous or
    unfinished tags must fail closed: ``clean_action`` can otherwise select a
    SUBMIT line inside reasoning or execute text from an unfinished thought.
    """
    text = content.lstrip()
    if "<think>" not in text and "</think>" not in text:
        return None, content

    if text.startswith("<think>"):
        remainder = text[len("<think>"):]
        close = remainder.find("</think>")
        if close < 0:
            raise ValueError("unfinished thinking block in model response")
        reasoning, action = remainder[:close], remainder[close + len("</think>"):]
    elif "<think>" not in text and "</think>" in text:
        # Qwen's opening marker may already be in the chat template prompt.
        reasoning, action = text.split("</think>", 1)
    else:
        raise ValueError("ambiguous thinking block in model response")

    if "<think>" in action or "</think>" in action or "<think>" in reasoning:
        raise ValueError("ambiguous thinking block in model response")
    if not action.strip():
        raise ValueError("thinking response did not contain a final action")
    return reasoning.strip(), action


def _reasoning_and_action(message: dict[str, Any]) -> tuple[str | None, str]:
    """Read current and legacy vLLM reasoning fields, then strip inline tags."""
    current = message.get("reasoning")
    legacy = message.get("reasoning_content")
    for value in (current, legacy):
        if value is not None and not isinstance(value, str):
            raise ValueError("chat-completions reasoning must be text")
    if current and legacy and current != legacy:
        raise ValueError("conflicting chat-completions reasoning fields")
    structured = current or legacy or None
    inline, action = _split_thinking(_text_content(message.get("content")))
    if structured and inline and structured.strip() != inline:
        raise ValueError("conflicting structured and inline reasoning")
    if not action.strip():
        raise ValueError("chat-completions response did not contain a final action")
    return (structured.strip() if structured else inline), action


def _finite_logprob(value: Any) -> float | None:
    if type(value) not in {int, float}:
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def _sampled_logprob_diagnostics(choice: dict[str, Any]) -> dict[str, Any] | None:
    """Bounded telemetry for generated tokens, never a correctness probability.

    Chat endpoints may return content logprobs for the whole generated stream,
    including thinking tokens. These cannot safely be aligned to a cleaned
    Python action or interpreted as P(answerable).
    """
    payload = choice.get("logprobs")
    content = payload.get("content") if isinstance(payload, dict) else None
    if not isinstance(content, list):
        return None

    values: list[float] = []
    prefix: list[dict[str, Any]] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        logprob = _finite_logprob(item.get("logprob"))
        if logprob is not None:
            values.append(logprob)
        if len(prefix) >= _LOGPROB_PREFIX_LIMIT:
            continue
        token = item.get("token")
        token_record: dict[str, Any] = {
            "token": token if isinstance(token, str) else None,
            "logprob": logprob,
        }
        alternatives = item.get("top_logprobs")
        if isinstance(alternatives, list) and len(prefix) < _TOP_LOGPROB_TOKEN_LIMIT:
            token_record["top_logprobs"] = [
                {"token": alternative["token"], "logprob": candidate}
                for alternative in alternatives[:_TOP_LOGPROB_LIMIT]
                if isinstance(alternative, dict)
                and isinstance(alternative.get("token"), str)
                and (candidate := _finite_logprob(alternative.get("logprob"))) is not None
            ]
        prefix.append(token_record)

    total = sum(values) if values else None
    if total is not None and not math.isfinite(total):
        total = None
    return {
        "schema_version": "sampled-content-logprobs-v1",
        "reported_token_count": len(content),
        "valid_logprob_count": len(values),
        "sum_logprob": total,
        "mean_logprob": total / len(values) if total is not None else None,
        "prefix": prefix,
        "prefix_truncated": len(content) > len(prefix),
        "scope": "provider generated content; not aligned to cleaned action or answer correctness",
    }


class OpenAICompatiblePolicy:
    """Use vLLM, Ollama, LM Studio, or a hosted OpenAI-compatible model.

    Configure it with constructor arguments or these environment variables:

    - ENVOY_MODEL_ENDPOINT (for example http://localhost:8000/v1)
    - ENVOY_MODEL_ID
    - ENVOY_MODEL_API_KEY (optional)
    - ENVOY_MODEL_EXTRA_JSON (optional request fields)
    """

    def __init__(
        self,
        endpoint: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
        timeout: float = 120.0,
        extra_body: dict[str, Any] | None = None,
        transport: Transport = _post_json,
        system_prompt: str | None = None,
    ) -> None:
        self.endpoint = endpoint or os.environ.get("ENVOY_MODEL_ENDPOINT", "")
        self.model = model or os.environ.get("ENVOY_MODEL_ID", "")
        self.api_key = api_key or os.environ.get("ENVOY_MODEL_API_KEY")
        if not self.endpoint or not self.model:
            raise ValueError("Set ENVOY_MODEL_ENDPOINT and ENVOY_MODEL_ID for openai_compatible")

        configured_extra = os.environ.get("ENVOY_MODEL_EXTRA_JSON")
        if extra_body is None and configured_extra:
            parsed = json.loads(configured_extra)
            if not isinstance(parsed, dict):
                raise ValueError("ENVOY_MODEL_EXTRA_JSON must contain a JSON object")
            extra_body = parsed

        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.extra_body = extra_body or {}
        self._transport = transport
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.history: list[dict[str, str]] = []
        self.last_reasoning: str | None = None
        self.last_logprob_diagnostics: dict[str, Any] | None = None
        # Shape-only response telemetry is useful when a provider returns
        # reasoning but no executable final action. Never retain raw failures.
        self.last_response_metadata: dict[str, Any] | None = None
        self.last_raw_response_content: str | None = None
        self.last_raw_response_truncated = False
        self._usage_requests: list[dict[str, int | float | bool | None]] = []
        self.config = {
            "backend": "openai_compatible",
            "endpoint": self.endpoint,
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

    def act(self, observation: str) -> str:
        self.last_reasoning = None
        self.last_logprob_diagnostics = None
        self.last_response_metadata = None
        self.last_raw_response_content = None
        self.last_raw_response_truncated = False
        self.history.append({"role": "user", "content": observation})
        messages = [{"role": "system", "content": self.system_prompt}, *self.history]
        payload = {
            **self.extra_body,
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        usage_record: dict[str, int | float | bool | None] = {
            "prompt_tokens": None,
            "completion_tokens": None,
            "cached_prompt_tokens": None,
            "duration_seconds": None,
            "failed": False,
        }
        # Record before dispatch so transport failures remain part of the
        # attempt count. Missing provider usage is unknown, never zero tokens.
        self._usage_requests.append(usage_record)
        started = time.monotonic()
        try:
            response = self._transport(
                _chat_completions_url(self.endpoint), payload, headers, self.timeout
            )
            usage = response.get("usage") if isinstance(response, dict) else None
            if isinstance(usage, dict):
                for name in ("prompt_tokens", "completion_tokens"):
                    value = usage.get(name)
                    if type(value) is int and value >= 0:
                        usage_record[name] = value
                details = usage.get("prompt_tokens_details")
                cached = details.get("cached_tokens") if isinstance(details, dict) else None
                prompt_tokens = usage_record["prompt_tokens"]
                if (
                    type(cached) is int
                    and cached >= 0
                    and (prompt_tokens is None or cached <= prompt_tokens)
                ):
                    usage_record["cached_prompt_tokens"] = cached
            try:
                choice = response["choices"][0]
                message = choice["message"]
            except (KeyError, IndexError, TypeError) as exc:
                raise ValueError("invalid chat-completions response shape") from exc

            if not isinstance(choice, dict) or not isinstance(message, dict):
                raise ValueError("invalid chat-completions response shape")
            finish_reason = choice.get("finish_reason")
            if not isinstance(finish_reason, str) and finish_reason is not None:
                finish_reason = "other"
            elif finish_reason not in {"stop", "length", "tool_calls", "content_filter", None}:
                finish_reason = "other"
            self.last_response_metadata = {
                "finish_reason": finish_reason,
                "visible_content_present": bool(message.get("content")),
                "reasoning_field_present": bool(
                    message.get("reasoning") or message.get("reasoning_content")
                ),
            }
            content = message.get("content")
            raw_content = (
                content if isinstance(content, str)
                else "".join(
                    item["text"] for item in content
                    if isinstance(item, dict)
                    and item.get("type") in {"text", "output_text"}
                    and isinstance(item.get("text"), str)
                ) if isinstance(content, list) else None
            )
            if raw_content is not None:
                self.last_raw_response_content = raw_content[:_REJECTED_CONTENT_LIMIT]
                self.last_raw_response_truncated = len(raw_content) > _REJECTED_CONTENT_LIMIT
            reasoning, raw_action = _reasoning_and_action(message)
            action = clean_action(raw_action)
            logprob_diagnostics = _sampled_logprob_diagnostics(choice)
        except Exception:
            usage_record["failed"] = True
            raise
        finally:
            usage_record["duration_seconds"] = time.monotonic() - started

        self.history.append({"role": "assistant", "content": raw_action})
        self.last_reasoning = reasoning
        self.last_logprob_diagnostics = logprob_diagnostics
        logger.debug(
            f"OpenAICompatiblePolicy ({self.model}) action ({len(action)} chars): {action[:100]}..."
        )
        return action

    def reset(self) -> None:
        self.history = []
        self.last_reasoning = None
        self.last_logprob_diagnostics = None
        self.last_response_metadata = None
        self.last_raw_response_content = None
        self.last_raw_response_truncated = False
        self._usage_requests = []

    def eval_metadata(self) -> dict[str, Any]:
        """Numeric usage and client timing; no prompts, endpoint, key, or error text.

        Observed totals sum only reported fields. Full totals require complete
        usage on every request and no failed requests. A failed response may
        still carry observed usage, but cannot establish complete attempt cost.
        Cached usage has its own coverage flags and is unknown when omitted.
        duration_seconds is client elapsed time for the full nonstreaming
        request, including response validation, not server timing or TTFT.
        """
        records = [dict(record) for record in self._usage_requests]
        observed_requests = sum(
            record["prompt_tokens"] is not None and record["completion_tokens"] is not None
            for record in records
        )
        failures = sum(bool(record["failed"]) for record in records)
        complete = bool(records) and observed_requests == len(records) and not failures
        cached_observed_requests = sum(
            record["cached_prompt_tokens"] is not None for record in records
        )
        cached_complete = (
            bool(records) and cached_observed_requests == len(records) and not failures
        )
        result = {
            "request_count": len(records),
            "failed_request_count": failures,
            "usage_observed_request_count": observed_requests,
            "usage_missing_request_count": len(records) - observed_requests,
            "usage_complete": complete,
            "cached_usage_observed_request_count": cached_observed_requests,
            "cached_usage_missing_request_count": len(records) - cached_observed_requests,
            "cached_usage_complete": cached_complete,
            "requests": records,
        }
        for name in ("prompt_tokens", "completion_tokens"):
            values = [record[name] for record in records if record[name] is not None]
            observed = sum(values) if values else None
            result[f"observed_{name}"] = observed
            result[name] = observed if complete else None
        cached_values = [
            record["cached_prompt_tokens"]
            for record in records
            if record["cached_prompt_tokens"] is not None
        ]
        observed_cached = sum(cached_values) if cached_values else None
        result["observed_cached_prompt_tokens"] = observed_cached
        result["cached_prompt_tokens"] = observed_cached if cached_complete else None
        return {"token_usage": result}
