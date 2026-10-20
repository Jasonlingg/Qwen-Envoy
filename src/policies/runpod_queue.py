"""Bounded, opt-in adapter from chat completions to Runpod's queued vLLM API.

One process owns an exclusive audit file and a shared reservation budget. A
submission with an uncertain outcome is never retried. Polling an existing job
does not create another inference request. This is not a provider dollar cap.
"""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from urllib import error, request


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class RunpodQueueTransport:
    def __init__(self, endpoint_id, api_key, tokenizer, audit_path: Path, *,
                 requests_max=12, wait_seconds=900, http=None):
        if not re.fullmatch(r"[a-z0-9]+", endpoint_id) or not api_key:
            raise ValueError("Runpod endpoint id and credential required")
        if type(requests_max) is not int or not 1 <= requests_max <= 12:
            raise ValueError("request budget must be 1..12")
        if not 1 <= wait_seconds <= 900:
            raise ValueError("queue deadline must be 1..900 seconds")
        self.base = f"https://api.runpod.ai/v2/{endpoint_id}"
        self._key, self.tokenizer = api_key, tokenizer
        self.audit = audit_path.open("x", buffering=1)
        self.lock = threading.Lock()
        self.requests_max = requests_max
        self.remaining, self.wait_seconds = requests_max, wait_seconds
        self._http = http or self._request
        self._write({"event": "budget", "requests": requests_max,
                     "input_tokens_reserved_max": requests_max * 7168,
                     "output_tokens_reserved_max": requests_max * 1024})

    def _write(self, event):
        self.audit.write(json.dumps(event, ensure_ascii=False) + "\n")
        self.audit.flush()

    def _request(self, method, path, payload=None):
        opener = request.build_opener(request.ProxyHandler({}), _NoRedirect())
        body = None if payload is None else json.dumps(payload).encode()
        if body is not None and len(body) > 1_000_000:
            raise ValueError("Runpod request too large")
        req = request.Request(self.base + path, data=body, method=method,
                              headers={"Authorization": "Bearer " + self._key,
                                       "Content-Type": "application/json",
                                       "User-Agent": "Qwen-Envoy/0.1 (Runpod queue client)"})
        try:
            with opener.open(req, timeout=30) as response:
                data = response.read(1_000_001)
        except error.HTTPError as exc:
            status = exc.code
            exc.close()
            raise RuntimeError(f"Runpod HTTP {status}; no retry") from None
        except OSError:
            raise RuntimeError("Runpod connection failed; no retry") from None
        if len(data) > 1_000_000:
            raise ValueError("Runpod response too large")
        result = json.loads(data)
        if not isinstance(result, dict):
            raise ValueError("Runpod response must be an object")
        return result

    def __call__(self, url, payload, headers, timeout):
        # The caller's URL and auth cannot redirect this transport elsewhere.
        allowed = {"model", "messages", "max_tokens", "temperature", "top_p", "seed",
                   "chat_template_kwargs", "stream", "n"}
        if (set(payload) - allowed or payload.get("model") != "envoy"
                or payload.get("stream", False) is not False
                or payload.get("n", 1) != 1 or payload.get("max_tokens") != 1024
                or payload.get("chat_template_kwargs") != {"enable_thinking": False}):
            raise ValueError("unsupported reader request")
        with self.lock:
            if self.remaining <= 0:
                raise RuntimeError("Qwen request budget exhausted")
            tokens = self.tokenizer.apply_chat_template(
                payload["messages"], tokenize=True, add_generation_prompt=True,
                enable_thinking=False)
            if len(tokens) > 7168:
                raise ValueError("reader prompt exceeds 7168-token input limit")
            self.remaining -= 1
            number = self.requests_max - self.remaining
            self._write({"event": "reserved", "request": number, "prompt_tokens": len(tokens),
                         "input_reservation": 7168, "output_reservation": 1024,
                         "payload": payload})
        job_id = None
        try:
            started = time.monotonic()
            job = self._http("POST", "/run", {"input": {
                "openai_route": "/v1/chat/completions", "openai_input": payload},
                "policy": {"executionTimeout": 120000, "ttl": 900000}})
            job_id = job.get("id")
            if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", job_id):
                raise ValueError("missing or invalid Runpod job id; no retry")
            while job.get("status") in {"IN_QUEUE", "IN_PROGRESS"}:
                if time.monotonic() - started >= self.wait_seconds:
                    raise TimeoutError("Runpod queue deadline exceeded")
                time.sleep(2)
                job = self._http("GET", "/status/" + job_id)
            if job.get("status") != "COMPLETED":
                raise RuntimeError("Runpod job did not complete")
            output = job.get("output")
            if isinstance(output, list) and len(output) == 1:
                output = output[0]
            if (not isinstance(output, dict) or "error" in output
                    or output.get("model") != "envoy" or not output.get("choices")):
                raise ValueError("Runpod returned invalid adapter chat output")
            with self.lock:
                self._write({"event": "completed", "request": number, "job_id": job_id,
                             "execution_time_ms": job.get("executionTime"),
                             "delay_time_ms": job.get("delayTime"), "response": output})
            return output
        except Exception as exc:
            if job_id:
                try:
                    self._http("POST", "/cancel/" + job_id, {})
                except Exception:
                    pass
            with self.lock:
                self._write({"event": "failed", "request": number, "job_id": job_id,
                             "error_type": type(exc).__name__, "reservation_retained": True})
            raise
