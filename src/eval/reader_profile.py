"""One fixed-input serial/concurrent diagnostic; no model-authored code execution."""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.eval.reader_profile_worker import PHASES, canonical_hash
from src.policies.runpod_queue import RunpodQueueTransport

SCHEDULE = (("serial", 1), ("concurrent", 1), ("concurrent", 2), ("serial", 2))


class ProfileQueueClient:
    def __init__(self, endpoint, key, plan, payloads, audit: Path, *, http=None):
        self.base, self._key = f"https://api.runpod.ai/v2/{endpoint}", key
        self.plan, self.payloads = plan, payloads
        self.lock, self.boot_id = threading.Lock(), None
        self.model_calls, self.admin_calls, self.failed = 0, 0, False
        self.audit = audit.open("x")
        self.http = http or (lambda method, path, body=None:
                            RunpodQueueTransport._request(self, method, path, body))

    def append(self, data):
        with self.lock:
            self.audit.write(json.dumps(data, ensure_ascii=False) + "\n")
            self.audit.flush()
            os.fsync(self.audit.fileno())

    def call(self, operation, sample_id=None):
        profile = {"session": self.plan["session"], "operation": operation}
        body = {"envoy_profile": profile}
        model = operation == "infer"
        with self.lock:
            if self.failed:
                raise RuntimeError("profile stopped after an uncertain or failed request")
            if model:
                if self.model_calls >= 10:
                    raise ValueError("ten-request model allowance exhausted")
                payload = self.payloads[sample_id]
                if canonical_hash(payload) != self.plan["payload_hashes"][sample_id]:
                    raise ValueError("profile input hash mismatch")
                profile["sample_id"] = sample_id
                body.update(openai_route="/v1/chat/completions", openai_input=payload)
                self.model_calls += 1
                number = self.model_calls
            else:
                if operation not in {"preflight", "begin", "end"} or self.admin_calls >= 9:
                    raise ValueError("telemetry operation or budget invalid")
                self.admin_calls += 1
                number = self.admin_calls
        # Every submission, including telemetry, is durably recorded before POST.
        self.append({"event": "reserved", "kind": "model" if model else "telemetry",
                     "request": number, "input_reservation": 7168 if model else 0,
                     "output_reservation": 1024 if model else 0, "operation": operation,
                     "sample_id": sample_id})
        job_id = None
        started = time.monotonic()
        try:
            job = self.http("POST", "/run", {"input": body,
                            "policy": {"executionTimeout": 120000, "ttl": 900000}})
            job_id = job.get("id")
            if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", job_id):
                raise ValueError("missing profile job ID; do not resubmit")
            self.append({"event": "submitted", "job_id": job_id, "kind": "model" if model
                         else "telemetry", "request": number})
            while job.get("status") in {"IN_QUEUE", "IN_PROGRESS"}:
                if time.monotonic() - started > 120:
                    raise TimeoutError("profile job deadline exceeded")
                time.sleep(0.2)
                job = self.http("GET", "/status/" + job_id)
            if job.get("status") != "COMPLETED":
                raise RuntimeError("profile job failed")
            output = job.get("output")
            if isinstance(output, list) and len(output) == 1:
                output = output[0]
            self.append({"event": "returned", "job_id": job_id,
                         "kind": "model" if model else "telemetry", "output": output})
            if not isinstance(output, dict) or "error" in output:
                raise ValueError("profile worker rejected request")
            boot = output.get("profile", {}).get("boot_id")
            with self.lock:
                if operation == "preflight" and self.boot_id is None:
                    self.boot_id = boot
                if not isinstance(boot, str) or not boot or boot != self.boot_id:
                    raise ValueError("profile worker changed; comparison stopped")
            result = {"sample_id": sample_id, "client_seconds": time.monotonic() - started,
                      "queue_delay_ms": job.get("delayTime"),
                      "execution_time_ms": job.get("executionTime"), **output}
            if model:
                response = result["response"]
                usage = response["usage"]
                if (response.get("model") != "envoy" or not response.get("choices")
                        or any(type(usage.get(k)) is not int or not 1 <= usage[k] <= limit
                               for k, limit in (("prompt_tokens", 7168),
                                                ("completion_tokens", 1024)))
                        or usage["prompt_tokens"] != self.plan["prompt_tokens"][sample_id]):
                    raise ValueError("profile response identity/usage invalid")
            self.append({"event": "completed", "job_id": job_id,
                         "kind": "model" if model else "telemetry", "result": result})
            return result
        except BaseException:
            with self.lock:
                self.failed = True
            self.append({"event": "failed", "job_id": job_id,
                         "kind": "model" if model else "telemetry", "reservation_retained": True})
            if job_id:
                try:
                    self.http("POST", "/cancel/" + job_id, {})
                except Exception:
                    pass
            raise


def run_profile(client, sample_ids, save):
    result = {"status": "incomplete", "kind": "fixed-input serving diagnostic",
              "warmups": [], "batches": [], "limitations": [
                  "Two rounds, not tail latency or an independent agent benchmark.",
                  "Fixed inputs do not measure responses to feedback or useful-answer quality.",
                  "Memory snapshots are not peak device memory; gauges are sampled at 100 ms.",
                  "Telemetry sampling overhead and queue polling affect measured wall time."]}
    save(result)
    try:
        result["preflight"] = client.call("preflight")
        config = result["preflight"]["profile"]["declared_config"]
        expected = {"MAX_NUM_SEQS": "2", "MAX_CONCURRENCY": "2",
                    "MAX_MODEL_LEN": "8192", "ENABLE_PREFIX_CACHING": "false"}
        if config != expected:
            raise ValueError("profile needs the pinned two-sequence, cache-disabled configuration")
        for sample in sample_ids:
            result["warmups"].append(client.call("infer", sample))
            save(result)
        for mode, repeat in SCHEDULE:
            before = client.call("begin")["profile"]
            batch = {"mode": mode, "repeat": repeat, "before": before, "requests": []}
            result["batches"].append(batch)
            save(result)
            started = time.monotonic()
            if mode == "serial":
                for sample in sample_ids:
                    batch["requests"].append(client.call("infer", sample))
            else:
                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures = [pool.submit(client.call, "infer", sample) for sample in sample_ids]
                    errors = []
                    for future in futures:
                        try:
                            batch["requests"].append(future.result())
                        except Exception as exc:
                            errors.append(type(exc).__name__)
                    save(result)  # Keep the other response when one member fails.
                    if errors:
                        batch["errors"] = errors
                        raise RuntimeError("concurrent batch incomplete; no retry")
            batch["pair_seconds"] = time.monotonic() - started
            batch["telemetry"] = client.call("end")["profile"]
            save(result)
            telemetry = batch["telemetry"]
            if telemetry["errors"] or any(
                    telemetry["after"]["metrics"][phase + "_count"]
                    - telemetry["before"]["metrics"][phase + "_count"] != 2
                    for phase in PHASES):
                raise ValueError("batch telemetry failed or does not cover exactly two requests")
        result["status"] = "completed_pending_comparability_check"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"[:500]
    finally:
        result["model_requests_reserved"] = client.model_calls
        result["tokens_reserved"] = client.model_calls * 8192
        save(result)
    return result


def assess_profile(result):
    reasons = []
    if result["status"] != "completed_pending_comparability_check":
        return {"comparable": False, "reasons": ["incomplete profile"], "rounds": []}
    signatures, rounds = {}, []
    for batch in result["batches"]:
        telemetry = batch["telemetry"]
        if telemetry["errors"]:
            reasons.append("telemetry sampler failed")
        before, after = telemetry["before"]["metrics"], telemetry["after"]["metrics"]
        for phase in PHASES:
            if after[phase + "_count"] - before[phase + "_count"] != 2:
                reasons.append("phase telemetry does not cover exactly two requests")
        batch["engine_peak_running_sampled"] = max(
            row["metrics"]["vllm:num_requests_running"] for row in
            [telemetry["before"], *telemetry["samples"], telemetry["after"]])
        if batch["mode"] == "concurrent" and batch["engine_peak_running_sampled"] < 2:
            reasons.append("same-engine overlap was not observed")
        for row in batch["requests"]:
            response = row["response"]
            signature = canonical_hash({"choices": response["choices"], "usage": response["usage"]})
            previous = signatures.setdefault(row["sample_id"], signature)
            if previous != signature:
                reasons.append("outputs, token counts or finish reasons changed")
        if not math.isfinite(batch["pair_seconds"]) or batch["pair_seconds"] <= 0:
            reasons.append("invalid pair duration")
    for repeat in (1, 2):
        pair = {b["mode"]: b["pair_seconds"] for b in result["batches"] if b["repeat"] == repeat}
        rounds.append({"repeat": repeat, **pair,
                       "serial_over_concurrent": pair["serial"] / pair["concurrent"],
                       "absolute_difference_seconds": pair["serial"] - pair["concurrent"]})
        if abs(pair["serial"] - pair["concurrent"]) <= 0.4:
            reasons.append("difference does not exceed the 0.4-second polling uncertainty margin")
    return {"comparable": not reasons, "reasons": sorted(set(reasons)), "rounds": rounds}
