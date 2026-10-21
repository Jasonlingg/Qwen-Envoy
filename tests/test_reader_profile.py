"""Bounded worker/client integration with scripted data; no model or cloud calls."""

from __future__ import annotations

import asyncio
import json
import threading
from copy import deepcopy
from datetime import datetime, timezone

import pytest

from scripts import profile_two_readers as cli
from src.eval.reader_profile import ProfileQueueClient, assess_profile, run_profile
from src.eval.reader_profile_worker import (
    GAUGES,
    PHASES,
    ProfileHook,
    canonical_hash,
    parse_metrics,
)


def metrics(completed=0, running=0):
    return {**{k: 0.0 for k in GAUGES}, "vllm:num_requests_running": float(running),
            **{phase + "_sum": completed * 0.1 for phase in PHASES},
            **{phase + "_count": float(completed) for phase in PHASES}}


def inputs():
    return {key: {"model": "envoy", "messages": [{"role": "user", "content": key}],
                  "max_tokens": 1024} for key in ("sample_a", "sample_b")}


def plan(payloads):
    return {"session": "a" * 32,
            "payload_hashes": {k: canonical_hash(v) for k, v in payloads.items()},
            "prompt_tokens": {k: 10 for k in payloads}}


def test_metric_parser_sums_labels_and_rejects_missing_or_nonfinite_values():
    values = metrics()
    lines = [f'{key}{{model_name="envoy",engine="0"}} {value}' for key, value in values.items()]
    lines.append('vllm:num_requests_running{model_name="envoy",engine="1"} 2')
    assert parse_metrics("\n".join(lines))["vllm:num_requests_running"] == 2
    for body in ("", "\n".join(lines).replace('engine="1"} 2', 'engine="1"} NaN')):
        with pytest.raises(ValueError):
            parse_metrics(body)


def fixture_hook():
    payloads = inputs()
    completed = []
    hook = ProfileHook("a" * 32, plan(payloads)["payload_hashes"],
        metrics=lambda: metrics(len(completed), hook.active),
        memory=lambda: {"used_mib": 18000, "total_mib": 24000})
    async def original(job):
        assert "envoy_profile" not in job["input"]
        await asyncio.sleep(0.12)
        completed.append(1)
        yield {"model": "envoy", "choices": [{"finish_reason": "stop",
                "message": {"content": "untrusted output; never execute this"}}],
               "usage": {"prompt_tokens": 10, "completion_tokens": 2}}
    return hook, hook.wrap(original), payloads, completed


def test_worker_pins_session_inputs_and_caps_model_requests():
    hook, handler, payloads, completed = fixture_hook()
    def invoke(operation, payload=None):
        data = {"envoy_profile": {"session": "a" * 32, "operation": operation}}
        if payload is not None:
            data["envoy_profile"]["sample_id"] = "sample_a"
            data.update(openai_route="/v1/chat/completions", openai_input=payload)
        async def collect():
            return [row async for row in handler({"input": data})]
        return asyncio.run(collect())[0]
    assert "preflight" in invoke("preflight")["profile"]
    for _ in range(10):
        assert "response" in invoke("infer", payloads["sample_a"])
    assert "error" in invoke("infer", payloads["sample_a"])
    assert hook.reserved == len(completed) == 10
    other, handle, _, called = fixture_hook()
    async def wrong():
        return [row async for row in handle({"input": {"envoy_profile": {
            "session": "b" * 32, "operation": "infer"}}})]
    assert "error" in asyncio.run(wrong())[0]
    assert not called and other.reserved == 0


def test_real_hook_client_schedule_and_durable_request_reservations(tmp_path, monkeypatch):
    for key, value in cli.CONFIG.items():
        monkeypatch.setenv(key, value)
    hook, handler, payloads, completed = fixture_hook()
    counter, lock = [0], threading.Lock()
    def http(method, path, body=None):
        assert method == "POST" and path == "/run"
        with lock:
            counter[0] += 1
            number = counter[0]
        async def collect():
            return [item async for item in handler(body)]
        return {"id": f"job-{number}", "status": "COMPLETED", "output": asyncio.run(collect())}
    client = ProfileQueueClient("test", "secret", plan(payloads), payloads,
                                tmp_path / "profile.requests.jsonl", http=http)
    snapshots = []
    try:
        result = run_profile(client, list(payloads), lambda x: snapshots.append(deepcopy(x)))
    finally:
        client.audit.close()
    assert result["status"] == "completed_pending_comparability_check"
    assert client.model_calls == hook.reserved == len(completed) == 10
    assert client.admin_calls == hook.admin_calls == 9 and counter[0] == 19
    assert len(result["warmups"]) == 2 and len(result["batches"]) == 4
    assert result["tokens_reserved"] == 81920
    assessment = assess_profile(result)
    assert any("polling uncertainty" in reason for reason in assessment["reasons"])
    assert all(b["engine_peak_running_sampled"] >= 2 for b in result["batches"]
               if b["mode"] == "concurrent")
    rows = [json.loads(line) for line in
            (tmp_path / "profile.requests.jsonl").read_text().splitlines()]
    reserved = [r for r in rows if r["event"] == "reserved" and r["kind"] == "model"]
    assert len(reserved) == 10 and sum(r["input_reservation"] for r in reserved) == 71680
    assert "secret" not in (tmp_path / "profile.requests.jsonl").read_text()


@pytest.mark.parametrize("failure", ["uncertain", "changed_worker", "bad_usage", "worker_error"])
def test_uncertain_or_invalid_response_consumes_allowance_and_stops(tmp_path, failure):
    payloads, calls = inputs(), []
    def http(method, path, body=None):
        calls.append((method, path))
        if failure == "uncertain":
            raise TimeoutError("uncertain POST outcome")
        if failure == "worker_error":
            return {"id": "job-1", "status": "COMPLETED",
                    "output": {"error": {"message": "required engine telemetry missing"}}}
        response = {"model": "envoy", "choices": [{}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 2048}}
        return {"id": "job-1", "status": "COMPLETED", "output": {
            "profile": {"boot_id": "other" if failure == "changed_worker" else "known"},
            "response": response}}
    client = ProfileQueueClient("test", "secret", plan(payloads), payloads,
                                tmp_path / "audit.jsonl", http=http)
    client.boot_id = "known"
    try:
        with pytest.raises((TimeoutError, ValueError)):
            client.call("infer", "sample_a")
        with pytest.raises(RuntimeError, match="stopped"):
            client.call("infer", "sample_a")
        assert client.model_calls == 1
        assert sum(method == "POST" and path == "/run" for method, path in calls) == 1
        if failure != "uncertain":
            rows = [json.loads(line) for line in
                    (tmp_path / "audit.jsonl").read_text().splitlines()]
            assert any(row["event"] == "returned" and "output" in row for row in rows)
    finally:
        client.audit.close()


def test_concurrent_failure_preserves_other_response_and_stops_later_batches():
    class Client:
        model_calls = 0
        def call(self, operation, sample=None):
            if operation == "preflight":
                return {"profile": {"declared_config": cli.CONFIG}}
            if operation == "begin":
                return {"profile": {}}
            if operation == "end":
                return {"profile": {"before": {"metrics": metrics()},
                                    "after": {"metrics": metrics(2)}, "errors": []}}
            self.model_calls += 1
            if self.model_calls == 6:
                raise TimeoutError("second concurrent response missing")
            return {"sample_id": sample, "response": {"saved": True}}
    snapshots = []
    result = run_profile(Client(), ["a", "b"], lambda x: snapshots.append(deepcopy(x)))
    assert result["status"] == "incomplete" and len(result["batches"]) == 2
    assert len(result["batches"][-1]["requests"]) == 1
    assert result["batches"][-1]["errors"] == ["TimeoutError"]
    assert snapshots[-1] == result


def test_preparation_recounts_pinned_inputs_and_rejects_code_changes(tmp_path, monkeypatch):
    class Tokenizer:
        def __init__(self):
            self.count = 0
        def apply_chat_template(self, *args, **kwargs):
            self.count += 1
            return list(range((1400, 2043)[self.count - 1]))
    output = tmp_path / "prepared"
    prepared = cli.prepare(output, Tokenizer())
    assert prepared["limits"] == cli.LIMITS and not (output / "authorization.json").exists()
    compile((output / "entrypoint.py").read_text(), "profile-entrypoint", "exec")
    assert cli.checked(output)[0] == prepared
    saved = (output / "plan.json").read_text()
    modified = json.loads(saved)
    modified["limits"]["model_requests"] = 11
    (output / "plan.json").write_text(json.dumps(modified))
    with pytest.raises(ValueError, match="changed"):
        cli.checked(output)
    (output / "plan.json").write_text(saved)
    (output / "entrypoint.py").write_text("raise RuntimeError('changed')")
    with pytest.raises(ValueError, match="changed"):
        cli.checked(output)


def test_missing_budget_or_stale_price_cannot_authorize_gpu(tmp_path):
    (tmp_path / "plan.json").write_text("{}")
    with pytest.raises(FileNotFoundError):
        cli.authorization(tmp_path, {})
    record = {"approved": True, "plan_sha256": cli.sha(tmp_path / "plan.json"),
              "limits": cli.LIMITS, "budget_usd": 1, "catalog_gpu_hourly_usd": 0.69,
              "user_authorization": "Synthetic unit-test declaration only", "attempts": 1,
              "price_checked_at": datetime.now(timezone.utc).isoformat()}
    (tmp_path / "authorization.json").write_text(json.dumps(record))
    assert cli.authorization(tmp_path, {}) == record
    record["budget_usd"] = True
    (tmp_path / "authorization.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="authorization"):
        cli.authorization(tmp_path, {})


def test_one_replacement_binds_original_without_resetting_its_marker(tmp_path, monkeypatch):
    original = tmp_path / "original"
    original.mkdir()
    (original / "plan.json").write_text('{"original":true}')
    marker = tmp_path / "consumed"
    marker.write_text(str(original) + "\n")
    outcome = tmp_path / "outcome.json"
    outcome.write_text(json.dumps({"shutdown_verified": True,
                                   "plan_sha256": cli.sha(original / "plan.json")}))
    replacement = tmp_path / "replacement"
    monkeypatch.setattr(cli, "MARKER", marker)
    monkeypatch.setattr(cli, "PRIOR_OUTCOME", outcome)
    monkeypatch.setattr(cli, "REPLACEMENT_MARKER", replacement)
    saved = marker.read_bytes()
    approved = {"replacement_attempts": 1, "replacement_of": {
        "outcome_sha256": cli.sha(outcome), "marker_sha256": cli.sha(marker),
        "plan_sha256": cli.sha(original / "plan.json")}}
    assert cli.attempt_marker({}) == marker
    with cli.attempt_marker(approved).open("x") as stream:
        stream.write("one newly authorized attempt")
    with pytest.raises(FileExistsError):
        cli.attempt_marker(approved).open("x")
    assert marker.read_bytes() == saved
    approved["replacement_attempts"] = 2
    with pytest.raises(ValueError, match="replacement"):
        cli.attempt_marker(approved)
    approved["replacement_attempts"] = 1
    (original / "plan.json").write_text("changed")
    with pytest.raises(ValueError, match="replacement"):
        cli.attempt_marker(approved)


def test_telemetry_failure_stops_before_more_model_requests():
    class Client:
        model_calls = 0
        def call(self, operation, sample=None):
            if operation == "preflight":
                return {"profile": {"declared_config": cli.CONFIG}}
            if operation == "end":
                return {"profile": {"errors": ["metrics unavailable"]}}
            if operation == "begin":
                return {"profile": {}}
            self.model_calls += 1
            return {"sample_id": sample}
    client = Client()
    result = run_profile(client, ["a", "b"], lambda x: None)
    assert result["status"] == "incomplete" and client.model_calls == 4
    assert len(result["batches"]) == 1 and "telemetry failed" in result["error"]


def test_uncertain_entrypoint_patch_restores_parked_original(tmp_path, monkeypatch):
    original = ["python3", "-u", "-c",
                (cli.ROOT / "scripts/runpod_qwen_bootstrap.py").read_text()]
    state = {"entrypoint": original, "workers": {"min": 0, "max": 0},
             "env": cli.CONFIG, "image":
             "registry.runpod.net/runpod-workers-worker-vllm-main-dockerfile:b7212acc6"}
    changes, controls = [], []
    def api(_self, method, path, body=None):
        assert path == ""
        if method == "PATCH":
            changes.append(deepcopy(body))
            state.update(deepcopy(body))
            if len(changes) == 1:
                raise TimeoutError("PATCH outcome uncertain")
        return deepcopy(state)
    monkeypatch.setattr(cli.RunpodQueueTransport, "_request", api)
    monkeypatch.setattr(cli, "MARKER", tmp_path / "live-attempt")
    (tmp_path / "plan.json").write_text("{}")
    (tmp_path / "entrypoint.py").write_text("# scripted instrumented worker")
    with pytest.raises(TimeoutError, match="uncertain"):
        cli.guarded_run(tmp_path, {}, {}, key="synthetic-secret",
            control=lambda *args: controls.append(args),
            supervisor=lambda *args, **kwargs: pytest.fail("activation after uncertain PATCH"))
    assert state["entrypoint"] == original and len(changes) == 2
    assert controls == [("update", cli.ENDPOINT, "--workers-min", "0", "--workers-max", "0")]
    assert json.loads((tmp_path / "entrypoint-restoration.json").read_text())["restored"]
