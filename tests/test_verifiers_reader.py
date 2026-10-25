"""Real VF 0.4 interception + frozen Docker controls; no paid or model calls."""

from __future__ import annotations

import asyncio
import copy
import importlib.metadata
import json
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

try:
    version = importlib.metadata.version("verifiers")
except importlib.metadata.PackageNotFoundError:
    version = None
if version != "0.4.0":
    pytest.skip("optional VF 0.4 adapter: use requirements-verifiers.txt", allow_module_level=True)

import verifiers.v1 as vf

from src.eval.reader_correction_replay import original_matches
from src.eval.reader_improvement import load, write
from src.training.verifiers_adapter import (
    PLUGIN,
    InterceptTransport,
    ReaderHarness,
    ReaderHarnessConfig,
    ReaderTasksetConfig,
    local_agent_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONTROLS = ROOT / "examples/verifiers-port/controls"


@contextmanager
def scripted_provider(control, *, fenced=False, fail_at=None):
    """A loopback fixture, not an inference endpoint or sampled training data."""
    fixture = load(control)
    payloads = copy.deepcopy(fixture["payloads"])
    if fenced:
        # The frozen policy keeps raw response text in its conversation history,
        # while only the cleaned Python is handed to Docker.
        for payload in payloads[1:]:
            payload["messages"][2]["content"] = "```python\n" + fixture["responses"][0] + "\n```"
    received, differences = [], []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            i = len(received)
            received.append(body)
            matches = i < len(payloads) and body == payloads[i]
            if not matches:
                differences.append(
                    {
                        "index": i,
                        "actual": body,
                        "expected": payloads[i] if i < len(payloads) else None,
                    }
                )
            status = 400 if not matches else 503 if i == fail_at else 200
            if status == 200:
                raw = fixture["responses"][i]
                if fenced and i == 0:
                    raw = "```python\n" + raw + "\n```"
                response = {
                    "id": f"scripted-control-{i}",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "envoy",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": raw},
                            "finish_reason": "stop",
                        }
                    ],
                }
            else:
                response = {
                    "error": {"message": "scripted fixture failure", "type": "fixture_error"}
                }
            data = json.dumps(response).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", received, differences
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def tasks():
    return list(vf.load_taskset(ReaderTasksetConfig()))


def test_native_plugin_discovers_only_training_tasks_without_gold_prompt_fields():
    selected = tasks()
    assert len(selected) == 82
    assert vf.default_harness_id(PLUGIN) == PLUGIN
    assert len({t.key for t in selected}) == 82
    for task in selected:
        assert "qasper_train_" in task.key
        assert not {"answer", "references", "keys", "claims"} & task.data.model_dump().keys()


def test_native_environment_defaults_to_local_controller_without_cloud_provisioning():
    config = vf.env_config_type(PLUGIN)(taskset=ReaderTasksetConfig())
    env = vf.load_environment(config)
    assert isinstance(env, vf.Env)
    assert isinstance(config.agent.runtime, vf.SubprocessConfig)
    assert config.agent.network_notice is False


@pytest.mark.parametrize("index,fenced", [(0, False), (1, False), (2, False), (2, True)])
def test_native_agent_replays_saved_docker_control_exactly(index, fenced, tmp_path):
    task = tasks()[index]
    control = CONTROLS / (task.key + ".json")
    if not control.exists():
        pytest.fail(
            "Saved controls missing; see docs/VERIFIERS_PORT.md for required local receipts"
        )
    original = load(control)
    with scripted_provider(control, fenced=fenced) as (url, received, differences):
        config = local_agent_config(
            url, tmp_path / "episodes", model_identity=original["packet"]["model_identity"]
        )

        async def run():
            async with vf.Agent(config) as agent:
                return await agent.run(task)

        trace = asyncio.run(run())
        write(tmp_path / "trace.json", trace.to_record())
        write(tmp_path / "payload-differences.json", differences)
        assert not differences, differences
        assert trace.ok, trace.errors
        assert trace.reward == 1.0
        assert len(trace.calls) == len(received) == 3
        assert trace.agent.trainable is False
        assert all(not node.token_ids and not node.logprobs for node in trace.nodes)
        assert original_matches(trace.state.episode["packet"], original["packet"])
        assert load(Path(trace.info["envoy"]["output"]) / "cleanup.json")["all_absent"]
        if fenced:
            assert any("```python" in str(node.message.content) for node in trace.nodes)
            assert "```python" in received[1]["messages"][2]["content"]


def test_provider_failure_is_failed_trace_with_no_numeric_reward_and_saved_evidence(tmp_path):
    task = tasks()[0]
    with scripted_provider(CONTROLS / (task.key + ".json"), fail_at=1) as (
        url,
        received,
        differences,
    ):

        async def run():
            async with vf.Agent(local_agent_config(url, tmp_path / "episodes")) as agent:
                return await agent.run(task)

        trace = asyncio.run(run())
    write(tmp_path / "failed-trace.json", trace.to_record())
    assert not differences
    assert not trace.ok and trace.errors
    assert not trace.rewards or all(value is None for value in trace.rewards.values())
    assert trace.agent.trainable is False
    assert len(received) == 2  # No hidden provider or rollout retries.
    episode = trace.state.episode
    assert not episode["infrastructure_ok"]
    assert episode["packet"]["inspection_steps"] == 1
    assert load(Path(trace.info["envoy"]["output"]) / "cleanup.json")["all_absent"]


def test_changed_prompt_fails_before_any_model_request(tmp_path):
    task = tasks()[0].with_data(system_prompt="changed prompt")
    with scripted_provider(CONTROLS / (task.key + ".json")) as (url, received, _):

        async def run():
            async with vf.Agent(local_agent_config(url, tmp_path / "episodes")) as agent:
                return await agent.run(task)

        trace = asyncio.run(run())
    assert not trace.ok and not received and not trace.rewards


@pytest.mark.parametrize(
    "endpoint", ["https://127.0.0.1", "http://localhost", "http://example.com"]
)
def test_interception_transport_refuses_non_framework_endpoint(endpoint):
    with pytest.raises(ValueError, match="loopback"):
        InterceptTransport(endpoint, "fixture")


def test_unverified_training_path_is_explicitly_blocked():
    harness = ReaderHarness(ReaderHarnessConfig())
    trace = SimpleNamespace(agent=SimpleNamespace(trainable=True))
    ctx = SimpleNamespace(client=vf.TrainClientConfig(api_key_var="ENVOY_VERIFIERS_UNUSED"))
    with pytest.raises(ValueError, match="evaluation-only"):
        asyncio.run(harness.launch(ctx, trace, None, "", "", {}, None))
    assert trace.agent.trainable is False


def test_cancellation_waits_for_worker_finally_cleanup(monkeypatch, tmp_path):
    import src.training.verifiers_adapter as adapter

    completed = threading.Event()
    started = threading.Event()

    def worker(*args, **kwargs):
        started.set()
        try:
            time.sleep(0.05)
        finally:
            completed.set()

    monkeypatch.setattr(adapter, "collect", worker)
    trace = SimpleNamespace(agent=SimpleNamespace(trainable=True), id="fixture")
    config = local_agent_config("http://127.0.0.1:1/v1", tmp_path)
    ctx = SimpleNamespace(client=config.client, model="envoy")

    async def run():
        work = asyncio.create_task(
            ReaderHarness(config.harness).launch(
                ctx, trace, None, "http://127.0.0.1:1/v1", "fixture", {}, tasks()[0].data
            )
        )
        while not started.is_set():
            await asyncio.sleep(0.005)
        work.cancel()
        with pytest.raises(asyncio.CancelledError):
            await work
        assert completed.is_set()

    asyncio.run(run())
