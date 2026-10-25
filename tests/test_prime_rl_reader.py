"""Actual native Env -> Docker -> TrainClient -> prime-rl transport, with authored actions.

All engine token probabilities below are synthetic contract fixtures. Original
episodes stay untrainable; only in-memory copies enter the native sample compiler.
No pretrained weights, optimizers, remote services, or model-authored host Python.
"""
from __future__ import annotations

import asyncio
import copy
import importlib.metadata
import json
import re
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

try:
    compatible = (importlib.metadata.version("prime-rl") == "0.10.0"
                  and importlib.metadata.version("verifiers") == "0.4.0")
except importlib.metadata.PackageNotFoundError:
    compatible = False
if not compatible:
    pytest.skip("use requirements-prime-rl-contract.txt", allow_module_level=True)

import msgspec
import verifiers.v1 as vf
from huggingface_hub import snapshot_download
from prime_rl.configs.algorithm import GRPOAlgoConfig
from prime_rl.configs.orchestrator import OrchestratorConfig, TrainSourceConfig
from prime_rl.orchestrator.algo.base import iter_trainable_traces
from prime_rl.orchestrator.batch import prepare_batch
from prime_rl.orchestrator.envs import TrainEnvs
from prime_rl.orchestrator.train_sink import TrainSink
from prime_rl.orchestrator.types import Progress
from prime_rl.transports.batch import FileSystemBatchReceiver, FileSystemBatchSender, MicroBatch
from renderers.configs import DefaultRendererConfig
from transformers import AutoTokenizer

from src.eval.reader_improvement import load, write
from src.training.verifiers_adapter import (
    PLUGIN,
    ReaderEnvConfig,
    ReaderHarness,
    ReaderHarnessConfig,
    ReaderTasksetConfig,
    complete_reader_group,
    local_agent_config,
)

ROOT = Path(__file__).resolve().parents[1]
REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
CONTROL = sorted((ROOT / "examples/verifiers-port/controls").glob("*.json"))[0]


@contextmanager
def token_engine(tokenizer, actions, mode):
    """Local authored-action fixture for vLLM's HTTP contract, not a vLLM engine."""
    received = []

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, body):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            assert self.path == "/v1/models"
            self.reply(200, {"data": [{"id": "envoy", "max_model_len": 32768,
                                      "root": "/scripted-fixture"}]})

        def do_POST(self):
            assert self.path == "/inference/v1/generate"
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            index = len(received)
            received.append(body)
            if mode == "provider_fail" and index == 1:
                return self.reply(503, {"error": {"message": "fixture provider failure"}})
            if index >= len(actions):
                return self.reply(400, {"error": {"message": "fixture request ceiling"}})
            ids = (tokenizer.encode(actions[index], add_special_tokens=False)
                   + [tokenizer.eos_token_id])
            body["_fixture_response_ids"] = ids
            probs = [{"token": f"token_id:{token}", "logprob": -1.0} for token in ids]
            if mode == "bad_metadata" and index == 1:
                probs.pop()
            self.reply(200, {"choices": [{"token_ids": ids,
                "logprobs": {"content": probs}, "finish_reason": "stop"}]})

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", received
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture(scope="module")
def contract_runs(tmp_path_factory):
    work = tmp_path_factory.mktemp("prime-contract")
    snapshot = snapshot_download("Qwen/Qwen3-8B", revision=REVISION, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    control = load(CONTROL)
    task = next(t for t in vf.load_taskset(ReaderTasksetConfig()) if t.key == control["case_id"])
    episodes = {}
    for mode in ("good", "bad", "fenced", "recover", "provider_fail", "bad_metadata"):
        actions = list(control["responses"])
        if mode == "bad":
            actions[-1] = re.sub(r"^SUBMIT: .*? CITATIONS:",
                                "SUBMIT: wrong_fixture_answer CITATIONS:", actions[-1])
        elif mode == "fenced":
            actions[0] = "```python\n" + actions[0] + "\n```"
        elif mode == "recover":
            actions.insert(0, "print(1 / 0)")
        with token_engine(tokenizer, actions, mode) as (url, received):
            config = local_agent_config(url, work / mode)
            config.harness.token_contract_only = True
            config.client = vf.TrainClientConfig(
                base_url=url, api_key_var="ENVOY_VERIFIERS_UNUSED",
                renderer_model_name=snapshot, renderer=DefaultRendererConfig(enable_thinking=False))
            config.sampling = vf.SamplingConfig(**config.sampling.model_dump(), top_k=-1, min_p=0.0)
            env = vf.load_environment(ReaderEnvConfig(taskset=ReaderTasksetConfig(), agent=config))

            async def run():
                async with env.serving():
                    return await env.run_episode(task, vf.ModelContext(
                        model="envoy", client=config.client, sampling=config.sampling))

            episode = asyncio.run(run())
        trace = episode.traces[0]
        write(work / (mode + "-episode.json"), episode.to_record(float_decimals=None))
        write(work / (mode + "-engine.json"), received)
        payloads = load(Path(trace.info["envoy"]["output"]) / "payloads.json")
        for payload, request in zip(payloads, received, strict=True):
            assert request["token_ids"] == tokenizer.apply_chat_template(
                payload["messages"], tokenize=True, return_dict=False,
                enable_thinking=False, add_generation_prompt=True)
            p = request["sampling_params"]
            assert (p["temperature"], p["top_p"], p["top_k"], p["min_p"], p["max_tokens"]
                    ) == (1.0, 1.0, -1, 0.0, 1024)
        if mode in {"good", "bad"}:
            assert payloads == control["payloads"]
        assert trace.agent.trainable is False
        assert not list(iter_trainable_traces([episode]))
        assert load(Path(trace.info["envoy"]["output"]) / "cleanup.json")["all_absent"]
        episodes[mode] = (episode, received, actions)
    return work, episodes


@pytest.mark.parametrize("mode", [
    "good", "bad", "fenced", "recover", "provider_fail", "bad_metadata",
])
def test_native_training_contract_through_real_docker(contract_runs, mode):
    _, runs = contract_runs
    episode, received, actions = runs[mode]
    trace = episode.traces[0]
    if mode in {"provider_fail", "bad_metadata"}:
        assert not episode.ok and trace.has_error and not trace.rewards
        assert len(received) == 2
        assert trace.state.episode["packet"]["inspection_steps"] == 1
        return
    assert episode.ok, episode.errors or trace.errors
    assert trace.rewards["short_fact"].score == (0.0 if mode == "bad" else 1.0)
    assert len(received) == (4 if mode == "recover" else 3)
    assert len(trace.branches) == len(received)
    for branch, request, raw in zip(trace.branches, received, actions, strict=True):
        assert branch.nodes[-1].message.content == raw
        prompt = request["token_ids"]
        n = len(prompt)
        assert branch.token_ids[:n] == prompt
        assert branch.token_ids[n:] == request["_fixture_response_ids"]
        assert branch.sampled_mask == [False] * n + [True] * (len(branch.token_ids) - n)
        assert branch.logprobs[n:] == [-1.0] * (len(branch.token_ids) - n)
    if mode == "recover":
        # The frozen passage reader rejects arbitrary Python before execution.
        assert any(n.message.role == "user" and "Action rejected:" in str(n.message.content)
                   for n in trace.nodes)
        assert trace.state.episode["packet"]["inspection_steps"] == 2


def train_sink(tmp_path, tokenizer=None):
    source = TrainSourceConfig(name=PLUGIN, env=ReaderEnvConfig(taskset=ReaderTasksetConfig()),
        algo=GRPOAlgoConfig(length_penalty=None, length_weighted_baseline=False,
                            loss_aggregation="token"), group_size=2,
        sampling={"temperature": 1.0, "top_p": 1.0, "max_completion_tokens": 1024})
    envs = TrainEnvs([source], {("train", PLUGIN): None}, tmp_path, clients=None)
    return TrainSink(OrchestratorConfig(constant_trainer_batch_size=False), tokenizer=tokenizer,
        train_envs=envs, progress=Progress(), batch_size=2,
        on_result=lambda group: complete_reader_group(group, 2))


def export_copy(episode, *, trainable=True):
    """Explicitly synthetic, in-memory compiler input; never persist as eligible data."""
    episode = copy.deepcopy(episode)
    episode.group = vf.GroupInfo(id="authored-contract-group")
    episode.env.name = PLUGIN
    episode.run = vf.TrainRunInfo(id="contract-only", work=vf.TrainWorkInfo(
        step=1, policy=vf.PolicySpan(start=0, end=0)))
    episode.traces[0].agent.trainable = trainable
    return episode


def test_native_grpo_sink_packer_and_trainer_transport(contract_runs, tmp_path):
    work, runs = contract_runs
    good, bad = [export_copy(runs[name][0]) for name in ("good", "bad")]
    sink = train_sink(tmp_path)

    async def compile_and_transfer():
        assert await sink.add(good) is None  # Incomplete groups never ship.
        batch = await sink.add(bad)
        assert batch is not None and len(batch.samples) == 6
        for sample in batch.samples:
            positive = sample.trace_id == good.traces[0].id
            expected = 0.5 if positive else -0.5
            assert sample.temperatures == [1.0] * len(sample.token_ids)
            assert sample.advantages == [expected if m else 0.0 for m in sample.mask]
            assert sample.logprobs == [-1.0 if m else 0.0 for m in sample.mask]
            assert sample.ce_weights is None and sample.ref_kl_weights is None
        grid = prepare_batch(batch.samples, seq_len=8192, num_train_workers=1,
                             bin_cost=lambda lengths: sum(lengths))
        assert sum(sum(b.loss_mask) for b in grid[0]) == sum(sum(s.mask) for s in batch.samples)
        sender = FileSystemBatchSender(tmp_path, data_world_size=1)
        receiver = FileSystemBatchReceiver(tmp_path, data_rank=0, rank=0, world_size=1)
        await asyncio.wait_for(
            asyncio.gather(sender.send(grid), asyncio.to_thread(receiver.wait)), 10)
        decoded = receiver.receive()
        assert decoded == grid[0]
        encoded = msgspec.msgpack.encode(decoded)
        assert msgspec.msgpack.decode(encoded, type=list[MicroBatch]) == decoded
        write(work / "native-export-summary.json", {
            "samples": len(batch.samples), "microbatches": len(decoded),
            "sampled_tokens": sum(sum(s.mask) for s in batch.samples),
            "synthetic_probabilities": True, "optimizer_updates": 0})
    asyncio.run(compile_and_transfer())


def test_original_contract_episodes_never_feed_training(contract_runs, tmp_path):
    _, runs = contract_runs
    episodes = [export_copy(runs[name][0], trainable=False) for name in ("good", "bad")]
    sink = train_sink(tmp_path)

    async def run():
        for episode in episodes:
            assert await sink.add(episode) is None
        assert not sink.pending_batch
    asyncio.run(run())


@pytest.mark.parametrize("other", ["provider_fail", "bad_metadata", "good"])
def test_failed_or_uniform_group_emits_no_training_payload(contract_runs, tmp_path, other):
    _, runs = contract_runs
    good, second = [export_copy(runs[name][0]) for name in ("good", other)]
    # Duplicate successful fixture instances need distinct trace identities.
    second.traces[0].id += "-second"
    sink = train_sink(tmp_path)

    async def run():
        assert await sink.add(good) is None
        result = await sink.add(second)
        assert result is None or not result.samples
        assert not sink.pending_batch
    asyncio.run(run())


@pytest.mark.parametrize("mutation", [
    "incomplete", "task", "group", "reward", "nonfinite", "untrainable", "policy", "run",
    "trace_task", "work", "weight",
])
def test_group_admission_requires_complete_matching_healthy_rollouts(contract_runs, mutation):
    _, runs = contract_runs
    group = [export_copy(runs[name][0]) for name in ("good", "bad")]
    assert complete_reader_group(group, 2)
    if mutation == "incomplete":
        group.pop()
    elif mutation == "task":
        group[1].task.hash = "different-task"
    elif mutation == "group":
        group[1].group.id = "different-group"
    elif mutation == "reward":
        group[1].traces[0].rewards.clear()
    elif mutation == "nonfinite":
        group[1].traces[0].rewards["short_fact"].score = float("nan")
    elif mutation == "policy":
        group[1].run.work.policy = vf.PolicySpan(start=1, end=1)
    elif mutation == "run":
        group[1].run = None
    elif mutation == "trace_task":
        group[1].traces[0].task.hash = "wrong-inner-task"
    elif mutation == "work":
        group[1].run.work = vf.EvalWorkInfo(step=1)
    elif mutation == "weight":
        group[1].traces[0].rewards["short_fact"].weight = float("nan")
    else:
        group[1].traces[0].agent.trainable = False
    assert not complete_reader_group(group, 2)


@pytest.mark.parametrize("renderer,model", [
    (None, REVISION), (DefaultRendererConfig(enable_thinking=True), REVISION),
    (DefaultRendererConfig(enable_thinking=False), "Qwen/Qwen3-8B"),
])
def test_contract_refuses_unpinned_or_changed_renderer(renderer, model):
    harness = ReaderHarness(ReaderHarnessConfig(token_contract_only=True))
    trace = SimpleNamespace(agent=SimpleNamespace(trainable=True))
    ctx = SimpleNamespace(client=vf.TrainClientConfig(renderer=renderer,
        renderer_model_name=model, api_key_var="ENVOY_VERIFIERS_UNUSED"))
    with pytest.raises(ValueError, match="token contract"):
        asyncio.run(harness.launch(ctx, trace, None, "", "", {}, None))
    assert trace.agent.trainable is False


def test_sequential_native_configs_resume_exactly_one_update_with_optimizer(tmp_path):
    from prime_rl.configs.inference import InferenceConfig
    from prime_rl.configs.trainer import TrainerConfig

    from src.training.prime_native import configure

    for step in (1, 2, 3):
        cycle = tmp_path / f"cycle-{step}"
        cycle.mkdir()
        configure({"output": str(tmp_path), "assets": "/workspace/assets"}, step)
        config = TrainerConfig.model_validate_json((cycle / "trainer.json").read_text())
        assert config.max_steps - config.resume.dir_step == 1
        assert not config.ckpt.skip_optimizer and not config.ckpt.skip_scheduler
        if step > 1:
            assert config.resume.dir == (
                tmp_path / f"cycle-{step - 1}/train/checkpoints/step_{step - 1}")
        assert config.model.compile is None
        assert config.optim.lr == 1e-6 and config.model.lora.dropout == 0
        assert config.model.offload == "none"
        inference = InferenceConfig.model_validate_json((cycle / "inference.json").read_text())
        assert inference.server.host == "127.0.0.1"
        assert inference.vllm.model == config.model.name
        assert inference.router is None and inference.vllm.generation_config == "vllm"
        # Compare effective native defaults to the real smoke, not just our draft.
        smoke = load(ROOT / "examples/verifiers-port/prime-rl-smoke/trainer.json")
        actual = config.model_dump(mode="json")
        for name in ("model", "optim", "loss", "scheduler"):
            expected = copy.deepcopy(smoke[name])
            if name == "model":
                expected["name"] = actual["model"]["name"]
            assert actual[name] == expected


def test_repeatable_collector_uses_current_policy_receipt_and_native_step_two_transport(
        tmp_path, monkeypatch):
    """Authored engine replies exercise real Docker/native batching; no GPU proof."""
    import time

    from src.eval.reader_improvement import file_hash
    from src.training import prime_budget
    from src.training.prime_native import collect_group, transport_stage

    control = load(CONTROL)
    tokenizer_path = Path(snapshot_download("Qwen/Qwen3-8B", revision=REVISION,
                                           local_files_only=True))
    tok = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    assets = tmp_path / "assets"
    (assets / "base").mkdir(parents=True)
    (assets / "base" / REVISION).symlink_to(tokenizer_path, target_is_directory=True)
    (assets / "v5").mkdir()
    (assets / "v5/adapter_model.safetensors").write_bytes(b"fixture-not-model-weights")
    plan = {"output": str(tmp_path), "assets": str(assets),
            "train_case_ids": [control["case_id"]],
            "limits": {"requests": 50, "reserved_tokens": 409600}}
    cycle = tmp_path / "cycle-2"
    cycle.mkdir()
    previous = tmp_path / "cycle-1/adapter"
    previous.mkdir(parents=True)
    (previous / "adapter_model.safetensors").write_bytes(b"different-fixture-policy")
    write(cycle / "loaded.json", {"version": 1, "sha256": file_hash(
        previous / "adapter_model.safetensors"), "fixture": True})
    write(tmp_path / "started.json", {"deadline": time.time() + 120})
    good = control["responses"]
    answer = good[-1].removeprefix("SUBMIT: ").split(" CITATIONS:")[0]
    bad = [*good[:-1], good[-1].replace(answer, "not the reference answer", 1)]
    # This fixture's original answer lives in its final SUBMIT payload.
    assert bad[-1] != good[-1]
    with token_engine(tok, good + bad + good + bad, "good") as (url, received):
        monkeypatch.setattr(prime_budget, "ENGINE", url.removesuffix("/v1"))
        asyncio.run(collect_group(plan, 2))
    candidate = load(cycle / "candidate.json")
    assert candidate["policy"]["version"] == 1
    assert candidate["policy"]["sha256"] == file_hash(previous / "adapter_model.safetensors")
    assert len(set(candidate["rewards"])) == 2 and candidate["training_ready"] is False
    assert len(received) == 12
    assert all(r["policy"]["version"] == 1 for r in load(tmp_path / "dispatches.json")["requests"])
    # Raw VF and investigator records stay untrainable; the batch has a scorer barrier.
    records = [load(p) for p in cycle.glob("episode-*.json")]
    assert len(records) == 4
    assert all(r["traces"][0]["agent"]["trainable"] is False for r in records)
    assert not (cycle / "prepared.json").exists()

    async def transfer():
        receiver = FileSystemBatchReceiver(cycle / "train", data_rank=0, rank=0,
                                          world_size=1, current_step=2)
        sender = asyncio.create_task(transport_stage(cycle, 2, "batch"))
        await asyncio.to_thread(receiver.wait)
        actual = receiver.receive()
        await sender
        expected = msgspec.msgpack.decode((cycle / "microbatches.bin").read_bytes(),
                                         type=list[MicroBatch])
        assert actual == expected
    asyncio.run(asyncio.wait_for(transfer(), timeout=15))


def test_paired_evaluation_native_renderer_real_docker_is_unscored(tmp_path, monkeypatch):
    """An explicit train-role scripted fixture, never the locked evaluation set."""
    from prime_rl.orchestrator.algo.base import iter_trainable_traces

    from src.training import reader_evaluation as ev
    from src.training.reader_data import load_bundle
    from src.training.verifiers_adapter import task_case

    snapshot = snapshot_download("Qwen/Qwen3-8B", revision=REVISION, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    ev.check_tokenizer(Path(snapshot))
    control = load(CONTROL)
    _, roles, _, _ = load_bundle(ev.BUNDLE)
    case = next(c for c in roles["train"] if c["case_id"] == control["case_id"])
    write(tmp_path / "session.json", {"synthetic_fixture_only": True})
    session = {"bundle_sha256": ev.file_hash(ev.BUNDLE / "manifest.json"),
               "arms": {ev.ARMS[0]: "not_real_weights"}}
    monkeypatch.setattr(ev, "validate_session", lambda output: (session, [case]))
    with token_engine(tokenizer, control["responses"], "good") as (engine, received):
        episode = asyncio.run(ev.collect_episode(
            tmp_path, case, snapshot, engine, ev.ARMS[0], session, 42))
    assert episode.ok and len(episode.traces) == 1
    trace = episode.traces[0]
    assert not trace.rewards and trace.agent.trainable is False
    assert not list(iter_trainable_traces([episode]))
    assert len(received) == len(control["responses"])
    assert all(r["sampling_params"]["seed"] == 42 for r in received)
    assert load(Path(trace.info["envoy"]["output"]) / "payloads.json") == control["payloads"]
    assert load(Path(trace.info["envoy"]["output"]) / "cleanup.json")["all_absent"]
    assert trace.state.episode["packet"]["execution"] == "docker_only"
    with pytest.raises(ValueError, match="cannot receive a training reward"):
        task_case(trace.task.data)


def test_paired_runner_two_arms_shared_budget_and_real_review_export(tmp_path, monkeypatch):
    from src.training import reader_evaluation as ev
    from src.training.reader_data import load_bundle

    snapshot = snapshot_download("Qwen/Qwen3-8B", revision=REVISION, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    control = load(CONTROL)
    _, roles, _, _ = load_bundle(ev.BUNDLE)
    case = next(c for c in roles["train"] if c["case_id"] == control["case_id"])
    write(tmp_path / "session.json", {"synthetic_fixture_only": True})
    session = {"bundle_sha256": ev.file_hash(ev.BUNDLE / "manifest.json"),
               "arms": {a: "not_real_weights_" + a for a in ev.ARMS}}
    # No locked reference is opened, and no candidate is falsely certified.
    monkeypatch.setattr(ev, "validate_session", lambda output: (session, [case]))
    monkeypatch.setattr(ev, "reference_review", lambda output, cases: "scripted_source_review")
    for arm in ev.ARMS:
        (tmp_path / arm).mkdir()
        backend = tmp_path / arm / "backend.json"
        write(backend, {"session_sha256": ev.file_hash(tmp_path / "session.json"),
            "arm": arm, "adapter_sha256": session["arms"][arm], "load_rpc_completed": True,
            "adapter_root": "/scripted-fixture", "gpu": "no_gpu_scripted_fixture",
            "versions": {"fixture": True}})
        with token_engine(tokenizer, control["responses"], "good") as (engine, received):
            asyncio.run(ev.run_arm(tmp_path, arm, engine.removesuffix("/v1"), snapshot, backend))
        assert len(received) == len(control["responses"])
        episode = load(tmp_path / arm / "episode-00.json")
        assert not episode["traces"][0].get("rewards")
    ledger = load(tmp_path / "dispatches.json")
    assert len(ledger["requests"]) == 2 * len(control["responses"])
    assert not ledger["closed"]
    assert {r["policy"]["arm"] for r in ledger["requests"]} == set(ev.ARMS)
    ev.export_review(tmp_path)
    review = load(tmp_path / "output-review.json")
    assert len(review["rows"]) == 2
    assert all(r["complete_correct_supported"] is None for r in review["rows"])
    assert ev.ARMS[0] not in json.dumps(review)
    assert ev.ARMS[1] not in json.dumps(review)


def test_native_adapter_loader_loopback_receipt_without_credentials(tmp_path, monkeypatch):
    """Real native admin HTTP, with explicitly mocked GPU identity; no model weights."""
    import torch

    from src.training import reader_evaluation as ev

    write(tmp_path / "session.json", {"synthetic_fixture_only": True})
    arm = ev.ARMS[0]
    (tmp_path / arm).mkdir()
    session = {"arms": {arm: "synthetic_adapter_digest"}}
    monkeypatch.setattr(ev, "validate_session", lambda output: (session, []))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i: "synthetic_gpu_identity")
    original_version = ev.metadata.version
    versions = {"prime-rl": "0.10.0", "verifiers": "0.4.0", "renderers": "0.1.13.dev5",
                "transformers": "5.6.2", "torch": "2.13.0+cu130", "vllm": "0.31.0"}
    monkeypatch.setattr(ev.metadata, "version", lambda n: versions.get(n) or original_version(n))
    monkeypatch.setenv("PRIME_API_KEY", "fixture_must_not_be_sent")
    calls = []

    class Admin(BaseHTTPRequestHandler):
        def reply(self, body):
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            calls.append((self.path, self.headers.get("Authorization")))
            self.reply({"data": [{"id": "envoy-base"},
                                  {"id": "envoy", "root": str(tmp_path / arm)}]})

        def do_POST(self):
            calls.append((self.path, self.headers.get("Authorization")))
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert self.path == "/load_lora_adapter"
            assert body == {"lora_name": "envoy", "lora_path": str(tmp_path / arm)}
            self.reply({"fixture": True})

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Admin)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        asyncio.run(ev.load_backend(tmp_path, arm, f"http://127.0.0.1:{server.server_port}"))
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
    assert any(p == "/load_lora_adapter" for p, _ in calls)
    assert all(auth is None for _, auth in calls)
    receipt = load(tmp_path / arm / "backend.json")
    assert receipt["load_rpc_completed"] and receipt["engine_tensor_attestation"] is False
    assert receipt["gpu"] == "synthetic_gpu_identity"


def test_external_training_handoff_real_docker_and_native_batch(tmp_path, monkeypatch):
    """One authored mixed group, two filesystem roots; no GPU or updated weights."""
    import shutil
    import time

    from src.eval.reader_improvement import file_hash
    from src.training import prime_controller as bridge
    from src.training.prime_run import prepare

    control = load(CONTROL)
    tokenizer_path = Path(snapshot_download("Qwen/Qwen3-8B", revision=REVISION,
                                            local_files_only=True))
    tok = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    assets = tmp_path / "assets"
    (assets / "base").mkdir(parents=True)
    (assets / "base" / REVISION).symlink_to(tokenizer_path, target_is_directory=True)
    (assets / "v5").mkdir()
    (assets / "v5/adapter_model.safetensors").write_bytes(b"fixture-not-model-weights")
    local, gpu = tmp_path / "local", tmp_path / "gpu"
    local_plan = prepare(local, assets, updates=2)
    gpu_plan = prepare(gpu, tmp_path / "gpu-assets", updates=2,
                       controller_plan_sha256=file_hash(local / "plan.json"))
    # Fixture narrowing is explicit; production validation still requires all 81 IDs.
    local_plan["train_case_ids"] = [control["case_id"]]
    gpu_plan["train_case_ids"] = [control["case_id"]]
    monkeypatch.setattr(bridge, "validate_plan", lambda output: local_plan)
    cycle = gpu / "cycle-1"
    cycle.mkdir()
    write(gpu / "started.json", {"deadline": time.time() + 120})
    write(cycle / "loaded.json", {"version": 0, "sha256": file_hash(
        assets / "v5/adapter_model.safetensors"), "load_rpc_completed": True,
        "engine_reported_path_matches": True})
    job = bridge.request(gpu_plan, 1)
    job_path = tmp_path / "job.json"
    write(job_path, job)
    good = control["responses"]
    answer = good[-1].removeprefix("SUBMIT: ").split(" CITATIONS:")[0]
    bad = [*good[:-1], good[-1].replace(answer, "not the reference answer", 1)]
    with token_engine(tok, good + bad + good + bad, "good") as (url, received):
        response = bridge.collect(local, job_path, url.removesuffix("/v1"))
    assert len(received) == 12
    assert response["healthy_original_episodes"] == 4
    assert response["all_recorded_containers_absent"] is True
    for name in ("microbatches.bin", "candidate.json", "collection-complete.json"):
        shutil.copyfile(local / "cycle-1" / name, cycle / name)
    bridge.accept(gpu_plan, 1, job, response)
    records = [load(p) for p in (local / "cycle-1").glob("episode-*.json")]
    assert all(r["traces"][0]["agent"]["trainable"] is False for r in records)
    with pytest.raises(FileExistsError):
        bridge.collect(local, job_path, url.removesuffix("/v1"))


def test_fresh_optimizer_native_dcp_does_not_add_a_training_step(tmp_path):
    """Real PyTorch DCP/get-state APIs used by native checkpoints; tiny CPU model."""
    import torch
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

    from src.training.prime_native import initialize_optimizer_state, optimizer_fingerprint

    torch.manual_seed(42)
    model = torch.nn.Linear(2, 1)
    reference = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
    ref_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-3, weight_decay=0)
    initialize_optimizer_state(model, optimizer)
    assert optimizer_fingerprint(optimizer, 0)[1] == [0]
    before = optimizer_fingerprint(optimizer, 0)
    weights, state = get_state_dict(model, [optimizer])
    dcp.save({"model": weights, "optimizer": state}, checkpoint_id=tmp_path / "initial")
    assert optimizer_fingerprint(optimizer, 0) == before
    restored = torch.nn.Linear(2, 1)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3, weight_decay=0)
    weights, state = get_state_dict(restored, [restored_optimizer])
    checkpoint = {"model": weights, "optimizer": state}
    dcp.load(checkpoint, checkpoint_id=tmp_path / "initial")
    set_state_dict(restored, [restored_optimizer], model_state_dict=checkpoint["model"],
                   optim_state_dict=checkpoint["optimizer"])
    assert optimizer_fingerprint(restored_optimizer, 0) == before
    for actual, expected in zip(restored.parameters(), reference.parameters(), strict=True):
        assert torch.equal(actual, expected)
    for learner, opt in ((restored, restored_optimizer), (reference, ref_optimizer)):
        learner(torch.tensor([[1., 2.]])).sum().backward()
        opt.step()
    assert optimizer_fingerprint(restored_optimizer, 1)[1] == [1]
    for actual, expected in zip(restored.parameters(), reference.parameters(), strict=True):
        assert torch.equal(actual, expected)
    # Initializer must never be applied to an existing learned optimizer state.
    with pytest.raises(ValueError, match="fresh"):
        initialize_optimizer_state(restored, restored_optimizer)
