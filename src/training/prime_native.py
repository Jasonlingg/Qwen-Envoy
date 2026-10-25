"""Native stages for prime_run. No custom loss, optimizer or learner loop."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import importlib.metadata as metadata
import json
import shutil
import socket
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

from src.eval.reader_improvement import file_hash
from src.training.prime_run import BASE_REVISION, V5_SHA, validate_plan, verify_base_assets, write


def configure(plan, step):
    """One unchanged native update; external resume carries optimizer and scheduler."""
    from prime_rl.configs.inference import InferenceConfig
    from prime_rl.configs.trainer import TrainerConfig

    root, assets = Path(plan["output"]), Path(plan["assets"])
    cycle, base = root / f"cycle-{step}", assets / "base" / BASE_REVISION
    resume = (
        root / "cycle-1/init/step_0"
        if step == 1
        else root / f"cycle-{step - 1}/train/checkpoints/step_{step - 1}"
    )
    trainer = TrainerConfig.model_validate(
        {
            "model": {
                "name": str(base),
                "seq_len": 8192,
            "attn": "flash_attention_2",
            "compile": None,
            "offload": "none",
                "lora": {
                    "rank": 4,
                    "alpha": 8,
                    "dropout": 0,
                    "target_modules": [
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                        "up_proj",
                        "down_proj",
                        "gate_proj",
                    ],
                },
            },
            "tokenizer": {"name": str(base)},
            "data": {"num_workers": 1},
            "loss": {"type": "ppo", "ratio_low": 0.8, "ratio_high": 1.2},
            "optim": {"type": "adamw", "lr": 1e-6, "weight_decay": 0},
            "scheduler": {"type": "constant"},
            "max_steps": step,
            "ckpt": {"interval": 1, "skip_optimizer": False, "skip_scheduler": False},
            "resume": {"dir": str(resume)},
            "output_dir": str(cycle / "train"),
            "weight_broadcast": {"type": "filesystem", "timeout": 180},
            "rollout_transport": {"type": "filesystem"},
            "dist_timeout_seconds": 180,
        }
    )
    inference = InferenceConfig.model_validate(
        {
            "server": {"host": "127.0.0.1", "port": 8000},
            "router": None,
            "backend_port": 8100,
            "vllm": {
                "model": str(base),
                "max_model_len": 8192,
                "enable_lora": True,
                "max_lora_rank": 8,
                "served_model_name": ["envoy-base"],
                "gpu_memory_utilization": 0.7,
                "enforce_eager": True,
                "max_num_seqs": 4,
                "enable_prefix_caching": False,
                "dtype": "bfloat16",
                "generation_config": "vllm",
                "seed": 42,
            },
            "deployment": {"type": "single_node", "gpus_per_node": 1},
            "weight_broadcast": {"type": "filesystem"},
            "output_dir": str(cycle / "inference"),
        }
    )
    write(cycle / "trainer.json", trainer.model_dump(mode="json"))
    write(cycle / "inference.json", inference.model_dump(mode="json"))


def preflight(plan):
    import prime_rl.orchestrator.clients as native_clients
    import torch
    from flash_attn import flash_attn_func

    versions = {
        n: metadata.version(n)
        for n in ("prime-rl", "verifiers", "renderers", "transformers", "torch", "vllm")
    }
    expected = {
        "prime-rl": "0.10.0",
        "verifiers": "0.4.0",
        "renderers": "0.1.13.dev5",
        "transformers": "5.6.2",
        "torch": "2.13.0+cu130",
        "vllm": "0.31.0",
    }
    if versions != expected or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("use the exact smoke GPU stack and one visible GPU")
    if torch.cuda.get_device_properties(0).total_memory < 75 * 1024**3:
        raise ValueError("sequential path requires the tested 80GB GPU class")
    root, assets = Path(plan["output"]), Path(plan["assets"])
    if plan.get("controller_plan_sha256") is None:
        subprocess.run(["docker", "image", "inspect", plan["frozen_runtime"]["sandbox_image"]],
                       check=True, capture_output=True, timeout=15)
    if file_hash(assets / "v5/adapter_model.safetensors") != V5_SHA:
        raise ValueError("initial adapter differs from pinned V5")
    catalog = json.loads((Path(__file__).parents[2] /
        "examples/verifiers-port/prime-base-files.json").read_text())
    config_bytes = (assets / "v5/adapter_config.json").read_bytes()
    config_git_id = hashlib.sha1(
        f"blob {len(config_bytes)}\0".encode() + config_bytes).hexdigest()
    if (catalog["v5_config"]["revision"] != plan["v5_revision"]
            or config_git_id != catalog["v5_config"]["git_blob_id"]):
        raise ValueError("V5 adapter config differs from pinned Hub revision")
    proof = json.loads(
        (
            Path(__file__).parents[2]
            / "examples/verifiers-port/prime-rl-smoke/native-source-proof.json"
        ).read_text()
    )
    package = Path(native_clients.__file__).parents[1]
    for name, sha in proof["matching_source_files"].items():
        if file_hash(package / name.removeprefix("src/prime_rl/")) != sha:
            raise ValueError("native Prime-RL source drift: " + name)
    renderer = json.loads(metadata.distribution("renderers").read_text("direct_url.json") or "{}")
    if renderer.get("vcs_info", {}).get("commit_id") != plan["renderer_revision"]:
        raise ValueError("renderer must be installed from the pinned commit")
    # Ports must be ours; never attach to a leftover engine serving older weights.
    for port in (8000, 8100, 29571):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))
    if shutil.disk_usage(root).free < (40 + 32 * plan["limits"]["updates"]) * 1024**3:
        raise ValueError("insufficient disk for retained optimizer-resumable checkpoints")
    base = assets / "base" / BASE_REVISION
    hashes = verify_base_assets(base)
    q = torch.randn(1, 16, 8, 64, device="cuda", dtype=torch.bfloat16)
    if not torch.isfinite(flash_attn_func(q, q, q)).all():
        raise ValueError("FA2 kernel failed")
    write(
        root / "preflight.json",
        {
            "versions": versions,
            "gpu": torch.cuda.get_device_name(0),
            "base_git_or_lfs_identities": hashes,
            "base_matches_pinned_hub_file_catalog": True,
            "v5_sha256": V5_SHA,
            "native_source_sha256": proof["matching_source_files"],
            "external_docker_controller_plan_sha256": plan.get("controller_plan_sha256"),
        },
    )


def policy_path(plan, step):
    root = Path(plan["output"])
    return Path(plan["assets"]) / "v5" if step == 1 else root / f"cycle-{step - 1}/adapter"


def optimizer_fingerprint(optimizer, version):
    """Stable across state-dict insertion order; require every AdamW parameter state."""
    import torch

    digest, steps = hashlib.sha256(), set()
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    for index, parameter in enumerate(parameters):
        state = optimizer.state.get(parameter, {})
        if version and set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("incomplete AdamW optimizer state")
        for name, value in sorted(state.items()):
            local = value.to_local() if hasattr(value, "to_local") else value
            if not torch.isfinite(local).all():
                raise ValueError("nonfinite optimizer state")
            digest.update(f"{index}:{name}:{local.dtype}:{tuple(local.shape)}".encode())
            digest.update(local.detach().cpu().contiguous().numpy().tobytes())
            if name == "step":
                if version and local.item() != version:
                    raise ValueError("optimizer was reset or contains the wrong update count")
                steps.add(int(local.item()))
    if version and steps != {version}:
        raise ValueError("optimizer was reset or contains the wrong update count")
    return digest.hexdigest(), sorted(steps)


def initialize_optimizer_state(model, optimizer):
    """Materialize fresh AdamW state without counting DCP's dummy zero-grad step.

    PyTorch get_state_dict initializes an empty optimizer by calling step().
    Only normalize this verified, fresh, zero-moment bootstrap; never a resume.
    The actual trainer, optimizer and checkpoint implementation stay native.
    """
    import torch
    from torch.distributed.checkpoint.state_dict import get_state_dict

    if (not isinstance(optimizer, torch.optim.AdamW) or optimizer.state
            or any(g["weight_decay"] != 0 for g in optimizer.param_groups)):
        raise ValueError("only a fresh zero-decay AdamW can be initialized")
    parameters = [p for g in optimizer.param_groups for p in g["params"]]

    def local(tensor):
        return tensor.to_local() if hasattr(tensor, "to_local") else tensor

    before = [local(p).detach().clone() for p in parameters]
    get_state_dict(model, [optimizer])
    for p, original in zip(parameters, before, strict=True):
        state = optimizer.state[p]
        if (not torch.equal(local(p), original)
                or set(state) != {"step", "exp_avg", "exp_avg_sq"}
                or local(state["step"]).item() != 1
                or any(torch.count_nonzero(local(state[n])).item()
                       for n in ("exp_avg", "exp_avg_sq"))):
            raise ValueError("checkpoint materialization changed weights or optimizer moments")
        local(state["step"]).zero_()


async def load_policy(plan, step):
    from prime_rl.configs.shared import ClientConfig
    from prime_rl.orchestrator.clients import AdminPlane, load_lora_adapter

    path = policy_path(plan, step).resolve()
    before = file_hash(path / "adapter_model.safetensors")
    plane = AdminPlane(
        ClientConfig(base_url="http://127.0.0.1:8000/v1", wait_for_ready_timeout=300)
    )
    try:
        await plane.wait_for_ready("envoy-base")
        await load_lora_adapter(plane, "envoy", path)
        for client in plane.clients:
            r = await client.get("/v1/models")
            r.raise_for_status()
            if not any(m["id"] == "envoy" and m["root"] == str(path) for m in r.json()["data"]):
                raise ValueError("engine did not report the requested adapter path")
        if before != file_hash(path / "adapter_model.safetensors"):
            raise ValueError("adapter changed during load")
        write(
            Path(plan["output"]) / f"cycle-{step}/loaded.json",
            {
                "version": step - 1,
                "path": str(path),
                "sha256": before,
                "load_rpc_completed": True,
                "engine_reported_path_matches": True,
                "engine_tensor_attestation": False,
            },
        )
    finally:
        for client in plane.clients:
            await client.aclose()


async def collect_group(plan, step, *, engine=None):
    import msgspec
    import verifiers.v1 as vf
    from prime_rl.configs.algorithm import GRPOAlgoConfig
    from prime_rl.configs.orchestrator import OrchestratorConfig, TrainSourceConfig
    from prime_rl.orchestrator.batch import prepare_batch
    from prime_rl.orchestrator.envs import TrainEnvs
    from prime_rl.orchestrator.train_sink import TrainSink
    from prime_rl.orchestrator.types import Progress
    from renderers.configs import DefaultRendererConfig
    from transformers import AutoTokenizer

    from src.training.prime_budget import proxy
    from src.training.verifiers_adapter import (
        PLUGIN,
        ReaderEnvConfig,
        ReaderTasksetConfig,
        complete_reader_group,
        local_agent_config,
    )

    root = Path(plan["output"])
    cycle, tokenizer_path = root / f"cycle-{step}", Path(plan["assets"]) / "base" / BASE_REVISION
    loaded = json.loads((cycle / "loaded.json").read_text())
    if loaded["sha256"] != file_hash(policy_path(plan, step) / "adapter_model.safetensors"):
        raise ValueError("loaded policy changed")
    tok = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    taskset = ReaderTasksetConfig(reward_revision=plan.get("reward_revision"))
    tasks = {t.key: t for t in vf.load_taskset(taskset)}
    source = TrainSourceConfig(
        name=PLUGIN,
        env=ReaderEnvConfig(taskset=taskset),
        algo=GRPOAlgoConfig(
            length_penalty=None, length_weighted_baseline=False, loss_aggregation="token"
        ),
        group_size=4,
        sampling={"temperature": 1.0, "top_p": 1.0, "max_completion_tokens": 1024},
    )
    cursor_path = root / "question-cursor.json"
    cursor = json.loads(cursor_path.read_text()) if cursor_path.exists() else 0
    with proxy(root, plan, loaded, engine=engine) as url:
        for position in range(cursor, len(plan["train_case_ids"])):
            case_id, group = plan["train_case_ids"][position], []
            write(cursor_path, position + 1)
            for index in range(4):
                config = local_agent_config(
                    url,
                    cycle / "rollouts",
                    model_identity={
                        "backend": "native_prime_vllm",
                        "policy_version": step - 1,
                        "adapter_sha256": loaded["sha256"],
                        "weights_verified": False,
                    },
                )
                # Originals always remain untrainable. The driver makes a new
                # candidate only after a full healthy group; scorer gates follow.
                config.harness.token_contract_only = True
                config.client = vf.TrainClientConfig(
                    base_url=url,
                    api_key_var="ENVOY_VERIFIERS_UNUSED",
                    renderer_model_name=str(tokenizer_path),
                    renderer=DefaultRendererConfig(enable_thinking=False),
                )
                config.sampling = vf.SamplingConfig(
                    **config.sampling.model_dump(),
                    top_k=-1,
                    min_p=0.0,
                    seed=42 + position * 4 + index,
                )
                env = vf.load_environment(
                    ReaderEnvConfig(taskset=taskset, agent=config)
                )
                async with env.serving():
                    episode = await env.run_episode(
                        tasks[case_id],
                        vf.ModelContext(
                            model="envoy", client=config.client, sampling=config.sampling
                        ),
                    )
                write(
                    cycle / f"episode-{position:03}-{index}.json",
                    episode.to_record(float_decimals=None),
                )
                if not episode.ok or len(episode.traces) != 1 or not episode.traces[0].ok:
                    raise ValueError(
                        "unhealthy group discarded; stop without a numeric failure reward"
                    )
                group.append(episode)
            rewards = [e.traces[0].rewards["short_fact"].score for e in group]
            write(
                cycle / f"group-{position:03}.json",
                {"case_id": case_id, "rewards": rewards, "mixed": len(set(rewards)) > 1},
            )
            if len(set(rewards)) < 2:
                continue
            candidates = copy.deepcopy(group)
            for e in candidates:
                e.group = vf.GroupInfo(id=f"{step}-{case_id}")
                e.env.name = PLUGIN
                e.run = vf.TrainRunInfo(
                    id=str(root),
                    work=vf.TrainWorkInfo(
                        step=step, policy=vf.PolicySpan(start=step - 1, end=step - 1)
                    ),
                )
                e.traces[0].agent.trainable = True
            envs = TrainEnvs([source], {("train", PLUGIN): None}, cycle, clients=None)
            sink = TrainSink(
                OrchestratorConfig(constant_trainer_batch_size=False),
                tokenizer=tok,
                train_envs=envs,
                progress=Progress(step=step),
                batch_size=4,
                on_result=lambda g: complete_reader_group(g, 4),
            )
            batch = None
            for e in candidates:
                batch = await sink.add(e)
            if batch is None:
                raise ValueError("native sink rejected complete candidate")
            grid = prepare_batch(
                batch.samples,
                seq_len=8192,
                num_train_workers=1,
                bin_cost=lambda lengths: sum(lengths),
            )
            (cycle / "microbatches.bin").write_bytes(msgspec.msgpack.encode(grid[0]))
            write(
                cycle / "candidate.json",
                {
                    "case_id": case_id,
                    "policy": loaded,
                    "batch_sha256": file_hash(cycle / "microbatches.bin"),
                    "rewards": rewards,
                    "training_ready": False,
                    "pending": "native scorer and checkpoint gates",
                },
            )
            return
    raise ValueError("training question pool exhausted without a complete mixed group")


def checkpoint_stage(plan, step, config, verify=False):
    """Import V5 once; thereafter use native DCP including optimizer state."""
    import prime_rl._compat  # noqa: F401
    import torch
    from prime_rl.trainer.ckpt import Progress, setup_ckpt_manager
    from prime_rl.trainer.lora import get_lora_state
    from prime_rl.trainer.model import setup_model
    from prime_rl.trainer.optim import setup_optimizer
    from prime_rl.trainer.parallel_dims import get_parallel_dims, resolve_ep
    from prime_rl.trainer.scheduler import setup_scheduler
    from prime_rl.trainer.utils import setup_torch_distributed
    from prime_rl.utils.logger import setup_logger
    from safetensors.torch import load_file, save_file

    root, cycle = Path(plan["output"]), Path(plan["output"]) / f"cycle-{step}"
    version = step if verify else step - 1
    setup_logger("INFO")
    setup_torch_distributed(timeout=timedelta(seconds=180))
    try:
        torch.manual_seed(42)
        resolve_ep(config.model)
        dims = get_parallel_dims(config.model)
        model = setup_model(config.model, dims, loading_from_checkpoint_later=version > 0)
        manager = setup_ckpt_manager(config.output_dir, config.ckpt, resume=config.resume)
        optimizer, _ = setup_optimizer(
            config.optim,
            list(model.named_parameters()),
            dims,
            offload=config.model.offload,
            model=model,
        )
        scheduler = setup_scheduler(optimizer, config.scheduler, config.max_steps, config.optim.lr)
        progress = Progress(step=0)
        if version > 0:
            checkpoint = cycle / f"train/checkpoints/step_{step}" if verify else config.resume.dir
            manager.load(
                version, model, [optimizer], scheduler, progress, path=checkpoint / "trainer"
            )
            if progress.step != version:
                raise ValueError("native checkpoint progress mismatch")
        else:
            old = load_file(Path(plan["assets"]) / "v5/adapter_model.safetensors")
            matched = set()
            with torch.no_grad():
                for prefix, module in get_lora_state()._modules:
                    for name, param in module.named_parameters_for_adapter(0):
                        key = "base_model.model." + prefix + "." + name + ".weight"
                        local = param.to_local() if hasattr(param, "to_local") else param
                        if local.shape != old[key].shape:
                            raise ValueError("V5/native adapter shape mismatch")
                        local.copy_(old[key].to(device=local.device, dtype=local.dtype))
                        matched.add(key)
            if matched != set(old) or len(matched) != 504:
                raise ValueError("incomplete V5 warm start")
        current = {
            k: (v.full_tensor() if hasattr(v, "full_tensor") else v).cpu().contiguous()
            for k, v in get_lora_state().adapter_state_dict().items()
        }
        if version == 0:
            initialize_optimizer_state(model, optimizer)
        opt_digest, optimizer_steps = optimizer_fingerprint(optimizer, version)
        if not all(torch.isfinite(v).all() for v in current.values()):
            raise ValueError("nonfinite model checkpoint")
        if verify:
            exported = load_file(cycle / f"train/broadcasts/step_{step}/adapter_model.safetensors")
            previous = load_file(cycle / "incoming.safetensors")
            if (
                current.keys() != exported.keys()
                or current.keys() != previous.keys()
                or not all(torch.equal(current[k], exported[k]) for k in current)
                or not any(not torch.equal(current[k], previous[k]) for k in current)
            ):
                raise ValueError("reload/export mismatch or no parameter update")
            startup = load_file(
                cycle / f"train/broadcasts/step_{step - 1}/adapter_model.safetensors"
            )
            if not all(torch.equal(startup[k], previous[k]) for k in previous):
                raise ValueError("trainer startup differs from sampled policy")
            adapter = cycle / "adapter"
            adapter.mkdir()
            save_file(
                {"base_model.model." + k: v for k, v in current.items()},
                adapter / "adapter_model.safetensors",
                metadata={"format": "pt"},
            )
            adapter_config = json.loads(
                (Path(plan["assets"]) / "v5/adapter_config.json").read_text()
            )
            adapter_config.update(
                base_model_name_or_path="Qwen/Qwen3-8B",
                revision=BASE_REVISION,
                inference_mode=True,
                lora_dropout=0.0,
            )
            write(adapter / "adapter_config.json", adapter_config)
            write(
                cycle / "verified.json",
                {
                    "step": version,
                    "optimizer_sha256": opt_digest,
                    "optimizer_steps": sorted(optimizer_steps),
                    "reload_matches_export": True,
                    "startup_matches_sampled_policy": True,
                    "adapter_sha256": file_hash(adapter / "adapter_model.safetensors"),
                },
            )
        else:
            old = load_file(policy_path(plan, step) / "adapter_model.safetensors")
            if {"base_model.model." + k for k in current} != set(old) or not all(
                torch.equal(v, old["base_model.model." + k].to(v.dtype)) for k, v in current.items()
            ):
                raise ValueError("checkpoint does not equal the sampled adapter")
            if step > 1:
                previous = json.loads((root / f"cycle-{step - 1}/verified.json").read_text())
                if previous["optimizer_sha256"] != opt_digest:
                    raise ValueError("optimizer state not preserved across cycles")
            save_file(current, cycle / "incoming.safetensors")
            score_batch(cycle, config, model)
            if step == 1:
                manager.save_to_path(
                    cycle / "init/step_0/trainer", model, [optimizer], scheduler, progress
                )
            write(
                cycle / "prepared.json",
                {
                    "version": version,
                    "optimizer_sha256": opt_digest,
                    "scorer_passed": True,
                    "optimizer_steps": optimizer_steps,
                    "batch_sha256": file_hash(cycle / "microbatches.bin"),
                },
            )
    finally:
        torch.distributed.destroy_process_group()


def score_batch(cycle, config, model):
    import msgspec
    import torch
    from prime_rl.trainer.model import forward
    from prime_rl.trainer.models.layers.lora import set_lora_num_tokens
    from prime_rl.trainer.rl.data import DataLoader, prepare_micro_batch
    from prime_rl.trainer.rl.loss import (
        selective_log_softmax,
        shift_tensor_left,
        shift_tensor_right,
    )
    from prime_rl.transports.batch import MicroBatch

    loader = DataLoader(config.output_dir, 1, 1, config.rollout_transport)
    batches = msgspec.msgpack.decode(
        (cycle / "microbatches.bin").read_bytes(), type=list[MicroBatch]
    )
    deltas = []
    model.eval()
    with torch.no_grad():
        for batch in batches:
            b = prepare_micro_batch(loader._micro_batch_to_tensor(batch), None, None)
            ids, positions = b["input_ids"].cuda(), b["position_ids"].cuda()
            set_lora_num_tokens(b["lora_num_tokens"].cuda())
            labels, temperatures = shift_tensor_left(ids), b["temperatures"].cuda()
            out = forward(
                model,
                ids,
                positions,
                labels=labels,
                temperature=temperatures,
                seq_lens=b["seq_lens"].cuda(),
            )
            lp = out.get("logprobs")
            if lp is None:
                lp = selective_log_softmax(out["logits"] / temperatures.unsqueeze(-1), labels)
            lp = shift_tensor_right(lp)
            mask = b["loss_mask"].cuda()
            deltas.append((lp[mask] - b["inference_logprobs"].cuda()[mask]).abs().cpu())
    delta = torch.cat(deltas)
    mean, maximum = delta.mean().item(), delta.max().item()
    write(cycle / "scorer.json", {"tokens": delta.numel(), "mean": mean, "max": maximum})
    if not torch.isfinite(delta).all() or mean > 0.03 or maximum > 0.25:
        raise ValueError("sampler/trainer scoring disagreement; no update")
    model.train()


async def transport_stage(cycle, step, mode):
    if mode == "batch":
        import msgspec
        from prime_rl.transports.batch import MicroBatch
        from prime_rl.transports.batch.filesystem import FileSystemBatchSender

        batches = msgspec.msgpack.decode(
            (cycle / "microbatches.bin").read_bytes(), type=list[MicroBatch]
        )
        sender = FileSystemBatchSender(cycle / "train", data_world_size=1, current_step=step)
        await asyncio.wait_for(sender.send([batches]), timeout=180)
    else:
        from prime_rl.configs.shared import FileSystemWeightBroadcastConfig
        from prime_rl.transports.weights.base import FINISHED_MARKER, WeightReceiver
        from prime_rl.utils.pathing import wait_for_path

        class ArtifactReceiver(WeightReceiver):
            async def receive(self, version):
                self._ack(version)
                directory = self.step_dir(version)
                await asyncio.wait_for(wait_for_path(directory / FINISHED_MARKER), timeout=180)
                if not (directory / "adapter_model.safetensors").is_file():
                    raise ValueError("native export missing")
                write(
                    cycle / f"artifact-{version}.json",
                    {"version": version, "artifact_received": True, "loaded_into_inference": False},
                )

        receiver = ArtifactReceiver(
            cycle / "train/broadcasts", FileSystemWeightBroadcastConfig(timeout=180), None, "envoy"
        )
        for version in (step - 1, step):
            await receiver.sync_startup(version, timeout=180)


def main():
    root, mode, step = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
    plan = validate_plan(root)
    cycle = root / f"cycle-{step}"
    if mode == "preflight":
        return preflight(plan)
    if not 1 <= step <= plan["limits"]["updates"]:
        raise ValueError("step outside run limit")
    if mode == "configure":
        return configure(plan, step)
    if mode == "load":
        return asyncio.run(load_policy(plan, step))
    if mode == "collect":
        return asyncio.run(collect_group(plan, step))
    if mode in {"batch", "weights"}:
        return asyncio.run(transport_stage(cycle, step, mode))
    if mode == "inference":
        from prime_rl.configs.inference import InferenceConfig
        from prime_rl.entrypoints.inference import inference

        return inference(
            InferenceConfig.model_validate_json((cycle / "inference.json").read_text())
        )
    from prime_rl.configs.trainer import TrainerConfig

    config = TrainerConfig.model_validate_json((cycle / "trainer.json").read_text())
    if mode in {"prepare", "verify"}:
        return checkpoint_stage(plan, step, config, verify=mode == "verify")
    if mode == "train":
        from prime_rl.trainer.rl.train import train

        prepared = json.loads((cycle / "prepared.json").read_text())
        if (
            prepared["version"] != step - 1
            or prepared["scorer_passed"] is not True
            or prepared["batch_sha256"] != file_hash(cycle / "microbatches.bin")
            or config.max_steps != step
            or config.resume.dir_step != step - 1
            or config.ckpt.skip_optimizer
            or config.ckpt.skip_scheduler
        ):
            raise ValueError("training gate failed")
        return train(config)
    raise ValueError(mode)


if __name__ == "__main__":
    main()
