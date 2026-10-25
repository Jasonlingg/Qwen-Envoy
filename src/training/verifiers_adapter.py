"""Verifiers 0.4 evaluation adapter; the frozen investigator still owns Docker.

Original episodes remain untrainable; a local option checks native token transport.
The separate one-update GPU smoke passed; generic training remains blocked.
The bounded sequential driver owns candidate admission and native scorer gates.
"""

from __future__ import annotations

import asyncio
import copy
import importlib.metadata
import json
import math
from pathlib import Path
from urllib.parse import urlsplit

import httpx

if importlib.metadata.version("verifiers") != "0.4.0":
    raise ImportError("Use the isolated requirements-verifiers.txt environment (verifiers==0.4.0)")

import verifiers.v1 as vf
from pydantic import Field

from src.eval.passage_pilot_replay import PROMPT, PROMPT_SHA
from src.eval.reader_improvement import file_hash, inside
from src.training.reader_data import ROOT, load_bundle, load_training_bundle, score_episode
from src.training.reader_rollout import MAX_ACTION_TOKENS, TEMPERATURE, collect

PLUGIN = "envoy_reader"
__all__ = ["ReaderTaskset", "ReaderHarness", "ReaderEnv"]


class ReaderTaskData(vf.TaskData):
    case_id: str
    bundle: Path
    manifest_sha256: str
    evaluation_session_sha256: str | None = None
    reward_revision: Path | None = None
    reward_revision_sha256: str | None = None


class ReaderState(vf.State):
    episode: dict | None = None


def task_case(data):
    if data.evaluation_session_sha256 is not None:
        raise ValueError("held-out evaluation cannot receive a training reward")
    bundle = inside(ROOT, data.bundle)
    if file_hash(bundle / "manifest.json") != data.manifest_sha256:
        raise ValueError("task's source bundle changed")
    if data.reward_revision is not None:
        revision = inside(ROOT, data.reward_revision)
        if file_hash(revision) != data.reward_revision_sha256:
            raise ValueError("task's training reward revision changed")
        _, roles, contexts, cards, selected, _ = load_training_bundle(bundle, revision)
        if data.case_id not in selected:
            raise ValueError("held question cannot receive a revised training reward")
    else:
        if data.reward_revision_sha256 is not None:
            raise ValueError("reward revision identity requires a revision")
        _, roles, contexts, cards = load_bundle(bundle)
    case = next((c for c in roles["train"] if c["case_id"] == data.case_id), None)
    if (
        case is None
        or data.id != case["question_id"]
        or data.prompt != case["question"]
        or data.system_prompt != PROMPT
    ):
        raise ValueError("only unchanged training-role questions and the frozen prompt are allowed")
    return bundle, case, contexts[data.case_id], cards[data.case_id]


class ReaderTask(vf.Task[ReaderTaskData, ReaderState]):
    @property
    def key(self):
        return self.data.case_id

    @vf.reward
    async def short_fact(self, trace):
        _, _, context, card = task_case(self.data)
        if trace.state.episode is None:
            raise ValueError("no completed Docker episode; reward is unavailable")
        return score_episode(trace.state.episode, context, card)["reward"]


class ReaderTasksetConfig(vf.TasksetConfig):
    id: str = PLUGIN
    bundle: Path = Path("data/rl/reader-facts-v1")
    reward_revision: Path | None = None


class ReaderTaskset(vf.Taskset[ReaderTask, ReaderTasksetConfig]):
    def load(self):
        bundle = inside(ROOT, self.config.bundle)
        revision = self.config.reward_revision
        if revision is None:
            _, roles, _, _ = load_bundle(bundle)
            selected = {c["case_id"] for c in roles["train"]}
            revision_sha = None
        else:
            revision = inside(ROOT, revision)
            _, roles, _, _, selected, _ = load_training_bundle(bundle, revision)
            revision_sha = file_hash(revision)
        for case in roles["train"]:
            if case["case_id"] not in selected:
                continue
            yield ReaderTask(
                ReaderTaskData(
                    id=case["question_id"],
                    case_id=case["case_id"],
                    bundle=bundle,
                    manifest_sha256=file_hash(bundle / "manifest.json"),
                    reward_revision=revision.relative_to(ROOT) if revision else None,
                    reward_revision_sha256=revision_sha,
                    prompt=case["question"],
                    system_prompt=PROMPT,
                ),
                self.config.task,
            )


class ReaderHarnessConfig(vf.HarnessConfig):
    id: str = PLUGIN
    output: Path = Path("out/verifiers-reader")
    token_contract_only: bool = False
    evaluation_session: Path | None = None
    model_identity: dict = Field(
        default_factory=lambda: {"backend": "verifiers_eval", "weights_verified": False}
    )


class InterceptTransport:
    """Forward exact production payloads to VF's local interception server."""

    def __init__(self, endpoint, secret):
        parsed = urlsplit(endpoint)
        if parsed.scheme != "http" or parsed.hostname != "127.0.0.1":
            raise ValueError("expected Verifiers' loopback interception endpoint")
        self.endpoint, self.secret = endpoint.rstrip("/"), secret
        self.payloads = []

    def __call__(self, url, payload, headers, timeout):
        if len(self.payloads) >= 12:
            raise ValueError("compatibility request ceiling reached")
        self.payloads.append(copy.deepcopy(payload))
        with httpx.Client(trust_env=False, follow_redirects=False, timeout=30) as client:
            with client.stream(
                "POST",
                self.endpoint + "/chat/completions",
                json=payload,
                headers={"Authorization": "Bearer " + self.secret},
            ) as response:
                response.raise_for_status()
                body = bytearray()
                for chunk in response.iter_bytes():
                    body.extend(chunk)
                    if len(body) > 2 * 1024 * 1024:
                        raise ValueError("interception response exceeds 2 MiB")
        return json.loads(body)


class ReaderHarness(vf.Harness[ReaderHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    # Only the trusted controller runs in VF's host runtime. All model-authored
    # Python is executed by collect() in the unchanged, pinned Docker sandbox.
    NEEDS_CONTAINER = False
    EXECUTES_CODE = False

    async def launch(self, ctx, trace, runtime, endpoint, secret, mcp_urls, data):
        trace.agent.trainable = False
        if not isinstance(ctx.client, vf.EvalClientConfig):
            if not (self.config.token_contract_only
                    and isinstance(ctx.client, vf.TrainClientConfig)):
                raise ValueError("evaluation-only adapter: prime-rl training is not yet verified")
            renderer = ctx.client.renderer
            tokenizer_path = Path(ctx.client.renderer_model_name or "")
            if (renderer is None or renderer.name != "default"
                    or (renderer.model_extra or {}).get("enable_thinking") is not False
                    or not tokenizer_path.is_dir() or tokenizer_path.name
                    != "b968826d9c46dd6066d109eabc6255188de91218"):
                raise ValueError(
                    "token contract requires the explicit non-thinking default renderer")
        parsed = urlsplit(ctx.client.base_url)
        if parsed.scheme != "http" or parsed.hostname != "127.0.0.1":
            raise ValueError("compatibility stage permits only an explicit loopback model endpoint")
        if ctx.model != "envoy" or mcp_urls:
            raise ValueError("preserve the envoy model alias and existing Python tool protocol")
        if self.config.evaluation_session is None:
            bundle, case, _, _ = task_case(data)
        else:
            from src.training.reader_evaluation import evaluation_case

            bundle, case = evaluation_case(self.config.evaluation_session, data)
        output = self.config.output / trace.id
        transport = InterceptTransport(endpoint, secret)
        work = asyncio.create_task(
            asyncio.to_thread(
                collect,
                case,
                inside(bundle, case["snapshot"]),
                transport,
                output,
                self.config.model_identity,
            )
        )
        try:
            episode = await asyncio.shield(work)
        except asyncio.CancelledError:
            # A cancelled to_thread() does not stop its worker. Wait for the
            # bounded investigator and its Docker finally-cleanup before returning.
            await asyncio.shield(work)
            raise
        trace.state.episode = episode
        trace.info["envoy"] = {
            "output": str(output),
            "prompt_sha256": PROMPT_SHA,
            "training_eligible": False,
            "weights_verified": False,
            "token_contract_only": self.config.token_contract_only,
        }
        if not episode["infrastructure_ok"]:
            raise ValueError("Docker/provider failure: discard rollout, do not assign zero reward")
        return vf.ProgramResult(exit_code=0, stdout="", stderr="")


def complete_reader_group(episodes, expected_size):
    """TrainSink admission callback: an infrastructure failure voids its whole group.

    Compose with the native train source's on_result callback. This does not make
    evaluation or token-contract traces trainable, or establish weight identity.
    """
    if expected_size < 2 or len(episodes) != expected_size:
        return False
    tasks = {(e.task.key, e.task.hash) for e in episodes}
    groups = {e.group.id if e.group else None for e in episodes}
    if len(tasks) != 1 or None in next(iter(tasks)) or len(groups) != 1 or None in groups:
        return False
    policies = set()
    for episode in episodes:
        if (not episode.ok or len(episode.traces) != 1
                or not isinstance(episode.run, vf.TrainRunInfo)
                or episode.run.work.type != "train"):
            return False
        policy = episode.run.work.policy
        if policy is None or policy.start != policy.end:
            return False
        policies.add((episode.run.id, episode.run.work.step, policy.start))
        trace = episode.traces[0]
        reward = trace.rewards.get("short_fact")
        if (not trace.ok or trace.has_error or not trace.agent.trainable
                or (trace.task.key, trace.task.hash) != (episode.task.key, episode.task.hash)
                or set(trace.rewards) != {"short_fact"} or reward is None
                or reward.weight != 1.0 or not math.isfinite(reward.score)):
            return False
    return len(policies) == 1


def local_agent_config(base_url, output, *, model_identity=None):
    """No provider credentials, cloud runtime, retries or prompt rewriting defaults."""
    return vf.AgentConfig(
        model="envoy",
        client=vf.EvalClientConfig(base_url=base_url, api_key_var="ENVOY_VERIFIERS_UNUSED"),
        harness=ReaderHarnessConfig(
            output=output,
            **({"model_identity": model_identity} if model_identity is not None else {}),
        ),
        runtime=vf.SubprocessConfig(),
        network_notice=False,
        sampling=vf.SamplingConfig(
            temperature=TEMPERATURE,
            top_p=1.0,
            max_tokens=MAX_ACTION_TOKENS,
            chat_template_kwargs={"enable_thinking": False},
        ),
        max_turns=12,
        retries=vf.RetryConfig(max_retries=0),
    )


class ReaderEnvConfig(vf.SingleAgentEnvConfig):
    # The generic VF default is a cloud runtime. This controller owns a separate
    # pinned Docker sandbox, so even CLI discovery must default to the local host.
    agent: vf.AgentConfig = vf.AgentConfig(runtime=vf.SubprocessConfig(), network_notice=False)


class ReaderEnv(vf.Env[ReaderEnvConfig]):
    async def run(self, task, agents):
        await agents.agent.run(task)
