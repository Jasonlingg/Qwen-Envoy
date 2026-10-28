"""Separate inference-only checklist ablation; never changes the frozen RL protocol.

prepare is offline; probe runs fixed actions in Docker, not model inference.
run consumes an already provisioned, guarded native engine; it never rents GPUs.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.passage_pilot_replay import PROMPT, frozen_runtime
from src.eval.reader_correction_replay import cleanup, track
from src.eval.reader_improvement import file_hash, inside, load
from src.policies.openai_compatible import OpenAICompatiblePolicy
from src.product.harness_v2 import clean_v2_action
from src.product.packet_verifier import verify_investigation_packet
from src.product.paper_assignment import assignment_observation, paper_assignment
from src.product.passage_scope import PASSAGES_V2
from src.product.qwen_investigator import QwenInvestigator
from src.research.agent import load_snapshot
from src.training.prime_budget import loopback_engine, proxy
from src.training.prime_run import BASE_REVISION, V5_REVISION, V5_SHA, write
from src.training.reader_data import ROOT, TRAINING_REVISION, load_training_bundle, score_episode

CHECKLIST = """

Before acting, use this checklist internally; output only the allowed action.
1. Identify every fact requested by the question.
2. Inspect the supplied passages and locate evidence for those facts.
3. Before submitting, check that the answer covers every requested part and that
   names, numbers, units, comparisons and negations agree with the evidence.
4. Cite only inspected spans that support the answer. If the inspected scope
   lacks a requested fact, report the specific gap using the existing protocol.
"""
PROMPTS = {"control": PROMPT, "checklist": PROMPT + CHECKLIST}
BUNDLE = ROOT / "data/rl/reader-facts-v1"
TOKENIZER = ROOT / "out/reader-prime-pilot-v2-20261025/local-assets/base" / BASE_REVISION


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def selection():
    manifest, roles, contexts, cards, ids, reward = load_training_bundle(BUNDLE, TRAINING_REVISION)
    random.Random(42).shuffle(ids)
    by_id = {c["case_id"]: c for c in roles["train"]}
    return manifest, [by_id[c] for c in ids[:6]], contexts, cards, reward


def schedule(cases):
    for index, case in enumerate(cases):
        for sample in range(4):
            arms = ("control", "checklist") if (index + sample) % 2 == 0 else (
                "checklist", "control")
            for arm in arms:
                yield case, sample, arm, 42 + index * 4 + sample


def initial_messages(case, prompt):
    manifest, docs = load_snapshot(inside(BUNDLE, case["snapshot"]))
    import json

    observation = (f"{prompt}\nQuestion: {case['question']}\n"
                   + assignment_observation(paper_assignment(manifest, docs))
                   + "\nController passage ranges (metadata only): " + json.dumps(case["scope"]))
    return [{"role": "system", "content": prompt}, {"role": "user", "content": observation}]


def specification():
    manifest, cases, _, _, reward = selection()
    runtime = frozen_runtime()
    # No reference answers, accepted aliases, source-review judgments or gold spans.
    tasks = [{k: c[k] for k in ("case_id", "question_id", "question", "doc_id", "scope",
                                "snapshot")} for c in cases]
    plan = {
        "schema": "reader-checklist-ablation-v1", "status": "prepared_not_executed",
        "hypothesis": "A generic checklist increases healthy mixed-reward groups per reserved "
                      "token, without reducing complete source-supported answer quality.",
        "selection": "first six eligible training questions after Python Random(42) shuffle; "
                     "development only, previously exposed, no held-out claim",
        "base_revision": BASE_REVISION, "v5_revision": V5_REVISION, "adapter_sha256": V5_SHA,
        "source_revision": manifest["source_revision"], "reward_version": reward,
        "bundle_sha256": file_hash(BUNDLE / "manifest.json"),
        "reward_revision_sha256": file_hash(TRAINING_REVISION),
        "tasks": tasks,
        "prompt_sha256": {a: sha(p) for a, p in PROMPTS.items()},
        "implementation_sha256": file_hash(Path(__file__)),
        "dependencies_sha256": {p: file_hash(ROOT / p) for p in (
            "src/training/reader_data.py", "src/training/prime_budget.py",
            "src/training/prime_run.py", "src/training/reader_evaluation.py")},
        "sandbox_image": runtime["sandbox_image"],
        "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0,
                     "max_tokens": 1024, "thinking": False, "steps": 4, "retries": 0},
        "schedule": [{"case_id": c["case_id"], "sample": s, "arm": a, "seed": seed}
                     for c, s, a, seed in schedule(cases)],
        "limits": {"requests": 576, "reserved_tokens": 576 * 8192, "seconds": 1500},
        "budget_proposal": {"estimated_ceiling_usd": 2.0, "maximum_owned_seconds": 2700,
                            "maximum_combined_hourly_usd": 1.90,
                            "provider_enforced_dollar_cap": False, "paid_authorized": False},
        "optimizer_updates": 0, "training_eligible": False, "human_review": None,
        "decision_rule": "Complete all 48 healthy episodes; checklist gains >=1 mixed group "
                         "and mixed groups per reserved token, reserved tokens <=1.25x control, "
                         "no execution-failure increase; source-review all answers, with no "
                         "supported-answer regression or unsupported-answer increase. Otherwise "
                         "retain control or report inconclusive; no paid retry.",
        "interpretation": "Rollout diagnostic only: not RL convergence, no-hint transfer or "
                          "held-out improvement. Uniform-one groups are solved, not failures. "
                          "Reward is mechanical; exact quotations prove provenance, not support.",
    }
    return plan


def prepare(output):
    plan = specification()
    output.mkdir(parents=True, exist_ok=False)
    for arm, prompt in PROMPTS.items():
        (output / f"{arm}.txt").write_text(prompt)
    write(output / "plan.json", plan)
    return plan


def validate(output):
    plan = load(output / "plan.json")
    _, cases, contexts, cards, _ = selection()
    if (plan != specification()
            or any((output / f"{a}.txt").read_text() != p for a, p in PROMPTS.items())):
        raise ValueError("ablation inputs or implementation changed; prepare a fresh experiment")
    return plan, cases, contexts, cards


def collect(case, arm, transport, output, identity):
    """A separate prompt experiment; existing production/RL collectors stay frozen."""
    prompt = PROMPTS[arm]
    policy = OpenAICompatiblePolicy(
        endpoint="http://127.0.0.1/no-direct-http", api_key="unused", model="envoy",
        transport=transport, system_prompt=prompt, action_cleaner=clean_v2_action,
        temperature=1.0, max_tokens=1024,
        extra_body={"top_p": 1.0, "chat_template_kwargs": {"enable_thinking": False}},
    )
    reader = QwenInvestigator(
        lambda: policy, identity, max_steps=4, image=frozen_runtime()["sandbox_image"],
        source_domain="public_papers", harness_version="v2", reader_mode="assigned",
        reader_guidance=PASSAGES_V2,
    )
    # Experiment-local override: identical policy, initial observation and packet identity.
    reader.system_prompt = prompt
    output.mkdir(parents=True, exist_ok=False)
    owned = []
    track(reader, output, owned)
    snapshot = inside(BUNDLE, case["snapshot"])
    try:
        packet = reader.investigate(case["question"], snapshot,
                                    passage_windows=case["scope"]["windows"])
    finally:
        proofs = cleanup(owned)
        write(output / "cleanup.json", {"containers": proofs,
                                       "all_absent": bool(proofs) and all(
                                           p["absent"] for p in proofs)})
        write(output / "payloads.json", transport.payloads)
    manifest, docs = load_snapshot(snapshot)
    write(output / "packet.json", packet)
    verified = None
    if packet["status"] not in {"error", "unavailable"}:
        verified = verify_investigation_packet(
            packet, manifest, docs, allowed_prompt_sha256s=frozenset({sha(prompt)}),
            expected_reader_mode="assigned", expected_reader_guidance=PASSAGES_V2,
            expected_passage_windows=case["scope"]["windows"],
        )
    episode = {"case_id": case["case_id"], "question_id": case["question_id"], "packet": packet,
               "verified": verified, "training_eligible": False,
               "infrastructure_ok": bool(proofs) and all(p["absent"] for p in proofs)
               and packet["status"] not in {"error", "unavailable"}}
    write(output / "rollout.json", episode)
    return episode


def probe(output, tokenizer):
    """Scripted sandbox and initial-prefix checks; zero model calls or learning."""
    from renderers import create_renderer
    from renderers.configs import DefaultRendererConfig
    from transformers import AutoTokenizer

    from src.training.reader_evaluation import check_native_sources, check_tokenizer

    _, cases, _, _ = validate(output)
    check_native_sources()
    check_tokenizer(tokenizer)
    renderer = create_renderer(AutoTokenizer.from_pretrained(tokenizer, local_files_only=True),
                               DefaultRendererConfig(enable_thinking=False))
    lengths = []
    for case in cases:
        for arm, prompt in PROMPTS.items():
            ids = renderer.render(initial_messages(case, prompt),
                                  add_generation_prompt=True).token_ids
            if len(ids) > 7168:
                raise ValueError("initial prompt exceeds reserved context")
            lengths.append({"case_id": case["case_id"], "arm": arm, "initial_tokens": len(ids)})
    packets = []
    case = cases[0]
    actions = case["scope"]["read_calls"] + [
        "submit_gap(missing='This scripted control makes no answer judgment', "
        "follow_up_query='Find evidence for the requested facts')"]

    class ScriptedTransport:
        def __init__(self):
            self.payloads = []

        def __call__(self, url, payload, headers, timeout):
            index = len(self.payloads)
            self.payloads.append(copy.deepcopy(payload))
            return {"choices": [{"message": {"content": actions[index]}}]}

    for arm, prompt in PROMPTS.items():
        transport = ScriptedTransport()
        episode = collect(case, arm, transport, output / "probe" / arm,
                          {"data_kind": "fixed_actions", "model_inference": False})
        packet = episode["packet"]
        if (not episode["infrastructure_ok"] or packet["status"] != "no_evidence"
                or packet["inspection_steps"] != 2
                or packet["system_prompt_sha256"] != sha(prompt)
                or transport.payloads[0]["messages"] != initial_messages(case, prompt)):
            raise ValueError("scripted sandbox/prompt control failed")
        packets.append({"arm": arm, "prompt_sha256": sha(prompt),
                        "source_inspections": 2, "containers_absent": True})
    receipt = {"status": "passed_scripted_probe", "new_model_calls": 0,
               "optimizer_updates": 0, "initial_tokens": lengths, "packets": packets,
               "limitation": "Fixed actions do not test model choices, checklist benefit, "
                             "semantic correctness or RL convergence."}
    write(output / "probe.json", receipt)
    return receipt


class NativeTransport:
    """Pinned native renderer -> existing durable request/token/time-budget proxy."""

    def __init__(self, url, renderer, seed):
        self.url, self.renderer, self.seed = url, renderer, seed
        self.payloads, self.usage = [], []

    def __call__(self, url, payload, headers, timeout):
        import httpx
        from openai import AsyncOpenAI
        from renderers.client import generate

        if len(self.payloads) >= 12:
            raise RuntimeError("episode request ceiling exhausted")
        self.payloads.append(copy.deepcopy(payload))

        async def request():
            async with AsyncOpenAI(
                base_url=self.url, api_key="unused", max_retries=0,
                http_client=httpx.AsyncClient(trust_env=False, follow_redirects=False, timeout=30),
            ) as client:
                return await generate(
                    client=client, renderer=self.renderer, messages=payload["messages"],
                    model="envoy", max_prompt_len=7168,
                    sampling_params={"max_tokens": 1024, "temperature": 1.0, "top_p": 1.0,
                                     "top_k": -1, "min_p": 0.0, "seed": self.seed},
                )
        result = asyncio.run(request())
        self.usage.append({"prompt_tokens": len(result["prompt_ids"]),
                           "completion_tokens": len(result["completion_ids"])})
        usage = self.usage[-1]
        return {"model": "envoy", "usage": {**usage, "total_tokens": sum(usage.values())},
                "choices": [{"message": {"content": result["content"],
                                         "reasoning_content": result.get("reasoning_content")},
                             "finish_reason": result.get("finish_reason")}]}


def diagnostics(rows):
    result = {}
    for arm in PROMPTS:
        groups, reserved = {}, 0
        for row in rows:
            if row["arm"] == arm:
                groups.setdefault(row["case_id"], []).append(row["reward"]["reward"])
                reserved += sum(u["prompt_tokens"] + 1024 for u in row["usage"])
        complete = [g for g in groups.values() if len(g) == 4]
        mixed = sum(len(set(g)) > 1 for g in complete)
        result[arm] = {"complete_groups": len(complete), "mixed_groups": mixed,
                       "uniform_one_groups": sum(all(r == 1 for r in g) for g in complete),
                       "uniform_zero_groups": sum(all(r == 0 for r in g) for g in complete),
                       "reserved_tokens_for_recorded_healthy_episodes": reserved,
                       "mixed_groups_per_million_reserved_tokens":
                       mixed * 1_000_000 / reserved if reserved else None}
    return result


def run(output, engine, tokenizer, adapter, authorization):
    """Inference only. External deployment must own rental deadline and deletion."""
    plan, cases, contexts, cards = validate(output)
    approval = load(authorization)
    if (approval.get("approved") is not True
            or approval.get("plan_sha256") != file_hash(output / "plan.json")
            or approval.get("estimated_ceiling_usd") != 2.0
            or not 0 < approval.get("combined_hourly_usd", float("inf")) <= 1.90
            or approval.get("owned_resource_seconds") != 2700
            or not approval.get("owned_pod_id")
            or approval.get("external_watchdog_verified") is not True):
        raise ValueError("plan-bound $2 approval and externally verified resource guard required")
    deadline = approval["resource_deadline_epoch"]
    if (not math.isfinite(deadline) or deadline <= time.time() + 300
            or deadline > time.time() + 2700):
        raise ValueError("invalid resource deadline; retain five minutes for export/cleanup")
    if (output / "started.json").exists():
        raise ValueError("no automatic resume or paid retry")
    from renderers import create_renderer
    from renderers.configs import DefaultRendererConfig
    from transformers import AutoTokenizer

    from src.training.reader_evaluation import check_native_sources, check_tokenizer, model_root

    check_native_sources()
    check_tokenizer(tokenizer)
    if file_hash(adapter / "adapter_model.safetensors") != V5_SHA:
        raise ValueError("only original pinned V5 is allowed")
    engine = loopback_engine(engine)
    model_root(engine, approval["engine_adapter_path"])
    renderer = create_renderer(AutoTokenizer.from_pretrained(tokenizer, local_files_only=True),
                               DefaultRendererConfig(enable_thinking=False))
    write(output / "started.json", {"deadline": min(deadline - 300, time.time() + 1500)})
    summary = {"status": "running", "optimizer_updates": 0, "training_eligible": False,
               "semantic_review": "pending", "human_review": None, "episodes": []}
    try:
        with proxy(output, plan, {"sha256": V5_SHA}, engine=engine) as url:
            for case, sample, arm, seed in schedule(cases):
                transport = NativeTransport(url, renderer, seed)
                start = time.monotonic()
                episode = collect(case, arm, transport, output / "episodes" /
                                  f"{case['case_id']}-{sample}-{arm}",
                                  {"adapter_sha256": V5_SHA, "weights_verified": False,
                                   "experiment": "checklist_ablation", "arm": arm})
                score = score_episode(episode, contexts[case["case_id"]], cards[case["case_id"]])
                summary["episodes"].append({"case_id": case["case_id"], "sample": sample,
                    "arm": arm, "seed": seed, "reward": score, "usage": transport.usage,
                    "seconds": time.monotonic() - start, "supported_answer": None,
                    "unsupported_answer": None, "review_notes": None})
                summary["diagnostics"] = diagnostics(summary["episodes"])
                write(output / "summary.json", summary)
        summary["status"] = "collected_pending_source_review"
    except Exception as exc:
        summary.update(status="incomplete", error=type(exc).__name__)
        raise
    finally:
        write(output / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "probe", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine")
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--authorization", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.output)
    elif args.command == "probe":
        probe(args.output, args.tokenizer)
    elif args.engine and args.adapter and args.authorization:
        run(args.output, args.engine, args.tokenizer, args.adapter, args.authorization)
    else:
        parser.error("run needs --engine, --adapter and --authorization; provisioning is separate")


if __name__ == "__main__":
    main()
