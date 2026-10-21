"""One live experiment-research invocation using the existing guarded Runpod transport."""
from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

from mcp import Client

from scripts.experiment_research_mcp import create_server
from src.eval.assigned_smoke_plan import MODEL_IDENTITY, write_json
from src.eval.assigned_smoke_run import track_sandbox
from src.policies.runpod_queue import RunpodQueueTransport
from src.product.arxiv_library import ArxivLibrary
from src.product.assigned_reader import build_assigned_reader
from src.product.experiment_discovery import ExperimentDiscovery
from src.product.experiment_protocol import controller_prompt
from src.product.experiment_research import ExperimentResearcher
from src.product.paper_discovery import PaperDiscovery

ROOT = Path(__file__).resolve().parents[2]
IMAGE = "sha256:79bd8bf84bd4b7f2a32a0596cb1fd90164716c1cfca879f78829411193646382"
ENDPOINT = "q05pjrvguakieg"
LIMITS = {"questions": 1, "attempts": 1, "requests_max": 12, "tokens_max": 98304,
          "watchdog_seconds": 900, "input_per_request": 7168, "output_per_request": 1024,
          "max_gpu_workers": 1, "inference_resubmissions": 0}
QUESTION = ("What does PagedAttention change about KV-cache allocation, and what memory "
            "measurements would I still need before running two concurrent Qwen3-8B "
            "research readers? Propose the smallest useful local experiment.")
CONSTRAINTS = ("One NVIDIA RTX A5000 with 24 GB VRAM; BF16 Qwen3-8B with the existing "
               "v5 adapter, vLLM serving, up to 7168 input and 1024 output tokens per request. "
               "Two concurrent readers are a target to test, not an established working setup.")


def hashes():
    paths = list((ROOT / "src").rglob("*.py")) + [ROOT / p for p in (
        "scripts/experiment_research_mcp.py", "scripts/run_experiment_research_smoke.py",
        "scripts/arxiv_research_mcp.py", "scripts/runpod_qwen_bootstrap.py",
        "scripts/reader_comparison.py")]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


def plan():
    prompt = controller_prompt(LIMITS["requests_max"], LIMITS["tokens_max"])
    return {"schema_version": "experiment-research-live-plan-v1", "endpoint": ENDPOINT,
            "limits": LIMITS, "question": QUESTION, "constraints": CONSTRAINTS,
            "sandbox_image": IMAGE, "model": MODEL_IDENTITY,
            "implementation_sha256": hashes(),
            "controller_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "hypothesis": "Bounded Qwen search and reading yields useful ML experiment evidence.",
            "decision_rule": "At least one relevant supported finding, accurate applicability "
                "limits, a usable proposed test, and no critical factual error or budget breach.",
            "review_requirements": ["Check each decisive claim against its cited passages",
                "Keep literature measurements separate from the user's unmeasured setup",
                "Review the proposed local control and measurements"],
            "semantic_review": "pending; model assistance must not be labeled human review",
            "fine_tuning_comparison": False, "provider_dollar_cap": False}


def checked_plan(output, name="plan.json"):
    saved = json.loads((output / name).read_text())
    if saved != plan():
        raise ValueError("live question, limits, prompts or implementation differ from frozen plan")
    return saved


def run(output, *, tokenizer, key, transport_factory=RunpodQueueTransport,
        reader_factory=build_assigned_reader, discovery=None, plan_name="plan.json"):
    saved = checked_plan(output, plan_name)
    with (output / "WORKER_STARTED").open("x"):
        pass
    audit = output / "research.requests.jsonl"
    with audit.open("x"):
        pass
    transport = transport_factory(ENDPOINT, key, tokenizer, audit,
                                  requests_max=12, wait_seconds=900)
    library = ArxivLibrary(output / "library")
    discovery = discovery or ExperimentDiscovery(PaperDiscovery(library.root / "discovery-cache"))

    def reader(budget):
        instance = reader_factory("qwen", qwen_endpoint=f"https://api.runpod.ai/v2/{ENDPOINT}",
            qwen_model="envoy", qwen_checkpoint=MODEL_IDENTITY["adapter"] + "@"
                + MODEL_IDENTITY["adapter_revision"], max_steps=6,
            reader_guidance="focused-v1", sandbox_image=IMAGE, policy_transport=budget)
        instance.model_identity.update(MODEL_IDENTITY)
        track_sandbox(instance, output)
        return instance

    def checkpoint(result):
        temporary = output / "progress.tmp"
        write_json(temporary, result)
        temporary.replace(output / "progress.json")

    worker = ExperimentResearcher(library, discovery, reader, tokenizer, transport,
        requests_max=12, tokens_max=98304, checkpoint=checkpoint)

    async def invoke():
        async with Client(create_server(library, discovery, worker,
                                        runs_root=output / "reports")) as client:
            return await client.call_tool("research_experiment", {
                "question": saved["question"], "constraints": saved["constraints"]})

    try:
        reply = asyncio.run(invoke())
        result = reply.structured_content
        write_json(output / "mcp.json", {"is_error": reply.is_error, "result": result,
                                        "transport": "in_process", "tool": "research_experiment"})
        if reply.is_error or not isinstance(result, dict):
            raise ValueError("live MCP invocation failed; raw response retained")
        rows = [json.loads(line) for line in audit.read_text().splitlines()]
        if (result["status"] == "inference_failed"
                or any(row.get("event") == "failed" for row in rows)
                or any(paper.get("status") in {"error", "unavailable", "unverified"}
                       for paper in result["papers"])):
            raise RuntimeError("live infrastructure/reader failure; partial result retained")
        return result
    finally:
        transport.audit.close()
