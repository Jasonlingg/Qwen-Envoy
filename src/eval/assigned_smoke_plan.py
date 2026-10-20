"""Prepare two development cases without credentials, inference or source fetching."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from scripts.runpod_qwen_bootstrap import (
    ADAPTER,
    ADAPTER_REVISION,
    ADAPTER_SHA256,
    BASE,
    BASE_REVISION,
    SUBFOLDER,
)
from src.env.tools import TOOL_PREAMBLE
from src.env.tools_v2 import HARNESS_V2_SHA256
from src.product.arxiv_library import ArxivLibrary
from src.product.paper_assignment import assignment_hash, assignment_observation, paper_assignment
from src.product.qwen_investigator import effective_system_prompt
from src.research.agent import load_snapshot

ROOT = Path(__file__).resolve().parents[2]
PAPER = "2211.17192v2"
EXPECTED_SOURCE_SHA256 = "fa425ba4ebcd43f7d6b66d49c1d74225adf894a2f66aa8002d1a53aab54d06ff"
ANCHOR = "We measure walltime improvements with a batch size of 1 on a single TPU-v4"
CASES = [
    {"case_id": "assigned-batch-size-v1", "kind": "answerable",
     "question": "What batch size was used for the walltime measurements?"},
    {"case_id": "assigned-a5000-batch8-v1", "kind": "target_setup_gap",
     "question": "From the available extracted text, what walltime speedup, if any, is "
                 "established specifically for speculative decoding on an NVIDIA RTX A5000 "
                 "at batch size 8?"},
]
LIMITS = {"cases": 2, "attempts_per_case": 1, "action_steps_per_case": 6,
          "model_requests_per_case": 6, "model_requests_total": 12,
          "input_tokens_per_request": 7168, "output_tokens_per_request": 1024,
          "input_tokens_reserved_total": 86016, "output_tokens_reserved_total": 12288,
          "whole_session_watchdog_seconds": 900, "provider_execution_timeout_ms": 120000,
          "inference_resubmissions": 0, "max_active_gpu_workers": 1}
MODEL_IDENTITY = {"base": BASE, "base_revision": BASE_REVISION, "adapter": ADAPTER,
                  "adapter_revision": ADAPTER_REVISION, "subfolder": SUBFOLDER,
                  "adapter_sha256": ADAPTER_SHA256, "temperature": 0.0, "top_p": 1.0,
                  "seed": 42, "thinking": False}
SOURCE_FILES = [
    "src/eval/assigned_smoke_plan.py", "src/eval/assigned_smoke_check.py",
    "scripts/prepare_assigned_reader_smoke.py", "scripts/runpod_qwen_bootstrap.py",
    "scripts/run_known_paper_diagnostic.py", "src/eval/native_pilot_client.py",
    "src/product/assigned_reader.py", "src/product/assigned_actions.py",
    "src/product/paper_assignment.py", "src/product/reader_gaps.py",
    "src/product/qwen_investigator.py", "src/product/packet_verifier.py",
    "src/product/parallel_readers.py", "src/product/arxiv_service.py",
    "src/product/arxiv_library.py", "src/product/arxiv_mcp_config.py",
    "scripts/arxiv_research_mcp.py", "src/policies/runpod_queue.py",
    "src/policies/openai_compatible.py", "src/env/repl.py", "src/env/tools.py",
    "src/env/tools_v2.py", "src/product/harness_v2.py", "src/product/submit_call.py",
]


def digest(value) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def source_digest(document: dict) -> str:
    # Match the original diagnostic's source pin, including JSON ASCII escaping.
    return hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def initial_payload(prompt: str, assignment: dict, question: str) -> dict:
    return {"model": "envoy", "max_tokens": 1024, "temperature": 0.0, "top_p": 1.0,
            "seed": 42, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "system", "content": prompt}, {"role": "user", "content":
                f"{prompt}\nQuestion: {question}\n" + assignment_observation(assignment)}]}


def prepare(library_root: Path, output: Path, *, sandbox_image: str, tokenizer,
            endpoint_id: str = "q05pjrvguakieg", implementation_commit: str = "unrecorded") -> dict:
    if output.exists():
        raise FileExistsError("use a new directory; frozen sessions are never overwritten")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", sandbox_image):
        raise ValueError("pin the Docker image by sha256 digest")
    if not re.fullmatch(r"[a-z0-9]+", endpoint_id):
        raise ValueError("invalid endpoint identifier")
    library = ArxivLibrary(library_root.expanduser().resolve(strict=True))
    doc = library.document(PAPER)  # Cached-only API: never open/fetch a paper here.
    if source_digest(doc) != EXPECTED_SOURCE_SHA256:
        raise ValueError("cached paper differs from the previously pinned source")
    snapshot = library.view([PAPER])
    manifest, docs = load_snapshot(snapshot)
    assignment = paper_assignment(manifest, docs)
    prompt = effective_system_prompt("public_papers", "default", "v2", reader_mode="assigned")
    model_inputs = []
    for case in CASES:
        payload = initial_payload(prompt, assignment, case["question"])
        tokens = tokenizer.apply_chat_template(payload["messages"], tokenize=True,
            add_generation_prompt=True, enable_thinking=False, return_dict=False)
        if (not isinstance(tokens, list) or any(type(t) is not int for t in tokens)
                or len(tokens) > LIMITS["input_tokens_per_request"]):
            raise ValueError("initial input must be a flat token list within the input limit")
        model_inputs.append({"case_id": case["case_id"], "payload": payload,
                             "initial_prompt_tokens": len(tokens)})
    text = doc["text"]
    position = text.index(ANCHOR)
    boundary = text.rfind("\n\n", 0, position)
    start = 0 if boundary < 0 else boundary + 2
    end = text.find("\n\n", position)
    end = len(text) if end < 0 else end
    references = {"reviewer_kind": "assistant_prepared", "human_review": "not_performed",
        "held_out": False, "sent_to_model": False, "source_review_scope":
        "Selected cached HTML paragraphs and context; figures/tables may be omitted. "
        "No-match string checks are not proof of absence in the full paper or literature.",
        "source_context": {"start": start, "end": end, "text": text[start:end]},
        "cases": [
            {"case_id": CASES[0]["case_id"], "reference_answer": "1",
             "review_requirements": "Correct batch-size claim, named paper citation and exact "
                                    "supporting spans after source inspection."},
            {"case_id": CASES[1]["case_id"], "reference_constraint":
             "Do not invent or transfer a measured speedup to A5000/batch8. The inspected context "
             "reports TPU-v4/batch1. Missing measurements do not mean zero/negative speedup. "
             "Report an unresolved target-setup gap and useful next query. "
             "Cited partial findings are allowed if clearly limited to their measured setup.",
             "allowed_terminal_statuses": ["no_evidence", "evidence_found"],
             "incomplete_with_gap": "partial, not a pass"}],
        "literal_no_match_counts": {term: text.casefold().count(term.casefold())
                                    for term in ["A5000", "GPU"]}}
    review_template = {"human_review": "not_performed",
        "instruction": "Save later reviews separately; keep this frozen template blank.",
        "cases": [{"case_id": c["case_id"], "reviewer_kind": None, "answer_supported": None,
                   "gap_appropriate": None, "unsupported_target_speedup": None,
                   "verdict": None, "notes": None} for c in CASES]}
    plan = {"schema_version": "assigned-reader-smoke-plan-v1", "status": "prepared_not_run",
        "split": "two assistant-authored development cases; familiar paper; not held out",
        "hypothesis": "v5 can inspect its assigned paper under the new protocol, return a "
                      "supported fact, and report a target-setup gap without fabrication.",
        "expected_signal": "A source-supported batch-size answer and a qualified gap report.",
        "decision_rule": "Proceed to another development comparison only if both cases meet "
            "their source/claim review requirements, with zero execution errors, repeated-call "
            "rejections or budget violations. Mechanical span checks alone cannot pass a case.",
        "cases": CASES, "limits": LIMITS, "paper_version": PAPER,
        "source_sha256": source_digest(doc),
        "snapshot": str(snapshot), "corpus_hash": manifest["corpus_hash"],
        "paper_assignment": assignment, "assignment_sha256": assignment_hash(assignment),
        "system_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "tool_preamble_sha256": hashlib.sha256(TOOL_PREAMBLE.encode()).hexdigest(),
        "harness_shim_sha256": HARNESS_V2_SHA256, "sandbox_image": sandbox_image,
        "model": {**MODEL_IDENTITY, "endpoint_id": endpoint_id},
        "budget_enforcement": "Two fixed cases, one exclusive session attempt; each uses its own "
            "RunpodQueueTransport(requests_max=6) durable audit, including format retries. "
            "Qwen max_steps=6; shared process watchdog=900 seconds with kill/reap cleanup. "
            "No new session/audit after a failure or uncertain submission.",
        "budget_status": "Frozen requirements for the later live executor; not executed here.",
        "activation": {"approval": "separate live activation approval required", "approved": False,
            "current_gpu_rate_usd_per_hour": None, "actual_spend_usd": None,
            "provider_dollar_cap": False, "required_before_activation":
            "Recheck current GPU rate and pinned worker identity; explain cold-start/idle billing. "
            "Start watchdog with activation; park endpoint and verify no jobs/workers afterward. "
            "Client termination alone does not cancel remote work or guarantee a dollar ceiling."},
        "implementation_commit": implementation_commit,
        "implementation_note": "File hashes describe the exact working tree, including preserved "
                               "uncommitted configuration; the commit alone is not its identity.",
        "implementation_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                                  for p in SOURCE_FILES},
        "artifact_sha256": {"model_inputs.json": digest(model_inputs),
                            "references.json": digest(references), "system-prompt.txt":
                            hashlib.sha256(prompt.encode()).hexdigest(),
                            "claim-review-template.json": digest(review_template)},
        "semantic_support": "not_reviewed", "reward": "not_computed", "model_calls": 0}
    output.mkdir(parents=True)
    write_json(output / "model_inputs.json", model_inputs)
    write_json(output / "references.json", references)
    (output / "system-prompt.txt").write_text(prompt)
    write_json(output / "plan.json", plan)
    write_json(output / "claim-review-template.json", review_template)
    checksum = hashlib.sha256((output / "plan.json").read_bytes()).hexdigest()
    (output / "plan.sha256").write_text(checksum)
    return plan
