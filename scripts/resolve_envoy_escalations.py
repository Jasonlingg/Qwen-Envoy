"""Resolve Envoy escalation packets with a pinned, budgeted host model."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import anthropic
import typer
from dotenv import load_dotenv

from src.eval.hashing import sha256 as _sha256
from src.policies.teacher_budget import BudgetedMessages, TeacherBudget
from src.research.escalation import (
    HOST_RESOLVER_SYSTEM_PROMPT,
    HostResolutionError,
    build_evidence_packet,
    host_message,
    parse_host_json,
    validate_host_resolution,
)

MODEL = "claude-sonnet-5"


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def _load_documents(path: Path) -> dict[str, str]:
    documents = {}
    for item in sorted(path.glob("*.json")):
        value = json.loads(item.read_text())
        documents[value["doc_id"]] = value["text"]
    if not documents:
        raise typer.BadParameter(f"No corpus documents found in {path}")
    return documents


def _response_text(response: Any) -> str:
    blocks = [block.text for block in response.content if hasattr(block, "text")]
    if not blocks:
        raise HostResolutionError("Host response had no text block")
    return blocks[-1]


def main(
    input_path: Path = typer.Option(..., "--input"),
    corpus_path: Path = typer.Option(..., "--corpus"),
    output: Path = typer.Option(..., "--output"),
    model: str = typer.Option(MODEL),
    budget_usd: float = typer.Option(1.0, min=0.01),
    max_tokens: int = typer.Option(1024, min=128, max=4096),
) -> None:
    """Call the host only for escalated rows and write one combined transcript."""
    if model != MODEL:
        raise typer.BadParameter(f"Host resolution is pinned to {MODEL}; got {model!r}")
    for key in ("ANTHROPIC_API_KEY",):
        if os.environ.get(key) == "":
            del os.environ[key]
    load_dotenv(override=True)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise typer.BadParameter("ANTHROPIC_API_KEY is not configured")
    if output.exists() or output.with_suffix(".resolver.json").exists():
        raise typer.BadParameter(f"Output already exists: {output}")

    rows = json.loads(input_path.read_text())
    if not isinstance(rows, list) or not rows:
        raise typer.BadParameter("Input must be a non-empty transcript list")
    documents = _load_documents(corpus_path)
    combined = []
    for row in rows:
        item = dict(row)
        item["qwen_status"] = row.get("status")
        item["qwen_escalation"] = row.get("escalation")
        item["final_source"] = "qwen" if row.get("status") == "completed" else None
        item["host_resolution"] = None
        item["policy"] = "qwen_host_cascade"
        item["run_label"] = "checkpoint150_two_recovery_sonnet5_cascade_dev25"
        combined.append(item)

    resolver_path = output.with_suffix(".resolver.json")
    ledger_path = output.with_suffix(".budget.json")
    metadata = {
        "schema_version": "envoy-host-resolution-v1",
        "status": "running",
        "input": str(input_path),
        "input_sha256": _sha256(input_path),
        "corpus": str(corpus_path),
        "model": model,
        "required_response_model": MODEL,
        "budget_usd": budget_usd,
        "max_tokens": max_tokens,
        "system_prompt": HOST_RESOLVER_SYSTEM_PROMPT,
        "system_prompt_sha256": hashlib.sha256(
            HOST_RESOLVER_SYSTEM_PROMPT.encode()
        ).hexdigest(),
        "escalated_question_ids": [
            row["question_id"] for row in rows if row.get("status") == "escalated"
        ],
        "resolved_question_ids": [],
        "errors": {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, combined)
    _write_json(resolver_path, metadata)

    budget = TeacherBudget(ledger_path, budget_usd)
    client = anthropic.Anthropic().with_options(max_retries=0, timeout=120)
    try:
        for index, source_row in enumerate(rows):
            if source_row.get("status") != "escalated":
                continue
            question_id = source_row["question_id"]
            packet = build_evidence_packet(source_row)
            messages = BudgetedMessages(client.messages, budget, question_id)
            try:
                response = messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=HOST_RESOLVER_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": host_message(packet)}],
                )
                returned_model = getattr(response, "model", None)
                raw_response = _response_text(response)
                usage = {
                    key: getattr(response.usage, key, None)
                    for key in ("input_tokens", "output_tokens")
                }
                combined[index]["host_attempt"] = {
                    "model": returned_model,
                    "usage": usage,
                    "raw_response": raw_response,
                    "packet_sha256": hashlib.sha256(
                        host_message(packet).encode()
                    ).hexdigest(),
                }
                if returned_model != MODEL:
                    raise HostResolutionError(
                        f"Requested {MODEL}, provider returned {returned_model!r}"
                    )
                resolution = validate_host_resolution(
                    parse_host_json(raw_response),
                    allowed_doc_ids=packet["allowed_document_ids"],
                    documents=documents,
                )
                combined[index].update(
                    status="completed",
                    predicted_answer=resolution["answer"],
                    predicted_citations=resolution["citations"],
                    predicted_evidence=[
                        {key: span[key] for key in ("doc_id", "start", "end")}
                        for span in resolution["evidence"]
                    ],
                    final_source="host",
                    host_resolution={
                        **resolution,
                        "model": returned_model,
                        "usage": usage,
                        "packet_sha256": hashlib.sha256(
                            host_message(packet).encode()
                        ).hexdigest(),
                    },
                )
                metadata["resolved_question_ids"].append(question_id)
            except Exception as exc:
                combined[index]["status"] = "resolver_error"
                combined[index]["error"] = f"{type(exc).__name__}: {exc}"
                metadata["errors"][question_id] = combined[index]["error"]
            _write_json(output, combined)
            _write_json(resolver_path, metadata)
    finally:
        metadata["budget"] = budget.summary()
        metadata["status"] = "complete" if not metadata["errors"] else "completed_with_errors"
        _write_json(output, combined)
        _write_json(resolver_path, metadata)
        budget.close()
        client.close()

    if metadata["errors"]:
        raise typer.Exit(1)
    typer.echo(json.dumps({
        "output": str(output),
        "resolved": len(metadata["resolved_question_ids"]),
        "host_call_rate": len(metadata["resolved_question_ids"]) / len(rows),
        "budget": metadata["budget"],
    }, indent=2))


if __name__ == "__main__":
    typer.run(main)
