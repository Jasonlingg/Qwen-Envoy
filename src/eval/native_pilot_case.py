"""One native answer attempt with setup checks and retained partial outcomes."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from src.eval.native_pilot_client import Deadline, NativeClient
from src.eval.native_pilot_config import (
    DEVELOPER,
    EFFORT,
    MODEL,
    check_inventory,
    command,
    settings,
)
from src.eval.native_pilot_config import prompt as make_prompt
from src.eval.native_pilot_events import TurnEvents, collect


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


async def setup(client, work, arm):
    await client.request("initialize", {"clientInfo": {"name": "ml_workflow_pilot", "version": "1"},
                                       "capabilities": {"experimentalApi": True}})
    await client.send("initialized", {}, notification=True)
    thread = await client.request("thread/start", {
        "cwd": str(work), "ephemeral": True, "permissions": "ml-pilot",
        "approvalPolicy": "never", "allowProviderModelFallback": False,
        "model": MODEL, "developerInstructions": DEVELOPER})
    if (thread["model"] != MODEL or thread["reasoningEffort"] != EFFORT
            or thread["modelProvider"] != "openai" or thread["instructionSources"]):
        raise RuntimeError("native host settings/instructions did not match")
    thread_id = thread["thread"]["id"]
    inventory, cursor = [], None
    while True:
        page = await client.request("mcpServerStatus/list", {"threadId": thread_id,
            "detail": "full", "limit": 100, "cursor": cursor})
        inventory.extend(page["data"])
        cursor = page.get("nextCursor")
        if not cursor:
            break
    check_inventory(inventory, arm)
    return thread, inventory


async def boundary_checks(client, work, output):
    (work / "public-canary.txt").write_text("public canary\n")
    outside = output / "private-canary.txt"
    outside.write_text("outside-workspace canary\n")
    results = {}
    for label, path in (("inside", work / "public-canary.txt"), ("outside", outside)):
        results[label] = await client.request("command/exec", {
            "command": ["/bin/cat", str(path)], "cwd": str(work),
            "permissionProfile": "ml-pilot", "timeoutMs": 3000, "outputBytesCap": 1024})
    if results["inside"]["exitCode"] != 0 or results["outside"]["exitCode"] == 0:
        raise RuntimeError("native filesystem canary boundary failed")
    return results


async def run_case(repo, output, common, question, arm, *, seconds=600):
    """Never retries a prompt; callers may not reuse an existing output path."""
    output.mkdir(parents=True, exist_ok=False)
    work = Path(tempfile.mkdtemp(prefix="ml-pilot-work-")).resolve()
    config = settings(repo, output / "servers", arm)
    text = make_prompt(common, question)
    (work / "question.txt").write_text(text)
    (output / "prompt.txt").write_text(text)
    write_json(output / "settings.json", config)
    stderr, trace = (output / "stderr.log").open("w"), (output / "events.jsonl").open("w")
    process = await asyncio.create_subprocess_exec(*command(config), cwd=work,
        env={**os.environ, "PWD": str(work)}, stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE, stderr=stderr, start_new_session=True,
        limit=16 * 1024 * 1024)
    client, events = NativeClient(process, trace), TurnEvents()
    deadline, started = None, None
    result = {"question_id": question["id"], "arm": arm, "status": "setup_error",
              "model_turns_attempted": 0, "workdir": str(work), "actual_spend_usd": None,
              "host_model_requests": None, "human_review": "not_performed",
              "prompt_sha256": hashlib.sha256(text.encode()).hexdigest()}
    try:
        thread, inventory = await setup(client, work, arm)
        write_json(output / "thread.json", thread)
        write_json(output / "mcp-inventory.json", inventory)
        checks = await boundary_checks(client, work, output)
        write_json(output / "boundary-checks.json", checks)
        started = time.monotonic()
        result.update(status="in_progress", model_turns_attempted=1,
                      started_at_utc=datetime.now(timezone.utc).isoformat())
        deadline = Deadline(process.pid, seconds)
        turn = await client.request("turn/start", {"threadId": thread["thread"]["id"],
            "input": [{"type": "text", "text": text}], "effort": EFFORT,
            "permissions": "ml-pilot", "approvalPolicy": "never"})
        events.turn_id = turn["turn"]["id"]
        collected = await collect(client, thread["thread"]["id"], events,
                                  seconds=seconds - (time.monotonic() - started))
        result.update(collected, status=collected["turn"]["status"])
    except Exception as exc:
        result.update(error=f"{type(exc).__name__}: {exc}"[:2000],
                      status="generation_error" if started is not None else "setup_error")
    finally:
        if deadline:
            deadline.close()
            result["watchdog_fired"] = deadline.fired.is_set()
            if deadline.fired.is_set():
                result["status"] = "timeout"
        if started is not None:
            result["elapsed_seconds"] = time.monotonic() - started
        # Preserve partial evidence even on EOF, errors, or process termination.
        for key, value in events.result().items():
            result.setdefault(key, value)
        result["answer_present"] = bool(result["answer"].strip())
        result["answer_words_including_sources"] = len(result["answer"].split())
        result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        try:
            await client.close()
        finally:
            stderr.close()
            trace.close()
            write_json(output / "result.json", result)
            (output / "answer.md").write_text(result["answer"])
    return result
