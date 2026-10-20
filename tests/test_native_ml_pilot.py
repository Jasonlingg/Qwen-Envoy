"""Controller boundaries: real local termination and retained native event evidence."""

import asyncio
import io
import json
import os
import subprocess
import sys
from collections import deque
from types import SimpleNamespace

import pytest

from src.eval.native_pilot_client import Deadline, NativeClient, kill_group
from src.eval.native_pilot_config import check_inventory, settings
from src.eval.native_pilot_events import TurnEvents, collect


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups required")
def test_deadline_kills_parent_and_its_running_child():
    code = ("import subprocess,time; child=subprocess.Popen(['sleep','60']); "
            "print(child.pid,flush=True); time.sleep(60)")
    parent = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE,
                              text=True, start_new_session=True)
    timer = None
    try:
        child_pid = int(parent.stdout.readline())
        timer = Deadline(parent.pid, .15)
        parent.wait(timeout=2)
        assert timer.fired.is_set() and parent.returncode < 0
        check = subprocess.run(["ps", "-o", "stat=", "-p", str(child_pid)],
                               capture_output=True, text=True)
        assert not check.stdout.strip() or check.stdout.strip().startswith("Z")
    finally:
        if timer:
            timer.close()
        kill_group(parent.pid)
        parent.wait(timeout=2)


def item(kind, ident, **extra):
    return {"method": "item/started", "params": {
        "item": {"id": ident, "type": kind, **extra}}}


@pytest.mark.asyncio
async def test_mixed_research_monitor_interrupts_and_retains_inflight_overshoot():
    events = TurnEvents()
    events.turn_id = "turn"
    messages = deque([item("webSearch", str(i)) for i in range(10)])
    messages.extend(item("mcpToolCall", str(i), server="arxiv-research")
                    for i in range(10, 22))
    messages.append({"method": "turn/completed", "params": {
        "turn": {"id": "turn", "status": "interrupted"}}})
    sent = []

    async def read(_timeout):
        return messages.popleft()

    async def send(method, params):
        sent.append((method, params))

    client = SimpleNamespace(event=read, send=send)
    result = await collect(client, "thread", events, seconds=5)
    assert len(sent) == 1 and sent[0][0] == "turn/interrupt"
    assert result["research_calls_observed"] == 22
    assert result["research_target_overshoot"] == 2
    assert result["research_interrupt_requested"]
    assert result["turn"]["status"] == "interrupted" and not result["answer"]


def test_duplicate_lifecycles_and_planning_tools_do_not_inflate_research_count():
    events = TurnEvents()
    start = item("webSearch", "one", action={"type": "search", "queries": ["a", "b"]})
    events.consume(start)
    events.consume({**start, "method": "item/completed"})
    events.consume(item("mcpToolCall", "plan", server="ml-experiments"))
    events.consume(item("agentMessage", "comment", phase="commentary", text="working"))
    events.consume(item("agentMessage", "answer", phase="final_answer", text="final answer"))
    result = events.result()
    assert result["research_calls_observed"] == 1
    assert result["answer"] == "final answer"
    assert result["items"][0]["action"]["queries"] == ["a", "b"]


@pytest.mark.asyncio
async def test_protocol_responses_do_not_drop_earlier_tool_notifications():
    reader = asyncio.StreamReader()
    reader.feed_data((json.dumps(item("webSearch", "before-response")) + "\n" +
                      json.dumps({"id": 1, "result": {"ok": True}}) + "\n").encode())

    class Writer:
        def write(self, data):
            pass

        async def drain(self):
            pass

    client = NativeClient(SimpleNamespace(stdin=Writer(), stdout=reader), io.StringIO())
    assert await client.request("turn/start", {}) == {"ok": True}
    assert (await client.event(1))["params"]["item"]["id"] == "before-response"


def test_partial_evidence_errors_and_unknown_phase_are_not_a_final_answer():
    events = TurnEvents()
    events.consume(item("agentMessage", "unknown", text="may be commentary"))
    events.consume({"method": "item/completed", "params": {"item": {
        "id": "failed-paper", "type": "mcpToolCall", "server": "arxiv-research",
        "status": "failed", "result": None, "error": {"message": "reader failed"}}}})
    assert not events.result()["answer"]
    assert events.result()["unclassified_assistant_messages"] == ["may be commentary"]
    assert events.result()["errors"]
    assert events.result()["token_usage"] is None
    with pytest.raises(RuntimeError, match="native RPC failure"):
        events.consume({"id": 1, "error": {"message": "model unavailable"}})


def test_case_state_cannot_be_reused_and_investigate_is_rejected(tmp_path):
    config = settings(tmp_path, tmp_path / "case", "B")
    assert config["shell_environment_policy.inherit"] == "none"
    assert config["permissions.ml-pilot.network.enabled"] is False
    assert "sources" in config["mcp_servers.arxiv-research.args"]
    with pytest.raises(FileExistsError):
        settings(tmp_path, tmp_path / "case", "B")
    with pytest.raises(ValueError):
        check_inventory([{"name": "arxiv-research", "tools": {"investigate": {}}}], "A")


@pytest.mark.asyncio
async def test_idle_read_timeout_is_not_an_early_answer_deadline():
    events, calls = TurnEvents(), 0

    async def read(_timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise asyncio.TimeoutError
        return {"method": "turn/completed", "params": {
            "turn": {"id": "turn", "status": "completed"}}}

    result = await collect(SimpleNamespace(event=read), "thread", events, seconds=1)
    assert calls == 2 and result["turn"]["status"] == "completed"
