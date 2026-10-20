import asyncio

import pytest

from scripts.run_qwen_mcp_live import connected_inventory


class Client:
    def __init__(self, statuses):
        self.statuses = iter(statuses)
        self.calls = []

    async def request(self, method, params):
        self.calls.append(method)
        state = next(self.statuses)
        return {"data": [
            {"name": "arxiv-research", "runtimeStatus": state, "tools": {"investigate": {}}},
            {"name": "ml-experiments", "runtimeStatus": "connected", "tools": {"save_experiment_brief": {}}}]}


def test_cached_tool_schema_is_not_readiness(monkeypatch):
    async def no_wait(_): pass
    monkeypatch.setattr("scripts.run_qwen_mcp_live.asyncio.sleep", no_wait)
    client = Client(["starting", "connected"])
    result = asyncio.run(connected_inventory(client, "thread"))
    assert result["data"][0]["runtimeStatus"] == "connected"
    assert client.calls == ["mcpServerStatus/list", "mcpServerStatus/list"]


def test_startup_failure_stops_before_answer_turn():
    client = Client(["failed"])
    with pytest.raises(RuntimeError, match="failed during startup"):
        asyncio.run(connected_inventory(client, "thread"))
    assert client.calls == ["mcpServerStatus/list"]
