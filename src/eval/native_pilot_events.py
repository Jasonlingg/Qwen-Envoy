"""Preserve native outcomes and monitor research calls without claiming a cap."""

from __future__ import annotations

import asyncio
import json
import time


class TurnEvents:
    def __init__(self, target=20):
        self.target = target
        self.research_ids = set()
        self.items = {}
        self.errors = []
        self.usage = None
        self.turn = None
        self.turn_id = None
        self.interrupt_requested = False

    def consume(self, message: dict):
        if "id" in message and "error" in message:
            raise RuntimeError("native RPC failure: " + json.dumps(message["error"]))
        method, params = message.get("method"), message.get("params", {})
        if method == "thread/tokenUsage/updated":
            self.usage = params["tokenUsage"]
        if method in {"turn/started", "turn/completed"}:
            self.turn_id = params["turn"]["id"]
            if method == "turn/completed":
                self.turn = params["turn"]
        if method in {"item/started", "item/completed"}:
            item = params["item"]
            self.items[item["id"]] = item
            if (item["type"] == "webSearch" or
                (item["type"] == "mcpToolCall" and item.get("server") == "arxiv-research")):
                self.research_ids.add(item["id"])
            if method == "item/completed" and (item.get("error") or
                    item.get("status") in {"failed", "declined"} or
                    (item.get("result") or {}).get("isError", False)):
                self.errors.append(item)
        if method in {"error", "warning", "configWarning"}:
            self.errors.append(message)

    @property
    def target_reached(self):
        return len(self.research_ids) >= self.target

    def result(self):
        finals = [item["text"] for item in self.items.values()
                  if item["type"] == "agentMessage" and item.get("phase") == "final_answer"]
        candidates = [item["text"] for item in self.items.values()
                      if item["type"] == "agentMessage"]
        return {"turn": self.turn, "research_calls_observed": len(self.research_ids),
                "research_target_overshoot": max(0, len(self.research_ids) - self.target),
                "items": list(self.items.values()), "errors": self.errors,
                "token_usage": self.usage, "answer": "\n\n".join(finals),
                "research_interrupt_requested": self.interrupt_requested,
                "unclassified_assistant_messages": candidates if not finals else [],
                "research_count_scope": "Unique native webSearch and source-MCP item IDs; "
                "compound query counts/actions remain in items; opaque hosted activity is unknown."}


async def collect(client, thread_id, events, *, seconds=600):
    """Independent caller watchdog terminates the process group at the deadline."""
    interrupted = None
    started = time.monotonic()
    while events.turn is None:
        remaining = seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("answer deadline reached")
        try:
            message = await client.event(min(remaining, 60))
        except asyncio.TimeoutError:
            continue
        events.consume(message)
        if (events.target_reached and interrupted is None and events.turn_id
                and events.turn is None):
            interrupted = time.monotonic()
            events.interrupt_requested = True
            await client.send("turn/interrupt", {"threadId": thread_id,
                                               "turnId": events.turn_id})
    return {**events.result(), "research_interrupt_requested": interrupted is not None}
