"""A logged, single-connection native app-server client; no model API fallback."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import threading
import time
from collections import deque


def kill_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


class Deadline:
    """Wall timer independent of the event reader; kills this process group."""

    def __init__(self, pid: int, seconds: float):
        self.fired = threading.Event()
        self.started = time.monotonic()

        def stop():
            self.fired.set()
            kill_group(pid)

        self.timer = threading.Timer(seconds, stop)
        self.timer.daemon = True
        self.timer.start()

    def close(self):
        self.timer.cancel()
        self.timer.join()


class NativeClient:
    def __init__(self, process, trace):
        self.process, self.trace = process, trace
        self.next_id = 0
        self.pending = deque()
        self.trace_bytes = 0

    def log(self, direction: str, message: dict):
        line = json.dumps({"monotonic": time.monotonic(), "direction": direction,
                           "message": message}, ensure_ascii=False) + "\n"
        self.trace_bytes += len(line.encode())
        if self.trace_bytes > 64 * 1024 * 1024:
            raise RuntimeError("native trace exceeded 64 MiB")
        self.trace.write(line)
        self.trace.flush()

    async def send(self, method: str, params: dict, *, notification=False):
        message = {"method": method, "params": params}
        if not notification:
            self.next_id += 1
            message["id"] = self.next_id
        self.log("client", message)
        self.process.stdin.write((json.dumps(message) + "\n").encode())
        await self.process.stdin.drain()
        return message.get("id")

    async def read(self, timeout=60):
        raw = await asyncio.wait_for(self.process.stdout.readline(), timeout)
        if not raw:
            raise EOFError("native app-server closed stdout")
        try:
            message = json.loads(raw)
        except (ValueError, UnicodeError):
            self.log("server", {"malformed_stdout": raw.decode(errors="replace")[:10000]})
            raise
        self.log("server", message)
        if "id" in message and "method" in message:
            # Never approve unplanned actions or fabricate a user's response.
            raise RuntimeError("unexpected native interaction: " + message["method"])
        return message

    async def request(self, method: str, params: dict):
        ident = await self.send(method, params)
        while True:
            message = await self.read()
            if message.get("id") == ident:
                if "error" in message:
                    raise RuntimeError(json.dumps(message["error"]))
                return message["result"]
            self.pending.append(message)

    async def event(self, timeout):
        return self.pending.popleft() if self.pending else await self.read(timeout)

    async def close(self):
        # Always kill our own group, even if its leader has already exited.
        kill_group(self.process.pid)
        await asyncio.wait_for(self.process.wait(), 5)
