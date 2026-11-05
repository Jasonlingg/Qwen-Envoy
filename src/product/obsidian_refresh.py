"""Refresh the reading vault after successful MCP writes, without a watcher."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path


class ObsidianRefresh:
    def __init__(self, vault: Path, stores: list[Path]):
        self.vault = vault.expanduser().absolute()
        self.stores = list(dict.fromkeys(p.expanduser().absolute() for p in stores))
        self._lock = asyncio.Lock()
        self._last_inputs = None
        self._writes = None

    def _refresh(self):
        # Lazy import keeps the optional exporter out of ordinary server startup.
        from scripts.export_research_vault import ROOT, export

        files = [ROOT / "docs/EXPERIMENT_ATLAS.md", ROOT / "README.md"]
        for store in self.stores:
            files.extend(sorted(store.glob("*.json")))
        digest = hashlib.sha256()
        for path in files:
            digest.update(str(path).encode() + b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        current = digest.hexdigest()
        if current == self._last_inputs and (self.vault / "Home.md").is_file():
            return {"status": "unchanged", "vault": str(self.vault)}
        counts = export(self.vault, self.stores, refresh=True)
        self._last_inputs = current
        return {"status": "updated", "vault": str(self.vault), **counts}

    async def __call__(self, ctx, call_next):
        result = await call_next(ctx)
        # SDK request middleware sees the serialized tool result.
        if (ctx.method != "tools/call" or not isinstance(result, dict)
                or result.get("isError") or "content" not in result
                or result.get("resultType", "complete") != "complete"):
            return result
        if self._writes is None:
            self._writes = {tool.name for tool in await self._server.list_tools()
                            if tool.annotations and tool.annotations.read_only_hint is False}
        if (ctx.params or {}).get("name") not in self._writes:
            return result
        async with self._lock:
            try:
                status = await asyncio.to_thread(self._refresh)
            except Exception as exc:
                # The authoritative write has already committed; never report it as failed.
                status = {"status": "failed", "vault": str(self.vault),
                          "detail": str(exc)[:300]}
                result["content"].append({
                    "type": "text", "text": "Record saved, but Obsidian refresh failed: "
                    + status["detail"] + ". Fix the vault and run the manual refresh command."})
        result["_meta"] = {**(result.get("_meta") or {}), "obsidian_refresh": status}
        return result

    def install(self, server):
        self._server = server
        server.middleware.append(self)
