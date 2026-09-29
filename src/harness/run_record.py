"""Immutable run inputs and append-only events outside the Obsidian vault."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RUN_RECORD_VERSION = "research-harness-run-v1"
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}\Z")
_ARTIFACT = re.compile(r"[a-z][a-z0-9_-]{0,79}\.json\Z")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class RunRecord:
    """A local record that never writes to the selected vault.

    The caller supplies a complete, reproducible input manifest. No credentials
    or unbounded paper text should be included in it; full trajectories belong
    in named artifacts under the same run directory.
    """

    def __init__(self, root: Path, vault: Path, run_id: str, inputs: dict[str, Any]):
        if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
            raise ValueError("run_id must contain only letters, digits, underscores or hyphens")
        if not isinstance(inputs, dict) or not inputs:
            raise ValueError("run inputs must be a nonempty mapping")
        vault = vault.expanduser().resolve(strict=True)
        if not vault.is_dir():
            raise ValueError("vault must be an existing directory")
        root = root.expanduser().resolve()
        if root.is_relative_to(vault):
            raise ValueError("run records must be outside the vault")
        manifest_text = (
            _json(
                {
                    "schema_version": RUN_RECORD_VERSION,
                    "run_id": run_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "inputs": inputs,
                }
            )
            + "\n"
        )
        root.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.directory = root / run_id
        self.directory.mkdir(mode=0o700, exist_ok=False)
        self.manifest = self.directory / "manifest.json"
        self.events = self.directory / "events.jsonl"
        self.manifest.write_text(manifest_text, encoding="utf-8")
        self.append_event("created", {})

    def append_event(self, state: str, details: dict[str, Any]) -> None:
        if (
            not isinstance(state, str)
            or not re.fullmatch(r"[a-z][a-z_]{0,49}", state)
            or not isinstance(details, dict)
        ):
            raise ValueError("event requires a simple state and object details")
        event = {
            "at": datetime.now(timezone.utc).isoformat(),
            "state": state,
            "details": details,
        }
        line = _json(event) + "\n"
        with self.events.open("a", encoding="utf-8") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())

    def save_artifact(self, name: str, value: Any) -> Path:
        """Write one immutable JSON artifact, such as a verified bundle or draft."""
        if not isinstance(name, str) or not _ARTIFACT.fullmatch(name):
            raise ValueError("artifact name must be a safe .json basename")
        target = self.directory / name
        data = _json(value) + "\n"
        with target.open("x", encoding="utf-8") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        return target
