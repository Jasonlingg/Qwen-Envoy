"""Resource-limited Docker transport for new, model-authored code runs.

The historical DockerREPL source is pinned by saved experiment manifests, so
its original container command remains unchanged for replay verification.
"""

from __future__ import annotations

import os
import subprocess

from loguru import logger

from src.env.repl import DockerREPL
from src.env.tools import TOOL_PREAMBLE


class HardenedDockerREPL(DockerREPL):
    """Run the existing cumulative-script protocol in a limited container."""

    def start_session(self) -> None:
        command = [
            "docker", "create",
            "--memory", self.memory_limit,
            "--memory-swap", self.memory_limit,
            "--network", "none",
            "--user", "1000:1000",
            "--pids-limit", "64",
            "--cap-drop", "ALL",
            "--read-only",
            "--tmpfs", "/tmp:rw,nosuid,nodev,size=16m,mode=1777",
            "--cpus", "1",
            "--security-opt", "no-new-privileges",
            "--mount",
            f"type=bind,src={self.corpus_path},"
            f"dst={self._container_corpus_dir},readonly",
            "-e", f"CORPUS_DIR={self._container_corpus_dir}",
        ]
        for name in ("ENVOY_SEARCH_WITHIN_TOP_K", "ENVOY_SEARCH_WITHIN_MODE"):
            value = os.environ.get(name)
            if value:
                command.extend(["-e", f"{name}={value}"])
        command.extend(["-i", self.image, "python3", "-i"])
        created = subprocess.run(command, capture_output=True, text=True, timeout=30)
        if created.returncode:
            raise RuntimeError(f"Docker create failed: {created.stderr}")
        self._container_id = created.stdout.strip()
        try:
            started = subprocess.run(
                ["docker", "start", self._container_id],
                capture_output=True, text=True, timeout=10,
            )
            if started.returncode:
                raise RuntimeError(f"Docker start failed: {started.stderr}")
            self._cumulative_script = TOOL_PREAMBLE + "\n" + self.extra_preamble
            self._step = 0
            self._last_execution_failed = False
            output = self._run_script(self._cumulative_script)
            if self._last_execution_failed:
                raise RuntimeError(f"Docker preamble failed: {output[:500]}")
            logger.info(f"Hardened Docker container started: {self._container_id[:12]}")
        except BaseException:
            self.kill_session()
            raise
