"""New reader sandbox limits without rewriting frozen historical REPL source."""

import subprocess
from pathlib import Path

import pytest

from src.env.hardened_repl import HardenedDockerREPL
from src.env.tools_v2 import HARNESS_V2_PREAMBLE
from src.product.arxiv_mcp_config import build_reader
from src.product.hardened_reader import HardenedQwenInvestigator


def test_new_reader_uses_hardened_docker_and_v2_preamble(tmp_path):
    reader = build_reader("qwen", qwen_endpoint="http://127.0.0.1:8000/v1",
                          qwen_model="fake", qwen_checkpoint="pinned")
    assert isinstance(reader, HardenedQwenInvestigator)
    repl = reader._repl(Path(tmp_path))
    assert isinstance(repl, HardenedDockerREPL)
    assert repl.extra_preamble == HARNESS_V2_PREAMBLE


def test_hardened_docker_has_limits_and_cleans_up_on_start_failure(monkeypatch, tmp_path):
    commands = []

    def fake_run(args, **kwargs):
        commands.append(args)
        if args[1] == "create":
            return subprocess.CompletedProcess(args, 0, "container-id\n", "")
        if args[1] == "start":
            return subprocess.CompletedProcess(args, 1, "", "start failed")
        assert args[1] == "rm"
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    repl = HardenedDockerREPL(corpus_path=str(tmp_path))
    with pytest.raises(RuntimeError, match="Docker start failed"):
        repl.start_session()
    create = commands[0]
    assert "--read-only" in create
    for flag, expected in (
        ("--network", "none"), ("--user", "1000:1000"),
        ("--memory-swap", "512m"), ("--pids-limit", "64"),
        ("--cap-drop", "ALL"), ("--cpus", "1"),
        ("--security-opt", "no-new-privileges"),
    ):
        assert create[create.index(flag) + 1] == expected
    assert create[create.index("--tmpfs") + 1] == "/tmp:rw,nosuid,nodev,size=16m,mode=1777"
    assert create[create.index("--mount") + 1].endswith(",readonly")
    assert [command[1] for command in commands] == ["create", "start", "rm"]
    assert repl._container_id is None
