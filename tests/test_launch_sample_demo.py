"""The judge test build must use only a disposable synthetic vault."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from scripts import launch_sample_demo
from src.product.graph import build_graph


def test_launcher_copies_only_sample_vault_and_stays_offline(monkeypatch):
    observed: dict = {}
    sample = launch_sample_demo.SAMPLE_VAULT
    before = {p.relative_to(sample): p.read_bytes() for p in sample.rglob("*.md")}
    monkeypatch.setenv("NEBIUS_API_KEY", "unused-test-key")

    def fake_run(app, *, host, port, log_level):
        assert (host, port, log_level) == ("127.0.0.1", 8866, "warning")
        with TestClient(app) as client:
            assert client.get("/").status_code == 200
            status = client.get("/api/status").json()
            vault = Path(status["vault"])
            observed["vault"] = vault
            assert vault != sample
            assert vault.parent / "state" != vault
            assert (vault.parent / "state").is_dir()
            assert {p.relative_to(vault): p.read_bytes() for p in vault.rglob("*.md")} == before
            graph = build_graph(next((vault.parent / "state").glob("snapshot-*")))
            assert len(graph["nodes"]) == 10
            assert any(edge["type"] == "wikilink" for edge in graph["edges"])
            assert not graph["unresolved"]
            assert status["nemotron_configured"] is False
            assert status["qwen_configured"] is False
            turn = client.post("/api/chat", json={
                "question": "How did my phone-drawer plan change?",
            }).json()
            assert turn["answer"] is None
            assert turn["retrieval"]["mode"] == "lexical_paragraph_baseline"
            assert client.get("/api/status").json()["model_calls"] == 0

    monkeypatch.setattr(launch_sample_demo.uvicorn, "run", fake_run)
    assert launch_sample_demo.main(["--port", "8866"]) == 0
    assert not observed["vault"].exists()
    assert {p.relative_to(sample): p.read_bytes() for p in sample.rglob("*.md")} == before


def test_launcher_removes_temporary_vault_on_sigterm():
    code = (
        "import time\n"
        "from scripts import launch_sample_demo as demo\n"
        "demo.uvicorn.run = lambda *args, **kwargs: time.sleep(30)\n"
        "demo.main([])\n"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        assert process.stdout is not None
        details = json.loads(process.stdout.readline())
        temporary_vault = Path(details["vault"])
        assert temporary_vault.is_dir()
    finally:
        process.terminate()
        process.communicate(timeout=5)
    assert process.returncode == 0
    assert not temporary_vault.exists()
