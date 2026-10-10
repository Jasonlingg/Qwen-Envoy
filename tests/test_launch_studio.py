"""The launcher includes completed release artifacts without running models."""

from __future__ import annotations

import json

import pytest

from scripts import launch_studio


@pytest.mark.parametrize("status", ["running", "complete", "incomplete"])
def test_launcher_loads_completed_supplement_and_deduplicates_paths(tmp_path, monkeypatch, status):
    study = tmp_path / "release/qasper-agent-study"
    nebius = study / "nebius-run"
    nebius.mkdir(parents=True)
    (nebius / "manifest.json").write_text(json.dumps({"status": status}))
    score = study / "qwen-official-score.json"
    score.write_text("{}")
    monkeypatch.setattr(launch_studio, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    argv = [
        "launch_studio",
        "--qasper-score",
        "release/qasper-agent-study/qwen-official-score.json",
    ]
    if status in {"complete", "incomplete"}:
        argv += ["--supplementary-run", "release/qasper-agent-study/nebius-run"]
    monkeypatch.setattr("sys.argv", argv)
    calls = []
    app = object()

    def create_app(**kwargs):
        calls.append(kwargs)
        return app

    monkeypatch.setattr("benchmarks.envoybench.demo.create_app", create_app)
    server = []
    monkeypatch.setattr("uvicorn.run", lambda value, **kwargs: server.append((value, kwargs)))
    assert launch_studio.main() == 0
    assert calls[0]["supplementary_runs"] == (
        [nebius] if status in {"complete", "incomplete"} else []
    )
    assert calls[0]["qasper_scores"] == [score]
    assert calls[0]["paper_models_path"] is None
    assert server == [(app, {"host": "127.0.0.1", "port": 8765})]
