"""Both queue CLI modes configure without relying on uncommitted builder arguments."""
from types import SimpleNamespace

import pytest

from scripts import arxiv_qwen_runpod_mcp as cli
from src.product.arxiv_mcp_config import build_reader


@pytest.mark.parametrize("assigned", [False, True])
def test_queue_cli_preserves_image_and_does_not_start_compute(tmp_path, monkeypatch, assigned):
    transformers = pytest.importorskip("transformers")
    monkeypatch.setenv("RUNPOD_API_KEY", "scripted-test-key")
    monkeypatch.delenv("ENVOY_STARTUP_DIAGNOSTICS", raising=False)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: object())
    monkeypatch.setattr(cli, "RunpodQueueTransport", lambda *a, **k: object())

    def committed_builder(kind, *, qwen_endpoint, qwen_model, qwen_checkpoint,
                          harness_version, max_steps, policy_transport):
        return build_reader(kind, qwen_endpoint=qwen_endpoint, qwen_model=qwen_model,
            qwen_checkpoint=qwen_checkpoint, harness_version=harness_version,
            max_steps=max_steps, policy_transport=policy_transport)

    monkeypatch.setattr(cli, "build_reader", committed_builder)
    readers, serving = [], []
    def server(library, discovery, *, investigator, **kwargs):
        readers.append(investigator)
        return SimpleNamespace(run=lambda **config: serving.append(config))
    monkeypatch.setattr(cli, "create_server", server)
    args = ["arxiv_qwen_runpod_mcp.py", "--endpoint-id", "scripted-endpoint",
            "--library-root", str(tmp_path / "library"),
            "--audit-path", str(tmp_path / "audit.jsonl"), "--sandbox-image", "pinned-image"]
    if assigned:
        args.append("--assigned-paper-reader")
    monkeypatch.setattr("sys.argv", args)
    cli.main()
    assert len(readers) == 1 and readers[0].image == "pinned-image"
    assert readers[0].reader_mode == ("assigned" if assigned else "discovery")
    assert serving == [{"transport": "stdio"}]
