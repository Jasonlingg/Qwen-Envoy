"""A Hub name must become a local directory before transformers sees it."""
import sys
from types import SimpleNamespace

from src.product.experiment_tokenizer import load_tokenizer


def test_cached_revision_resolved_without_network_metadata(monkeypatch, tmp_path):
    calls = []
    def snapshot(model, **kwargs):
        assert model == "Qwen/Qwen3-8B"
        assert kwargs == {"revision": "pinned-revision", "local_files_only": True}
        calls.append("cache")
        return str(tmp_path)
    def from_pretrained(path, **kwargs):
        assert path == str(tmp_path.resolve())
        assert kwargs == {"local_files_only": True, "trust_remote_code": False}
        calls.append("load")
        return "tokenizer"
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=from_pretrained)))
    assert load_tokenizer("Qwen/Qwen3-8B", "pinned-revision") == "tokenizer"
    assert calls == ["cache", "load"]
