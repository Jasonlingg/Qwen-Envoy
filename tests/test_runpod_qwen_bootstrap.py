import hashlib
import json

import pytest

from scripts import runpod_qwen_bootstrap as bootstrap


def test_pinned_local_snapshots_and_adapter_selection(tmp_path, monkeypatch):
    base, root = tmp_path / "base", tmp_path / "adapter"
    base.mkdir()
    adapter = root / bootstrap.SUBFOLDER
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": bootstrap.BASE, "r": 4}))
    (adapter / "adapter_model.safetensors").write_bytes(b"verified-fixture")
    monkeypatch.setattr(bootstrap, "ADAPTER_SHA256", hashlib.sha256(b"verified-fixture").hexdigest())
    monkeypatch.setattr(bootstrap.os, "environ", {})
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(base if kwargs["repo_id"] == bootstrap.BASE else root)

    identity = bootstrap.configure(download)
    assert [x["revision"] for x in calls] == [bootstrap.BASE_REVISION, bootstrap.ADAPTER_REVISION]
    assert bootstrap.os.environ["MODEL_NAME"] == str(base)
    assert bootstrap.os.environ["MODEL_REVISION"] == ""
    assert bootstrap.os.environ["OPENAI_SERVED_MODEL_NAME_OVERRIDE"] != "envoy"
    assert json.loads(bootstrap.os.environ["LORA_MODULES"])["path"] == str(adapter)
    assert identity["adapter_rank"] == 4
    (adapter / "adapter_model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="verification failed"):
        bootstrap.configure(download)
