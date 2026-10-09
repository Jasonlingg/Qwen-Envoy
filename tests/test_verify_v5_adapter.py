"""The serving gate must reject local files that are not the pinned adapter."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from scripts.verify_v5_adapter import verify_adapter

ROOT = Path(__file__).resolve().parents[1]


def test_adapter_identity_requires_matching_base_and_weight_bytes(tmp_path):
    config = tmp_path / "adapter_config.json"
    weights = tmp_path / "adapter_model.safetensors"
    config.write_text(json.dumps({"base_model_name_or_path": "Qwen/Qwen3-8B", "r": 4}))
    weights.write_bytes(b"pinned test adapter")
    expected = hashlib.sha256(weights.read_bytes()).hexdigest()

    identity = verify_adapter(tmp_path, "Qwen/Qwen3-8B", expected)
    assert identity["adapter_sha256"] == expected
    assert identity["rank"] == 4

    weights.write_bytes(b"another adapter")
    with pytest.raises(ValueError, match="SHA-256 does not match"):
        verify_adapter(tmp_path, "Qwen/Qwen3-8B", expected)

    weights.write_bytes(b"pinned test adapter")
    config.write_text(json.dumps({"base_model_name_or_path": "Qwen/Qwen2.5-7B"}))
    with pytest.raises(ValueError, match="not the pinned"):
        verify_adapter(tmp_path, "Qwen/Qwen3-8B", expected)


def test_adapter_identity_requires_both_files(tmp_path):
    (tmp_path / "adapter_config.json").write_text("{}")
    with pytest.raises(ValueError, match="both required"):
        verify_adapter(tmp_path, "Qwen/Qwen3-8B", "0" * 64)


def test_serving_script_rejects_wrong_local_weights_before_starting_vllm(tmp_path):
    (tmp_path / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": "Qwen/Qwen3-8B", "r": 4,
    }))
    (tmp_path / "adapter_model.safetensors").write_bytes(b"wrong checkpoint")
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/serve_vllm.sh")],
        env={**os.environ, "ADAPTER_DIR": str(tmp_path)},
        capture_output=True, text=True, timeout=10, check=False,
    )
    assert result.returncode != 0
    assert "SHA-256 does not match" in result.stderr
    assert "Serving Qwen/" not in result.stderr
