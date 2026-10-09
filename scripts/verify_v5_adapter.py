"""Verify the adapter files before serving the pinned Qwen3-8B v5 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def verify_adapter(adapter_dir: Path, base_model: str, expected_sha256: str) -> dict:
    config_path = adapter_dir / "adapter_config.json"
    weights_path = adapter_dir / "adapter_model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise ValueError("adapter config and weights are both required")
    config = json.loads(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError("adapter config must be a JSON object")
    trained_on = config.get("base_model_name_or_path")
    if trained_on != base_model:
        raise ValueError(
            f"adapter was trained on {trained_on!r}, not the pinned {base_model!r}"
        )
    digest = hashlib.sha256()
    with weights_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "adapter weights SHA-256 does not match the pinned v5 checkpoint "
            f"(expected {expected_sha256}, got {actual_sha256})"
        )
    return {
        "adapter_dir": str(adapter_dir.resolve()),
        "base_model": base_model,
        "adapter_sha256": actual_sha256,
        "rank": config.get("r"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(verify_adapter(
        args.adapter_dir, args.base_model, args.expected_sha256,
    ), sort_keys=True))


if __name__ == "__main__":
    main()
