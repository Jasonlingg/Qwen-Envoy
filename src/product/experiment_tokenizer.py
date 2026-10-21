"""Resolve cached files before loading; avoid tokenizer remote metadata heuristics."""
from pathlib import Path


def load_tokenizer(model, revision):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    local = Path(model).expanduser()
    if not local.is_dir():
        # Resolve the revision locally, then pass a directory rather than a Hub ID.
        # Some transformers versions still request model_info for Hub tokenizer IDs
        # despite local_files_only=True in from_pretrained.
        local = Path(snapshot_download(model, revision=revision, local_files_only=True))
    return AutoTokenizer.from_pretrained(
        str(local.resolve()), local_files_only=True, trust_remote_code=False)
