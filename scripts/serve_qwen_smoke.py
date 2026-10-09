"""Serve pinned Qwen3-8B + v5 LoRA for a local, single-client product smoke.

This is a diagnostic bridge, not a deployment server. It binds only to the pod's
loopback interface and processes one request at a time. Reach it through an SSH
tunnel; never expose it directly to the Internet. Model-authored code still runs
in the separate Docker sandbox on the client machine.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.verify_v5_adapter import verify_adapter  # noqa: E402

BASE_ID = "Qwen/Qwen3-8B"
BASE_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
ADAPTER_ID = "jasonlingg/qwen-envoy-qwen3-8b-qasper-sft-v5"
ADAPTER_REVISION = "27b912a863ff914ad45baa03624d8911dc1e17fb"
ADAPTER_SUBFOLDER = "artifacts/full/checkpoint-50"
ADAPTER_SHA256 = "7afba233aecc9fb9f1f23638514e309d139caf5c55d81af4138a05edd89d62a3"
SERVED_ID = "envoy-v5"
MAX_REQUEST_BYTES = 512 * 1024
MAX_PROMPT_TOKENS = 16_384
MAX_NEW_TOKENS = 1_024


def resolve_weights(base_path: Path | None, adapter_path: Path, cache_dir: Path | None) -> Path:
    """Require the actual cached HF snapshot for the pinned base revision."""
    from huggingface_hub import snapshot_download

    cached = Path(snapshot_download(
        repo_id=BASE_ID, revision=BASE_REVISION, local_files_only=True,
        cache_dir=str(cache_dir) if cache_dir else None,
    )).resolve(strict=True)
    if base_path is not None and not base_path.expanduser().resolve(strict=True).samefile(cached):
        raise ValueError(
            "--base-path is not the cached snapshot for the pinned base revision; "
            "set --cache-dir to its Hugging Face hub cache or remove --base-path"
        )
    if cached.name != BASE_REVISION or not (cached / "config.json").is_file():
        raise ValueError("cached base is not the pinned Qwen3-8B revision")
    verify_adapter(adapter_path.expanduser().resolve(strict=True), BASE_ID, ADAPTER_SHA256)
    return cached


def validate_request(payload: Any) -> tuple[list[dict[str, str]], int]:
    if not isinstance(payload, dict):
        raise ValueError("request must be a JSON object")
    if payload.get("model") != SERVED_ID:
        raise ValueError(f"model must be {SERVED_ID!r}")
    if payload.get("stream", False) is not False:
        raise ValueError("streaming is unsupported by the smoke server")
    if payload.get("n", 1) != 1 or "tools" in payload or "tool_choice" in payload:
        raise ValueError("only one plain-text completion is supported")
    if payload.get("chat_template_kwargs") != {"enable_thinking": False}:
        raise ValueError("chat_template_kwargs must disable Qwen thinking mode")
    if payload.get("temperature") != 0 or payload.get("top_p") != 1 or payload.get("seed") != 42:
        raise ValueError("smoke requires temperature=0, top_p=1, seed=42")
    max_tokens = payload.get("max_tokens")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
        raise ValueError("max_tokens must be an integer")
    if not 1 <= max_tokens <= MAX_NEW_TOKENS:
        raise ValueError(f"max_tokens must be between 1 and {MAX_NEW_TOKENS}")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not 1 <= len(messages) <= 64:
        raise ValueError("messages must be a non-empty list of at most 64 turns")
    if any(
        not isinstance(message, dict)
        or message.get("role") not in {"system", "user", "assistant"}
        or not isinstance(message.get("content"), str)
        for message in messages
    ):
        raise ValueError("messages must contain text-only system/user/assistant turns")
    return [{"role": item["role"], "content": item["content"]} for item in messages], max_tokens


class QwenEngine:
    """One GPU copy of the verified base and adapter, using greedy generation."""

    def __init__(self, base_path: Path, adapter_path: Path) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("this smoke server requires a CUDA GPU")
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(base_path), local_files_only=True, trust_remote_code=False,
        )
        base = AutoModelForCausalLM.from_pretrained(
            str(base_path), local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.bfloat16, device_map={"": "cuda:0"},
        )
        self.model = PeftModel.from_pretrained(
            base, str(adapter_path), is_trainable=False, local_files_only=True,
        )
        self.model.eval()

    def generate(
        self, messages: list[dict[str, str]], max_tokens: int,
    ) -> tuple[str, int, int, str]:
        self.torch.manual_seed(42)
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = self.tokenizer(rendered, return_tensors="pt")
        prompt_tokens = inputs["input_ids"].shape[1]
        if prompt_tokens > MAX_PROMPT_TOKENS:
            raise ValueError(f"prompt exceeds smoke limit of {MAX_PROMPT_TOKENS} tokens")
        inputs = inputs.to("cuda:0")
        with self.torch.inference_mode():
            tokens = self.model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )[0][prompt_tokens:]
        completion_tokens = len(tokens)
        finished = bool(completion_tokens and tokens[-1].item() == self.tokenizer.eos_token_id)
        content = self.tokenizer.decode(tokens, skip_special_tokens=True)
        return content, prompt_tokens, completion_tokens, "stop" if finished else "length"


def make_handler(engine: QwenEngine) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _json(self, status: int, body: dict[str, Any]) -> None:
            encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/v1/models":
                self._json(404, {"error": {"message": "unknown path"}})
                return
            self._json(200, {"object": "list", "data": [{
                "id": SERVED_ID, "object": "model", "created": 0,
                "owned_by": "local-smoke", "base_revision": BASE_REVISION,
                "adapter_revision": ADAPTER_REVISION, "adapter_sha256": ADAPTER_SHA256,
            }]})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/chat/completions":
                self._json(404, {"error": {"message": "unknown path"}})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_REQUEST_BYTES:
                    raise ValueError("invalid or oversized request body")
                payload = json.loads(self.rfile.read(length))
                messages, max_tokens = validate_request(payload)
                content, prompt_tokens, completion_tokens, reason = engine.generate(
                    messages, max_tokens,
                )
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self._json(400, {"error": {"message": str(exc)}})
                return
            except RuntimeError as exc:
                self._json(500, {"error": {"message": f"inference failed: {type(exc).__name__}"}})
                return
            self._json(200, {
                "id": f"chatcmpl-smoke-{uuid4().hex}", "object": "chat.completion",
                "created": int(time.time()), "model": SERVED_ID,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                             "finish_reason": reason}],
                "usage": {"prompt_tokens": prompt_tokens,
                          "completion_tokens": completion_tokens,
                          "total_tokens": prompt_tokens + completion_tokens},
            })

        def log_message(self, format: str, *args: Any) -> None:
            # The standard handler logs the path but not prompt bodies. Keep
            # this test bridge quiet; trajectory logging belongs to the client.
            pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-path", type=Path,
                        help="optional path that must match the pinned cached HF snapshot")
    parser.add_argument("--cache-dir", type=Path,
                        help="Hugging Face hub cache containing the pinned base snapshot")
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    base_path = resolve_weights(args.base_path, args.adapter_path, args.cache_dir)
    engine = QwenEngine(base_path, args.adapter_path)
    server = HTTPServer(("127.0.0.1", args.port), make_handler(engine))
    print(json.dumps({
        "listen": f"127.0.0.1:{args.port}", "served_id": SERVED_ID,
        "base_id": BASE_ID, "base_revision": BASE_REVISION,
        "adapter_id": ADAPTER_ID, "adapter_revision": ADAPTER_REVISION,
        "adapter_subfolder": ADAPTER_SUBFOLDER, "adapter_sha256": ADAPTER_SHA256,
        "mode": "single-client-local-smoke",
    }), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
