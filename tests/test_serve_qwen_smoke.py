"""Request-contract checks for the loopback Qwen bridge; no weights are loaded."""

from __future__ import annotations

import json
import threading
from http.server import HTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from scripts.serve_qwen_smoke import (
    BASE_REVISION,
    SERVED_ID,
    make_handler,
    resolve_weights,
    validate_request,
)
from src.policies.openai_compatible import OpenAICompatiblePolicy


def _payload(**changes):
    payload = {
        "model": SERVED_ID,
        "messages": [{"role": "system", "content": "Write code"},
                     {"role": "user", "content": "Find the answer"}],
        "max_tokens": 1024,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    payload.update(changes)
    return payload


@pytest.mark.parametrize("changes", [
    {"model": "wrong"},
    {"temperature": 0.1},
    {"seed": 0},
    {"chat_template_kwargs": {"enable_thinking": True}},
    {"stream": True},
    {"max_tokens": 2048},
    {"messages": [{"role": "tool", "content": "unsupported"}]},
])
def test_rejects_requests_that_break_pinned_smoke(changes):
    with pytest.raises(ValueError):
        validate_request(_payload(**changes))


def test_rejects_unpinned_base_path_before_loading_weights(tmp_path, monkeypatch):
    pinned = tmp_path / "snapshots" / BASE_REVISION
    pinned.mkdir(parents=True)
    (pinned / "config.json").write_text("{}")
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download", lambda **_kwargs: str(pinned),
    )
    monkeypatch.setattr(
        "scripts.serve_qwen_smoke.verify_adapter",
        lambda *_args: pytest.fail("adapter should not be checked after base mismatch"),
    )
    with pytest.raises(ValueError, match="not the cached snapshot"):
        resolve_weights(other, Path("adapter"), None)


def test_product_policy_round_trips_through_local_http_server():
    class FakeEngine:
        calls = []

        def generate(self, messages, max_tokens):
            self.calls.append((messages, max_tokens))
            return 'print(search("answer"))', 18, 8, "stop"

    engine = FakeEngine()
    server = HTTPServer(("127.0.0.1", 0), make_handler(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}/v1"
    try:
        with urlopen(f"{endpoint}/models") as response:
            models = json.load(response)
        assert [item["id"] for item in models["data"]] == [SERVED_ID]

        policy = OpenAICompatiblePolicy(
            endpoint=endpoint, model=SERVED_ID, system_prompt="Write code",
            max_tokens=1024, temperature=0.0,
            extra_body={"top_p": 1.0, "seed": 42,
                        "chat_template_kwargs": {"enable_thinking": False}},
        )
        assert policy.act("Find the answer") == 'print(search("answer"))'
        assert engine.calls == [(_payload()["messages"], 1024)]

        request = Request(
            f"{endpoint}/chat/completions", data=json.dumps(_payload(seed=7)).encode(),
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(HTTPError) as exc:
            urlopen(request)
        assert exc.value.code == 400
        assert "seed=42" in json.loads(exc.value.read())["error"]["message"]
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
