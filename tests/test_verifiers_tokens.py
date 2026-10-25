"""Native training-format contracts only; no V5 weights, optimizer or code execution."""

from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path

import httpx
import pytest

try:
    version = importlib.metadata.version("verifiers")
except importlib.metadata.PackageNotFoundError:
    version = None
if version != "0.4.0":
    pytest.skip("requires isolated requirements-verifiers.txt", allow_module_level=True)

import verifiers.v1 as vf
from huggingface_hub import snapshot_download
from openai import AsyncOpenAI
from renderers import create_renderer
from renderers.client import MalformedGenerateResponseError
from renderers.configs import DefaultRendererConfig, Qwen3RendererConfig
from transformers import AutoTokenizer
from verifiers.v1.clients.train import TrainClient
from verifiers.v1.dialects import ChatDialect
from verifiers.v1.graph import prepare_turn
from verifiers.v1.trace import AgentInfo, TraceTask

from src.training.verifiers_adapter import local_agent_config

ROOT = Path(__file__).resolve().parents[1]
REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
CONTROLS = sorted((ROOT / "examples/verifiers-port/controls").glob("*.json"))


@pytest.fixture(scope="module")
def tokenizer():
    try:
        snapshot = snapshot_download("Qwen/Qwen3-8B", revision=REVISION, local_files_only=True)
    except FileNotFoundError:
        pytest.skip("pinned Qwen tokenizer not cached; never download weights for this check")
    return AutoTokenizer.from_pretrained(snapshot, local_files_only=True)


def oracle(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages, tokenize=True, return_dict=False,
        enable_thinking=False, add_generation_prompt=True,
    )


@pytest.mark.parametrize("control_path", CONTROLS, ids=lambda p: p.stem)
def test_exact_saved_v5_prefixes(tokenizer, control_path):
    control = json.loads(control_path.read_text())
    # Public token hashes bind this oracle to the earlier pinned V5 formatter.
    frozen = json.loads((ROOT / "examples/verifiers-port/token-preflight.json").read_text())
    fallback = create_renderer(tokenizer, DefaultRendererConfig(enable_thinking=False))
    optimized = create_renderer(tokenizer, Qwen3RendererConfig(enable_thinking=False))
    for i, payload in enumerate(control["payloads"]):
        expected = oracle(tokenizer, payload["messages"])
        assert hashlib.sha256(json.dumps(expected).encode()).hexdigest() == frozen[
            "prefix_sha256"
        ][control["case_id"]][i]
        assert (fallback.render(payload["messages"], add_generation_prompt=True).token_ids
                == expected)
        # The optimized renderer retains historical empty thinking blocks.
        assert (optimized.render(payload["messages"], add_generation_prompt=True).token_ids
                == expected) is (i == 0)


@pytest.fixture(scope="module")
def tiny_model(tokenizer):
    torch = pytest.importorskip("torch")
    from transformers import Qwen3Config, Qwen3ForCausalLM

    with torch.random.fork_rng():
        torch.manual_seed(17)
        model = Qwen3ForCausalLM(Qwen3Config(
            vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            head_dim=8, max_position_embeddings=16384,
        )).eval()
    return model


@pytest.mark.parametrize("mode", [
    "sample", "fenced", "short_logprobs", "nan", "wrong_id", "sentinel",
    "packed", "packed_short", "packed_nan", "packed_sentinel",
])
def test_native_train_client_and_graph(tokenizer, tiny_model, tmp_path, mode):
    """Mock HTTP, real native client/renderer/graph; sampled mode uses real CPU logits.

    Other modes use explicitly synthetic transport fixtures, never training data.
    Raw sampled text is never executed, and no reward or optimizer is involved.
    """
    import torch

    records = []

    def engine(request):
        assert request.url.host == "127.0.0.1"
        if request.method == "GET":
            assert request.url.path == "/v1/models"
            return httpx.Response(200, json={"data": [{"id": "envoy", "max_model_len": 16384}]})
        assert request.url.path == "/inference/v1/generate"
        body = json.loads(request.content)
        assert body["token_ids"] == oracle(tokenizer, messages)
        params = body["sampling_params"]
        assert params["temperature"] == params["top_p"] == 1.0
        assert params["max_tokens"] == (4 if mode == "sample" else 32)
        assert params["logprobs"] == 1
        if mode == "sample":
            ids, lps = [], []
            generator = torch.Generator().manual_seed(101 + len(records))
            # Real ancestral sampling, without a generator/optimizer implementation.
            with torch.inference_mode():
                for _ in range(4):
                    logits = tiny_model(torch.tensor([body["token_ids"] + ids])).logits[0, -1]
                    lp = logits.log_softmax(-1)
                    token = int(torch.multinomial(lp.exp(), 1, generator=generator))
                    ids.append(token)
                    lps.append(float(lp[token]))
        else:
            ids = tokenizer.encode("```python\n1 / 0\n```", add_special_tokens=False)
            lps = [-1.0] * len(ids)  # Synthetic evidence only for serialization/failure checks.
        entries = [{"token": f"token_id:{t}", "logprob": p} for t, p in zip(ids, lps)]
        if mode == "short_logprobs":
            entries.pop()
        elif mode == "nan":
            entries[0]["logprob"] = float("nan")
        elif mode == "wrong_id":
            entries[0]["token"] = "token_id:-1"
        elif mode == "sentinel":
            entries[0]["logprob"] = -9999.0
        records.append({"prompt_ids": body["token_ids"], "ids": ids, "logprobs": lps})
        # Raw bytes also exercise a provider's invalid non-finite JSON number.
        choice = {
            "token_ids": ids, "logprobs": {"content": entries}, "finish_reason": "length",
        }
        if mode.startswith("packed"):
            import numpy as np

            values = np.asarray(lps, dtype=np.float32)
            if mode == "packed_short":
                values = values[:-1]
            elif mode == "packed_nan":
                values[0] = float("nan")
            elif mode == "packed_sentinel":
                values[0] = -9999.0
            choice["logprobs"] = None
            choice["completion_logprobs"] = {
                "data": base64.b64encode(values.tobytes()).decode(),
                "shape": list(values.shape), "dtype": "float32",
            }
        return httpx.Response(200, content=json.dumps({"choices": [choice]}).encode(),
                              headers={"Content-Type": "application/json"})

    async def run():
        config = vf.TrainClientConfig(
            base_url="http://127.0.0.1:1/v1", api_key_var="ENVOY_VERIFIERS_UNUSED",
            renderer_model_name=tokenizer.name_or_path,
            renderer=DefaultRendererConfig(enable_thinking=False),
        )
        client = TrainClient(config)
        await client.client.close()
        client.client = AsyncOpenAI(
            base_url=config.base_url, api_key="fixture", max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(engine)),
        )
        trace = vf.Trace(task=TraceTask(type="token_contract", data=vf.TaskData(prompt="fixture")),
                         agent=AgentInfo(config=local_agent_config(config.base_url, tmp_path),
                                         trainable=False))
        try:
            for i in range(3):
                body = {"model": "envoy", "messages": messages}
                turn = prepare_turn(trace, ChatDialect().parse_request(body).messages)
                if mode not in {"sample", "fenced", "packed"}:
                    with pytest.raises(MalformedGenerateResponseError):
                        await client.get_response(ChatDialect(), body, vf.SamplingConfig(
                            temperature=1.0, top_p=1.0, max_tokens=32), turn=turn)
                    turn.abandon()
                    assert not trace.nodes and len(records) == 1
                    return
                response = await client.get_response(ChatDialect(), body, vf.SamplingConfig(
                    temperature=1.0, top_p=1.0,
                    max_tokens=4 if mode == "sample" else 32), turn=turn)
                assert response.tokens.prompt_ids == records[-1]["prompt_ids"]
                assert response.tokens.completion_ids == records[-1]["ids"]
                assert response.tokens.completion_logprobs == records[-1]["logprobs"]
                turn.commit(response)
                if mode == "fenced":
                    assert response.message.content == "```python\n1 / 0\n```"
                messages.extend([response.raw["choices"][0]["message"], {
                    "role": "user", "content": "ZeroDivisionError: correct the action" if i == 0
                    else "Source observation: literal text, not a sampled assistant action",
                }])
            assert len(trace.branches) == 3  # Full renders fork when historical templates change.
            for branch, record in zip(trace.branches, records):
                n = len(record["prompt_ids"])
                assert branch.token_ids == record["prompt_ids"] + record["ids"]
                assert branch.sampled_mask == [False] * n + [True] * len(record["ids"])
                assert branch.logprobs == [0.0] * n + record["logprobs"]
            (tmp_path / "token-trace.json").write_text(json.dumps(trace.to_record(), indent=2))
            (tmp_path / "engine-records.json").write_text(json.dumps(records, indent=2))
        finally:
            await client.close()

    messages = copy.deepcopy(json.loads(CONTROLS[0].read_text())["payloads"][0]["messages"])
    asyncio.run(run())
