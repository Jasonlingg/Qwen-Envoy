# Qwen v5 thinking smoke

**Hypothesis:** enabling Qwen3 thinking may improve the pinned v5 model's paper investigation, but it may also spend the output budget before producing executable Python. This is an inference comparison, not training.

The [two-arm config](models.thinking.example.json) sends the same pinned v5 LoRA to the same endpoint. Both arms use seed 42, temperature 0.6, top-p 0.95, top-k 20, min-p 0, a 4,096-token per-turn output limit, an 8,192-token serving context, and a 300-second request timeout. The only request difference is `enable_thinking`. The Qwen settings avoid greedy decoding, which [Qwen warns against in thinking mode](https://huggingface.co/Qwen/Qwen3-8B); the larger limit addresses the earlier 1,024-token pilot that exhausted its budget before an action. The vLLM [`qwen3` reasoning parser](https://docs.vllm.ai/en/v0.30.0/features/reasoning_outputs/) exposes generated reasoning separately from the final action. This is *generated reasoning text*, not hidden model internals.

On the GPU host, run the pinned v5 server with parsing enabled (install vLLM 0.30.0 first if needed):

```bash
QWEN_REASONING_PARSER=qwen3 MAX_MODEL_LEN=8192 MAX_NUM_SEQS=1 bash scripts/serve_vllm.sh
```

On the client, use an SSH tunnel if the server is remote, then set `ENVOY_QWEN_ENDPOINT`. The first command checks frozen data and config offline; the second makes billable model calls and requires the labeled Docker sandbox. Verify that the endpoint actually runs vLLM 0.30.0 with the pinned v5 adapter: runtime and checkpoint identity in the run manifest are operator-declared, not attested by the endpoint. Record the actual GPU and hourly rate in a copy of the config before a real run.

```bash
export ENVOY_QWEN_ENDPOINT=http://127.0.0.1:8000/v1
python -m benchmarks.envoybench.run \
  --dataset benchmarks/envoybench/data --split dev \
  --models benchmarks/envoybench/models.thinking.example.json \
  --question-id qasper_validation_0d34c0812f1e69ea33f76ca8c24c23b0415ebc8d \
  --seed 42 --max-steps 4 \
  --output out/envoybench/thinking-smoke-check --validate-only

python -m benchmarks.envoybench.run \
  --dataset benchmarks/envoybench/data --split dev \
  --models benchmarks/envoybench/models.thinking.example.json \
  --question-id qasper_validation_0d34c0812f1e69ea33f76ca8c24c23b0415ebc8d \
  --seed 42 --max-steps 4 \
  --output out/envoybench/thinking-smoke
```

**Smoke signal and decision:** inspect `results.json` for a nonempty `trajectory[].reasoning` in the on arm, no reasoning in the off arm, and a valid Python or `SUBMIT` action outside the reasoning in each arm. If the on arm never yields an action or errors, fix the serving/budget path before spending more GPU time. If it works, run a paired development set with the same settings, then judge supported-answer quality, execution failures, output tokens, and latency before considering reasoning-specific training. One question is a plumbing check, not a quality score. The historical 40-question run used greedy decoding and a 1,024-token limit, so its scores are **not comparable** to this configuration.
