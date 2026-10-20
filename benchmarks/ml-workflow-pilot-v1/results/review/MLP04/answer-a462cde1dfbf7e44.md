**Yes—speculative decoding merits a small, conditional test, but a 2–3× speedup is not a responsible expectation.** Start with prompt-lookup speculation if answers commonly repeat document passages. Defer selecting a neural draft until the checkpoint, runtime compatibility and memory headroom are known. No experiment was executed.

The primary evidence supports the mechanism, with substantial transfer limitations:

| Primary paper | Reported conditions and results | Applicability here |
|---|---|---|
| Leviathan et al., *Fast Inference from Transformers via Speculative Decoding*, ICML 2023 | T5-XXL 11B encoder–decoder, batch 1, one TPU-v4, T5X. A 77M draft achieved 2.3× sampled summarization speedup with acceptance 0.53; a larger 800M draft raised acceptance to 0.56 but reduced speedup to 1.7×. | Strong evidence that acceptance alone is insufficient: draft cost matters. Neither the architecture nor hardware/concurrency matches this scenario. [Paper](https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf) |
| Chen et al., *Accelerating Large Language Model Decoding with Speculative Sampling*, v1, February 2023 | Chinchilla 70B plus a specially trained, shallow 4B draft on 16 TPU-v4s, batch 1, four-token lookahead. Draft/target token times were 1.8/14.1 ms; sampled XSum achieved 1.92× and HumanEval 2.46×. | Demonstrates same-hardware drafting, but with a deliberately optimized draft and much larger target. Reported decoding gains do not establish end-to-end gains for 4,000-token inputs. [Paper](https://arxiv.org/pdf/2302.01318v1) |

For this workload:

- **Overhead and acceptance:** A neural draft adds its own 4,000-token prefill, sequential drafting, KV cache and verification overhead. Its parameter ratio does not determine its latency ratio. Measure accepted draft tokens/proposed tokens, committed tokens/verification round, and actual round time. Break-even requires speculative time per committed token below baseline.
- **Memory:** Calculated raw 8B weights occupy approximately 16 GB at two bytes/parameter, before caches and runtime allocations. Two requests reach roughly 8,400 resident tokens combined, plus speculative slots. Exact KV memory depends on unknown layer count, KV heads, head dimension and cache precision. Target and draft weights, both caches, workspaces and graph allocations must fit together on the existing GPU.
- **Concurrency:** Two active requests already provide batching opportunities. Verification adds compute and may compete with the other request’s prefill. Concurrency two does not establish QPS or a memory-bound regime; vLLM explicitly targets medium-to-low-QPS, memory-bound workloads. [Implementation guidance](https://docs.vllm.ai/en/v0.17.1/features/speculative_decoding/)
- **End-to-end benefit:** With 4,000 input and 200 output tokens, measure prefill’s latency share rather than infer it from token counts. If fraction \(f\) of baseline latency is decode, ideal decode speedup \(s\) gives total speedup \(1/[(1-f)+f/s]\), before added overhead.

**Decoding guarantees matter.** Both papers preserve the target sampling distribution through corrected acceptance/rejection sampling, subject to numerical precision. Require support for the actual temperature, top-p, penalties and constraints; prohibit approximate acceptance. Distribution preservation does not imply identical sampled strings for the same seed. vLLM documents numerical and batching qualifications and no stable-logprob guarantee. Keep target precision unchanged: quantizing it solely to accommodate a draft changes the reference behavior. [Guarantees](https://docs.vllm.ai/en/v0.17.1/features/speculative_decoding/)

**Smallest useful proposed test:**

1. Record unknowns: checkpoint/tokenizer, GPU, runtime/version, weight/cache precision, decoding settings, scheduler/cache policy, baseline latency and free memory. If the runtime lacks a supported lossless path, defer.
2. Compare target-only serving against **one** speculative configuration: prompt lookup, three proposed tokens, lookup length four. This avoids neural draft weights; document-copying overlap remains unknown. vLLM documents both [prompt lookup](https://docs.vllm.ai/en/v0.17.1/features/speculative_decoding/n_gram/) and [neural drafts](https://docs.vllm.ai/en/v0.17.1/features/speculative_decoding/draft_model/).
3. After warm-up, replay 100 representative request pairs twice per arm, alternating arm order, at concurrency two and approximately 4,000/200 tokens. Preserve prompts, production stopping rules, precision and cache policy.
4. Measure median/p95 request completion latency, TTFT, time/output token, aggregate output tokens/second, acceptance, verification/draft time, peak memory and preemptions. Audit sampler correctness; greedy equality alone cannot validate stochastic distribution preservation.
5. **Proposed decision rule:** continue only with ≥10% median completion-latency reduction in both repeats, ≤5% p95/TTFT regression, no throughput loss, and no correctness failure. Stop immediately on OOM, new preemption or unsupported sampling behavior; otherwise stop after the capped comparison. Inconclusive results mean defer.

Source versions, all before 2026-10-19:

- [Leviathan et al.](https://proceedings.mlr.press/v202/leviathan23a.html), ICML 2023, PMLR 202.
- [Chen et al.](https://arxiv.org/abs/2302.01318v1), arXiv v1, 2023-02-02.
- [vLLM documentation](https://docs.vllm.ai/en/v0.17.1/features/speculative_decoding/), pinned v0.17.1; cited pages display 2026-02-18. Examples establish configuration support, not performance on this GPU.