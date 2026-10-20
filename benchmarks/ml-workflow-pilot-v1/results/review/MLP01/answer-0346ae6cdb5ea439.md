Test **Qwen/Qwen3-4B-Instruct-2507** and **microsoft/Phi-4-mini-instruct** against your existing 8B model. Both are plausible candidates for this workload; defer replacing the baseline until a matched evaluation measures extraction errors, memory, and latency.

| Candidate | Applicability and qualifications |
|---|---|
| **Qwen3-4B-Instruct-2507 — 4.0B** | Instruction-tuned, explicitly non-thinking, with a documented 262,144-token context window. Non-thinking output is useful with your tight 256-token extraction budget. Its instruction-following results support testing it, but do not establish accuracy on technical-document extraction. [Official model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/main/README.md). |
| **Phi-4-mini-instruct — 3.8B** | Instruction-tuned, with a documented 128K context window and training emphasizing reasoning-dense material. It offers a different model family with slightly fewer parameters. Microsoft explicitly recommends downstream evaluation; its general benchmarks do not establish extraction quality. GPU attention-kernel compatibility needs checking once your GPU is identified. [Official model card](https://huggingface.co/microsoft/Phi-4-mini-instruct/blob/main/README.md). |

**Commercial use is permitted:** Qwen’s Apache-2.0 license requires applicable license, attribution, modification, and NOTICE handling when redistributing. Phi’s MIT license requires preserving its copyright and permission notice. Neither grants trademark rights. These conclusions concern the named releases and their published terms. [Qwen license](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/main/LICENSE), [Phi license](https://huggingface.co/microsoft/Phi-4-mini-instruct/blob/cfbefacb99257ffa30c83adab238a50856ac3083/LICENSE).

**Memory looks plausible, but fit is unverified.** Start with 16-bit weights and KV cache to avoid introducing quantization as another variable. Calculated estimates for one loaded model serving two requests, each reaching 6,256 tokens:

| Candidate | Approximate weights | Combined KV cache | Subtotal |
|---|---:|---:|---:|
| Qwen | 7.45 GiB | 1.72 GiB | 9.17 GiB |
| Phi | 7.08 GiB | 1.53 GiB | 8.61 GiB |

These use rounded parameter counts and `KV bytes = 2 requests × 6,256 tokens × layers × 2(K,V) × KV heads × head dimension × 2 bytes`. Architecture values come from the [Qwen configuration](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/main/config.json) and [Phi configuration](https://huggingface.co/microsoft/Phi-4-mini-instruct/blob/main/config.json). They exclude activations, prefill workspaces, graph capture, allocator overhead, and runtime cache reservation. Configure a bounded context rather than allocating for the advertised maximum. Load the three models sequentially on the GPU.

Unknowns are the GPU model, baseline checkpoint/license/accuracy, runtime, precision, field schema, document preprocessing, acceptable quality loss, and production latency target. Whether “6,000 input tokens” includes instructions and chat-template overhead is also unknown. Reserve that overhead explicitly.

**One matched evaluation, proposed only:**

1. **Freeze the setup.** Identify and pin the 8B checkpoint; pin all model/tokenizer revisions, runtime, kernels, and precision. Use 20 development documents to finalize one extraction instruction, schema, null policy, and normalization rules. Allow native chat templates, but keep task content, greedy decoding, 256-token cap, and any schema-constrained decoding policy consistent.
2. **Evaluate 200 held-out documents.** Use identical source text across models, fitting within 6,000 total input tokens under every tokenizer. Include tables, identifiers, units, absent fields, conflicting values, and evidence near the document end. Review gold labels and adjudicate disagreements. Verify that expected records can fit within 256 output tokens.
3. **Measure at concurrency two.** Serve two active requests from one model instance. Record whole-record exact match, field precision/recall/F1, missing and unsupported values, invalid JSON, truncation, peak device memory, OOMs, p50/p95 end-to-end latency, time to first token, and documents/second. Include separate resource fixtures exercising exactly 6,000 input tokens and the full 256-token decode per request.
4. **Use a provisional decision rule.** Advance a candidate only if its paired, document-level confidence interval excludes more than a **2 percentage-point exact-match loss**, review finds no new critical-field failure pattern, and it delivers either **20% lower p95 latency or 25% lower peak memory**. These thresholds are proposed engineering choices, not supplied requirements.
5. **Stop** on OOM, unsupported execution, or silent truncation; otherwise stop after the fixed pilot. If uncertainty crosses the quality margin, retain the 8B and defer selection. Production approval also requires a latency target and acceptable error tolerance.

No inference or experiments were executed.

Sources and versions, all before 2026-10-19:

- Qwen card: revision [`eb25fbe`, 2025-08-06](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/commit/eb25fbe4f35f7147763bc24445679d1c00588d89); configuration/license last-change revision `1b4199c`; Apache-2.0, January 2004.
- Phi card: revision [`cfbefac`, 2025-12-10](https://huggingface.co/microsoft/Phi-4-mini-instruct/commit/cfbefacb99257ffa30c83adab238a50856ac3083); configuration/license last-change revision `618e63e`; MIT license.