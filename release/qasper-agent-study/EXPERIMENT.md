# Bounded NVIDIA/Nebius reference run — October 9, 2026

Recorded before sending the first inference request for this comparison.

**Hypothesis:** a stronger hosted NVIDIA model may improve literal QASPER Answer
F1 and produce longer source investigations than the saved Qwen v5 worker.
This is an additional reference configuration, not a controlled weight-only
comparison or a new held-out benchmark.

**Protocol:** run all 40 frozen `test_candidate` questions once, in their existing
order, against `nvidia/Nemotron-3-Ultra-550b-a55b` on Nebius Token Factory.
Use the same paper snapshot, known-paper prompts and existing Python tool API;
maximum 15 actions including submission, 1,024 output tokens per call,
temperature 0, top-p 1, local seed 42. Request thinking disabled using
`chat_template_kwargs.enable_thinking=false`; request nonempty final content.
Record any returned reasoning separately. No internet access in the agent sandbox.
No retries, answer-guided prompt changes, best-of selection, or new training.

**Primary result:** pinned official QASPER Answer F1 against all original
annotations for these IDs. Score the literal submitted answer; absent submission
scores zero. Separately show submission rate, steps, tool errors, token usage,
and catalog-rate estimated cost. Do not invent independent support-review labels
for the hosted run. Quote integrity is not semantic support.

**Decision rule:** publish the result whichever direction it goes. A higher F1
supports only the claim that this configuration obtained greater answer-token
overlap on this subset. It does not prove better supported answers, a general
model ranking, or a fine-tuning effect. If infrastructure or the budget stops
the run, publish it as incomplete, never as a full 40-question result.

**Bounds:** at most 600 sequential API requests, 100,000 UTF-8 input bytes per
request, and a $2 local estimated-cost ceiling. Before each call reserve input
bytes plus generous chat framing at $1/million input tokens and the output
allowance at $3/million output tokens; settle from returned usage. Keep the full
reservation for failed or unreported usage. Stop after any endpoint failure or
three consecutive error episodes. This conservative local control is not a
provider invoice or account spending limit. No Runpod machine is needed.

**Known comparison limitations:** the saved Qwen endpoints used an 8,192-token
context cap; the hosted Ultra catalog advertises 1,048,576 and this run uses a
byte ceiling rather than the identical tokenizer/context cap. Qwen base had
11 context failures. The current runner/tool implementation differs from
September 30 and is hash-recorded. The provider revision, hardware and serving
implementation are not attested; catalog quantization is FP4. The 40 questions
have already been inspected, so this is descriptive development analysis.
Independent human reference and answer-support review remains unfinished.

## Reproduce (paid API run)

Load `NEBIUS_API_KEY` into the environment without putting it in Git, then:

```sh
python -m benchmarks.envoybench.run \
  --dataset benchmarks/envoybench/data --split test_candidate \
  --models release/qasper-agent-study/models.json \
  --budget release/qasper-agent-study/budget.json \
  --max-steps 15 --seed 42 --output out/qasper-nebius-reproduction
```

The original run is saved under `release/qasper-agent-study/nebius-run/`.
The runner checkpoints each finished question before continuing.

The run began from a working tree based on `e82e33b`; its manifest records that
HEAD and the exact implementation file hashes. Those ten source hashes all
match commit `e99b165`, committed while the run was in progress. This identifies
the executed sources without pretending the working tree was clean at launch.

Pricing source: [Nebius public model catalog](https://tokenfactory.nebius.com/api/public/models_info),
retrieved October 9 and preserved in `provider-catalog.json`.
Thinking parameter source: [NVIDIA Ultra model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4).

## Recorded outcome

The request guard stopped at **$1.947862** estimated usage after 366 successful
API requests. Of the planned 40 questions, 38 reached their episode end, the
39th was interrupted after 11 actions, and the 40th was not attempted. The
39 saved rows contain 27 submissions, 11 episodes with no submission, and one
budget error. The incomplete manifest and `results.partial.json` are preserved;
there is no `results.json`, full-run F1, or completed-run Studio row. No
configuration was changed during inference. Permission for a higher ceiling
was requested asynchronously but had not been received when the guard stopped.

## Subsequent authorized amendment

After this original run stopped, the user authorized a **$25 cumulative local
estimated-cost ceiling** to finish the final two questions. The
[amendment recorded before further inference](CONTINUATION_2026_10_09.md)
preserves this original $2/no-retry protocol as the historical record. Because
the live state of question 39 could not be restored, its continuation restarted
at step 1; question 40 received its first attempt. The first 38 completed rows
were retained unchanged. The derived
[40-row result](nebius-amended-run/results.json), [lineage manifest](nebius-amended-run/manifest.json),
and [official score](nemotron-official-score.json) are the completed **amended**
result, not a completion under the original no-retry protocol. Its Answer F1 is
**19.02565047926817%** over all 40 selected questions (28 submissions, 12
missing predictions). Total recorded usage is 383 requests and a $2.035422
catalog-rate estimate across both phases. No independent support review was
performed; the different provider and runtime prevent a controlled comparison
with the September 30 Qwen runs.
