# Envoy demo and evidence

Envoy is a research prototype: Qwen writes Python in a persistent environment to
search papers, inspect passages, and submit an answer with evidence. It is not yet
an unattended weekly research assistant. The demo below uses **recorded Qwen
inference**, so it runs without a GPU or API key. The viewer labels these traces
as recorded; it does not pretend to run Qwen live.

## Three-minute walkthrough (after setup)

Install dependencies once from a fresh checkout; the ML packages may take longer
than the walkthrough itself:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[viewer]'
python scripts/viewer.py
```

Open <http://127.0.0.1:8000>. In **Agent & Replays**, leave the recorded-trace selector
on **one answerable question, three checkpoints** and click **Watch Replay**.
The three lanes are Qwen3-8B with the starting SFT adapter, then the same adapter
after one and two further training epochs. On this question, the starting model
refuses, epoch 1 finds the answer, and epoch 2 refuses again. The actions and
observations come from saved evaluation episodes, not a script that invents model
actions.

Select **an unanswerable question, three checkpoints** and replay it. The starting
model searches and abstains; both continuation checkpoints confuse a mention of
“10 random seeds” with the requested GPU count and submit unsupported answers.
That example explains a major regression in the aggregate evaluation. Both
replays were selected to illustrate the observed tradeoff; neither is a substitute
for the 40-question result. Observations in the bundled replays are excerpted to
400 characters per action. Actions, answers, rewards, and review verdicts are
copied from the recorded runs.

The [public evaluation report](../reports/qasper-scale-sft-2026-09-25.json)
contains the exact question IDs, model and corpus hashes, decoding settings,
hardware, paired results, and decision rule. The model-assisted review was blind
to checkpoint identity, but it was **not independent human review**. The full raw
episodes live in the ignored local `out/research/qasper-scale-sft-v1/` directory;
the public report gives their archive hash. If those local artifacts are present,
rebuild the two replay bundles with:

```bash
python scripts/build_qwen_eval_replays.py \
  --eval-root out/research/qasper-scale-sft-v1/eval-20260925 \
  --review-dir out/research/qasper-scale-sft-v1/eval-20260925-blind-review
```

## What the result shows

The earlier targeted SFT checkpoint improved over base Qwen on a separate locked
QASPER evaluation (9/40 to 19/40 semantic passes). The later scale-continuation
experiment used a *different* reserved 40-question set, so its counts must not be
compared directly with that earlier set. On the later set, starting SFT passed
15/40, epoch 1 passed 10/40, and epoch 2 passed 4/40. Neither continuation
checkpoint passed the preregistered promotion rule.

The error analysis is more useful than “more training was bad.” On 20 answerable
questions, epoch 1 improved from 5 to 8 passes. On 20 insufficient-evidence
questions, it fell from 10 to 2 passes; epoch 2 passed none. The continuation
training set contained only 31 abstention examples among 796 training
conversations. That imbalance is a plausible explanation for over-answering,
not a proven causal mechanism. More training is paused until the same kind of
failure is tested on named personal-vault questions.

## What to show an ML interviewer

Open the code for the [persistent Python environment](../src/env/repl.py),
[trajectory training and action-only loss](../scripts/train_sft.py),
[audited training-data assembly](../scripts/assemble_qasper_scale_sft.py), and
[evaluation harness](../scripts/run_eval.py). Explain one concrete engineering
choice in each: code actions retain state across turns; training masks retrieved
text rather than optimizing on tool output; examples are split by paper; and
comparisons pin the same questions, corpus, prompt, seed, and decoding settings.
Then show the negative result and the decision not to promote a checkpoint.

The honest claim is: **the research loop and evaluation pipeline work, and one
targeted SFT improved a paper-QA benchmark; a later scaling attempt failed and
was rejected.** Do not claim that the personal-vault product or weekly digest is
finished, that the model reliably answers arbitrary research questions, or that
the model-assisted review is a human judgment.

## What would make the full product demo credible

The next milestone is a small frozen vault with substantive notes and 10–20 named
questions. Run base Qwen and the selected starting SFT adapter on the same
questions, review their evidence-backed answers, and record failures, latency,
and cost. Then demonstrate one useful weekly digest saved to Obsidian and query
the accumulated evidence through MCP. Those are the project’s product and model
improvement gates; the recorded QASPER replays are a clear view into the ML work
while those gates remain open.

QASPER questions and annotations are attributed to [Dasigi et al., 2021](https://arxiv.org/abs/2105.03011)
and the [AllenAI QASPER dataset](https://huggingface.co/datasets/allenai/qasper).
