# QASPER targeted SFT continuation

**Run date:** September 20, 2026  
**Status:** training and development evaluation complete; promotion gate failed

## Question

Can a small, balanced continuation set teach the Qwen3-8B Envoy adapter to recover after a weak
within-paper search, answer when evidence exists, and retain correct abstention?

The prior `checkpoint-50` result improved semantic passes from 9/40 to 19/40, required-paper recall
from 0.65 to 1.00, and execution-error episodes to zero. Its main remaining weakness was the
answerable half of the set: 5/20 passes and seven false refusals. Failure attribution found no case
where the existing retrieval backend was fundamentally unable to retrieve the annotated evidence.

## Hypothesis and gates

**Hypothesis:** annotation-grounded recovery demonstrations will reduce false refusals and teach a
second, answer-shaped search after a topical query misses, without losing the adapter's tool
reliability.

The development gate is at least 6/15 rescued answerable failures, at least 9/10 retained controls,
and at most one execution-error episode on the existing 25-question diagnostic. A checkpoint that
passes that gate is compared with the old adapter on a newly frozen, paper-disjoint 40-question
confirmation set. The promotion target is at least 28/40 semantic passes and a paired improvement
over the old adapter. These are project gates, not a statistical SOTA claim.

The old 40-question evaluation has been consumed by failure analysis. It remains development data
and will not be described as held out again.

## Data

`scripts/build_qasper_annotation_sft.py` considered 200 questions from separate QASPER training
papers. It generated only actions whose search strings contain question words plus a fixed generic
vocabulary. Gold annotations select and verify a path, but unseen answer entities are never inserted
into a search query. Every accepted action sequence was executed from scratch in the persistent
REPL. For answerable examples, the final answer appears only after the annotated evidence has been
printed, re-located with `read()`/`passage()`, and assigned exact offsets.

The new set contains 148 accepted trajectories:

| Behavior | Conversations |
| --- | ---: |
| Direct top-three evidence retrieval | 39 |
| Recovery with a distinct top-eight search | 69 |
| Investigated, annotation-backed abstention | 40 |

Fifty-two answerable candidates were excluded because no query built under the non-leakage rule
surfaced their evidence. The generated split contributes 118 training and 30 validation
conversations. Replaying the 35 original v5 training conversations produces the final 153-example,
611-action training set. The new validation split has 125 actions and is disjoint by paper.

All 611 training actions and 125 validation actions match the Qwen3 non-thinking inference prefix.
The longest sequence is 4,655 tokens, below the 8,192-token limit.

## Training protocol

- Base: `Qwen/Qwen3-8B`, revision `b968826d9c46dd6066d109eabc6255188de91218`
- Starting adapter: QASPER v5 `checkpoint-50`
- Method: rank-4 QLoRA continuation, NF4 base, assistant-action-only loss
- Learning rate: `5e-5`
- Epochs: 2
- Effective batch size: 4 (`batch_size=1`, `gradient_accumulation_steps=4`)
- Context limit: 8,192 tokens
- Checkpoints: every 25 optimizer steps
- Hardware: one NVIDIA A40 on RunPod
- Harness during evaluation: unchanged raw `search_within()` top-three default

### What the training details mean

The 8,192-token context limit is the maximum size of one training example. One example can include
the research question, several Python actions, the paper passages returned by those actions, and
the final cited answer. It is not the number of examples or the total amount of training data. The
longest example in this run is 4,655 tokens, so no target answer is silently cut off.

"Prefix aligned" means that each supervised action begins with exactly the same Qwen conversation
tokens it receives when the agent runs. Qwen's chat template inserts model-specific control tokens
around system, user, and assistant messages. If an intermediate action is trained with one set of
control tokens but generated with another, the optimization target and production input differ even
though the visible text looks identical. The exporter therefore creates one next-action example per
assistant turn and verifies token-for-token equality with the non-thinking inference prefix before
training. Only the assistant's next Python action or final `SUBMIT` line receives loss; retrieved
paper text is context, not text the model is asked to reproduce.

These are implementation and reproducibility details. The project result is determined by the
paired behavioral evaluation below, not by context length or token accuracy alone.

### How this QLoRA run works

Qwen3-8B starts with about eight billion learned parameters. This run does not rewrite all of
them. LoRA inserts a much smaller set of trainable adapter parameters into selected attention and
feed-forward layers. The original Qwen weights remain frozen; inference combines the frozen model
with the learned adapter. The `Q` in QLoRA means the frozen base is loaded in 4-bit NF4 form to
reduce GPU memory use. The resulting adapter is only tens of megabytes, although inference still
needs the Qwen3-8B base model.

The adapter targets seven projections in every transformer block. `q_proj`, `k_proj`, `v_proj`,
and `o_proj` are the attention pathways that route information between the question, prior actions,
and retrieved evidence. `gate_proj`, `up_proj`, and `down_proj` are the feed-forward pathways that
transform that information into the next-token decision. There is no known isolated
"paper-research parameter" in Qwen; the behavior is distributed. Broad attention-plus-MLP coverage
is therefore used with a low rank to constrain capacity. Attention-only, MLP-only, and all-linear
targeting could be compared in a future controlled ablation, but the present run changes only the
data while preserving the existing adapter architecture.

Each example contains the question, the Python actions and paper passages observed so far, and the
correct next action. That target can be another search/read command or the final cited answer. Loss
is applied only to the next assistant action. Retrieved paper text provides context and is not text
the model is trained to reproduce.

The learning rate is not constant. It warms up during the first 5% of optimizer steps until it
reaches `5e-5`, then follows a cosine schedule down toward zero. Early updates can move the adapter
meaningfully; late updates make small refinements. The effective batch size is four: the GPU
processes one action example at a time and accumulates four examples before an optimizer update.

The optimizer is PyTorch's fused AdamW (`adamw_torch_fused`), with beta values 0.9 and 0.999,
epsilon `1e-8`, and weight decay 0. AdamW performs the gradient-descent updates after
backpropagation, adapting each parameter's step using moving averages of its recent gradients and
squared gradients. Only LoRA adapter parameters are updated; the 4-bit base remains frozen.

Regularization and overfitting controls are:

- Rank-4 LoRA limits how much the adapter can change the base model.
- LoRA dropout of 0.05 randomly disables some adapter connections during training.
- The run is limited to two epochs.
- Thirty paper-disjoint validation conversations never contribute gradients.
- Both the epoch-1 and epoch-2 candidates are retained, and behavioral evaluation selects between
  them rather than automatically using the last checkpoint.

Gradient checkpointing is enabled to reduce GPU memory use. Despite its name, it is not a form of
regularization and is unrelated to the saved model checkpoints.

A one-step smoke run successfully loaded the previous adapter as trainable, completed an optimizer
step and validation pass, and saved a reloadable adapter before the full run began. The full run and
checkpoint evaluation completed in `tmux`. Both adapter candidates and the resumable epoch-1
checkpoint were downloaded and hash-verified before the RunPod GPU was stopped.

## Result

Training completed all 306 optimizer steps in 2,459 seconds. Held-out next-action loss improved in
both epochs, while token accuracy was effectively flat between the two saved candidates:

| Candidate | Validation loss | Validation token accuracy |
| --- | ---: | ---: |
| Epoch 1 (`checkpoint-150`) | 0.1356 | 0.9664 |
| Epoch 2 (`final`) | **0.1272** | 0.9661 |

This establishes that the adapter learned the new trajectory distribution on paper-disjoint
validation examples. It does not by itself establish better autonomous research behavior.

Both candidates were then evaluated on the predeclared 25-question development diagnostic under
the unchanged raw top-three harness. The semantic review was performed by the assistant while
blind to candidate identity; it is not independent human review.

| System | Pass | Partial | Fail | Target rescues | Controls retained | Execution errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Previous `checkpoint-50` | 10 | 4 | 11 | 0/15 | 10/10 | 0 |
| Epoch 1 (`checkpoint-150`) | **11** | **5** | **9** | **1/15** | 10/10 | 0 |
| Epoch 2 (`final`) | **11** | 4 | 10 | **1/15** | 10/10 | 0 |

Epoch 1 is the best behavioral candidate: compared with the previous adapter, one diagnosed failure
became a supported answer and one became partially useful, with all controls and tool reliability
preserved. The improvement is much smaller than the required six rescues. The development gate
therefore **failed**, and the adapter is archived as experimental rather than promoted.

The new 40-question paper-disjoint confirmation set remains untouched. Running it after the failed
development gate would spend the project's final clean comparison on a checkpoint that did not meet
the preregistered threshold.

Artifacts and exact hashes are recorded in
[`out/research/qasper-targeted-sft-v1/summary.json`](../out/research/qasper-targeted-sft-v1/summary.json).
The verified local archive includes both adapter candidates, the epoch-1 optimizer state, logs,
manifests, and complete evaluation transcripts. The A40 pod was stopped after 4,066 seconds of
uptime; estimated GPU cost for the session is $0.55 at $0.49/hour.
