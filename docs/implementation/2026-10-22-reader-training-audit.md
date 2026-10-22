# Audit V5 before another reader SFT experiment

The user approved a small targeted SFT pilot after asking whether distillation
fits the observed reader failures. This increment audits existing training data
and defines a concrete next data/pilot protocol. No GPU launch, teacher/model
request, optimizer update, reward change or runtime harness change is included.
Read AGENTS.md, the Astra improvement brief, technical case study, V5 dataset
card/manifests and the earlier learning/V6 experiment records first.

Hypothesis: the V5 examples do not demonstrate the current narrow passage-reader
contract, especially exact evidence submission and terminal submit_gap. Expected
signal: verify original input hashes, count answerable/insufficient labels and
terminal styles without treating lexical flags as semantic judgments, compare
action/prompt forms with frozen passages-v2, and reproduce next-action masking
and supervised-token totals with the cached pinned tokenizer. Decision rule:
do not reuse incompatible old conversations as current-protocol training data,
do not infer that refusal frequency caused the model failures, and do not launch
another training job from this audit. Every number must be traceable to rows.

Reuse existing audit/preprocessing helpers; add only a read-only alignment audit
and focused tests. Preserve V5 data, trainer, reward, old reviews and all saved
baseline/replay artifacts. Record current and historical trainer hashes separately.
Compare paper/question identities against the exposed EnvoyBench sets, current
three-case regression set and the old V6 holdout lock without opening new reserved
question packs or full holdout papers. A lock's historical untouched label must
not override the completed V6 comparison: that holdout is now consumed.

The September tiny-overfit diagnostic already passed, and October V6 already
tested a broader protocol-aligned SFT revision without passing promotion. Do not
repeat either as an untested new idea. Define a distinct narrow pilot focused on
evidence-conditioned extraction versus scoped structured gaps. Keep source review,
Docker replay, train/development paper exclusions, independent evaluation and
paid budget gates explicit. No invented demonstration observations or automatic
semantic labels, and no promotion from training loss or fixed-action replay.
