# Investigate RL for the frozen passage reader

Authorization: user asked to investigate RL after questioning the sufficiency of
24 authored trajectories. This scope permits research, read-only trainer/history
inspection, and offline reward counterexamples. No model inference, paid endpoint,
GPU activation, optimizer update, reward deployment or runtime change. Preserve
all prior examples, datasets, protected files, paid markers and `cce82c2` runtime.

Hypothesis: outcome-supervised RL can reduce the need for authored trajectories
if the model has useful exploratory behavior and the task has trustworthy reward.
Expected signal in this local investigation: (a) direct research evidence with
actual task/data/model differences, (b) an explicit accounting of prior local RL
results and current trainer incompatibilities, and (c) reproducible counterexample
scores showing whether candidate proxies separate accepted answers/gaps from
incorrect claims, irrelevant real quotes, false refusals and useless queries.

Decision rule: any full-reward unsupported claim, incorrect task-relevant binding,
or generic/unhelpful gap blocks recommending that scorer for training. Also report
partial bad rewards and false negatives. This is an adversarial development audit,
not a calibrated semantic judge or an unbiased quality estimate. Source judgments
are implementing-assistant/model-assisted; independent human fields remain null.
Even a local pass would not establish exploration signal, optimizer readiness,
held-out quality, hardware capacity or cost effectiveness.

Use only already exposed saved runs and the twelve new training-paper pairs;
leave the six development pairs and reserved research_reader_v1 pack out of this
audit. Counterfactual terminal text is explicitly authored and not a new rollout
or replay. Source/read prefixes stay bound to saved artifacts; no model Python
executes on the host. Keep legacy MuSiQue and QASPER scorers and trainers unchanged.
Save actual inputs/results and a bounded future experiment design with gates.
