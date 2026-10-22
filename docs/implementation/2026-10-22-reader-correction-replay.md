# Frozen-harness Docker correction replay

The user said “ok lets go” after the proposed next step: replay the two reviewed
final-action corrections in Docker, without GPU inference or training. Keep the
runtime baseline `cce82c2` and the saved queue/review/drafts unchanged.

Hypothesis: the proposed source-reviewed answer and terminal gap fit the existing
interface without a harness change. Expected signal: all three original hybrid
smoke trajectories first reproduce their saved actions, observations, outcomes
and actual request payloads; then the two corrected final actions terminate with
verified answer/gap packets while their original tool actions and observations
remain identical. Decision rule: fail closed before any correction if any
original replay differs; require zero replay errors, exact prompt/source/decoding
bindings, unchanged source inspections, valid terminal results and verified
cleanup of every owned container. Evaluation papers remain excluded from training.

Use the frozen Docker image with networking disabled and the exact corpus mounted
read-only. Never run recorded Python through LocalREPL or on the host. Record
container configuration and verify removal. Use the real policy formatting and
investigator with an explicit offline transport that returns recorded actions
and the two proposed replacements. Assert each generated request payload matches
its historical input; dispatch no network/model request, read no credentials and
load no tokenizer/model. The maximum is five episodes / twenty action slots;
each episode retains the original four-step limit.

Save per-case/action receipts, raw packets, provenance verification, input hashes,
baseline comparison, correction checks and cleanup proof. A passing fixed-action
replay demonstrates interface compatibility only. It cannot measure how Qwen
would choose actions or respond to different feedback. Corrected actions remain
reviewer-authored regression demonstrations, not Qwen improvements, training-ready
examples, semantic rewards or independent human judgments.

Add focused tests for changed inputs/drafts, baseline mismatch blocking corrections,
the offline transport, and actual Docker replay. Run the full relevant suite after
focused checks. Preserve protected dirty files, prior results and consumed paid
markers. Separate training data, a training job and model promotion remain future
scoped work with no authorization to activate paid infrastructure here.
