# Training-answer audit and native pilot cost projection

The two-update native integration gate passed. This follow-up reviewed **all 22
training answers rejected for unrecognized wording**, then rescored all 44 saved
episodes with proposed aliases in memory. It used no GPU, new inference calls or
optimizer updates. Active reward keys, sources, runtime and the locked evaluation
remain unchanged. No commits or uploads were made.

The hypothesis, scope and decision rule were recorded in a
[review policy](../examples/verifiers-port/reader-reward-review-20261024/review-policy.json)
before reviewing answer contents: distinguish complete source-supported answers
from incomplete, unsupported or ambiguous responses; retain every existing
inspection and citation check when testing wording additions. These are attributed
**model-assisted judgments**, not independent human labels. Exact serving model ID
is unavailable; independent human review remains null.

## What the 22 answers contain

| Model-assisted classification | Answers | Examples / reason |
| --- | ---: | --- |
| Complete answer supported by the reviewed paper source | 7 | “300 hours”; reordered five-domain list; “Neutral”; BiLSTM/max-pooling; two German macro-F1 answers |
| Incomplete | 7 | fastText without SVM; method name without Axelrod; doctrine name without the requested meaning; F1 without its macro qualification |
| Unsupported or irrelevant assertions | 7 | Wrong categories/class, invented language list, unrelated fragments, additional unsubstantiated assertions |
| Ambiguous question target | 1 | “Gating Mechanism”: paper discusses both word and sentence representations; key targets sentence encoding but question only says “representations” |

“Supported by the reviewed paper source” describes answer content, **not** whether
the episode submitted sufficient citations. For example, the BiLSTM answer is in
the reference paragraph but its submitted fragment only mentions hyperparameters.
The “Neutral” answer cites an abstract that never names the class. The two size
answers give the requested number and unit; the original variants additionally
require the description “English conversational speech.” The first-person German
answer is interpreted as reporting the paper's result, with the German language
binding supplied by the question. These interpretations remain reviewable.

The ambiguous gating answer receives no proposed alias. Do not declare every
wording rejection a model error, or promote answers by matching a corpus/name
substring while ignoring other assertions. The long BABEL response mentions the
right corpus but adds claims its citation does not substantiate.

Every row has its question, exact answer, decision, rationale, source/context hash,
raw rollout hash and counterfactual result in the
[answer review](../examples/verifiers-port/reader-reward-review-20261024/answer-review.json).
[Review inputs](../examples/verifiers-port/reader-reward-review-20261024/review-inputs.json)
retain the original finite variants, required support quotes, submitted source
quotes and neighboring text for all 22 rows. All questions have training roles;
none is a reward-validation or locked evaluation question.

## Fixed-output counterfactual

Seven exact candidate phrases were appended to copied cards in memory. The
unchanged reference validator accepted their structure/bindings. The unchanged
scorer then evaluated all 44 original episodes, retaining inspection receipts,
scope, quote clipping, required support spans and extra-citation checks.

| Result | Original keys | Proposed aliases, in memory only |
| --- | ---: | ---: |
| Reward-positive saved episodes | 4 / 44 | 7 / 44 |
| Mixed-reward four-episode groups | 2 / 11 | 4 / 11 |

Of the seven proposed aliases, **three** pass all existing checks: both Switchboard
size answers and one German macro-F1 answer. **Two** still lack the required
reviewed support span (BiLSTM and Neutral). **Two** include an additional citation
without the required reviewed support (domain list and German first-place answer).
Those mechanical outcomes do not establish that every extra contextual citation
is semantically wrong; they reflect the existing strict anti-stuffing contract.

The two newly mixed groups are Switchboard size and German sub-task performance.
The other nine groups retain their prior mixed/uniform status. All original
episodes remain untrainable; no batch was admitted or regenerated. Details:
[proposed aliases](../examples/verifiers-port/reader-reward-review-20261024/proposed-aliases.json),
[all episode/group results](../examples/verifiers-port/reader-reward-review-20261024/counterfactual.json).

This is a **fixed-output replay**, not a run of the model with new rewards or
feedback. It cannot predict how later policies would answer, how collection order
would change after an earlier mixed group, or whether training would improve the
model. The 4/11 counterfactual mixed fraction is a sensitivity input, not a measured
acceptance rate for a revised policy. No positive-score change is an improvement
in model behavior. The audit supports a training-key coverage revision; it does
not support attributing all remaining zeros solely to model capability.

## Cost and timing

Projection uses the successful run's preserved remote receipt timestamps and
quoted **$1.90/hour combined planning rate**, including temporary storage. It is
a historical scenario, not a current GPU offer, invoice or provider-enforced dollar
cap. The actual earlier bill is still pending. Gradient time alone is unsuitable:

| Observed stage | Cycle 1 | Cycle 2 |
| --- | ---: | ---: |
| Startup/load until collection | 91 s | 54 s |
| Collect four-episode groups | 77 s / 1 group | 547 s / 10 groups |
| Scoring and preparation | 156 s | 133 s |
| Native training, save and fresh reload | 401 s | 321 s |
| Gradient update, included in preceding row | 24.9 s | 22.0 s |

Native work was 1,783 s. Setup, evidence export and cleanup outside native work
added 419 s, for 2,203 s owned lifetime. File timestamps have one-second granularity
and include coordination delays. The projection averages 56.7 s per collected
group and 579.7 s of other workflow time per verified update. It keeps the current
30-minute **evaluation allowance**, whose sufficiency has not been measured.

| Ten-update scenario, plus setup/export and 24-episode evaluation allowance | Training time | Total rental time | Estimate | With 25% time margin |
| --- | ---: | ---: | ---: | ---: |
| Every group mixed, optimistic sensitivity case | 1.77 h | 2.38 h | $4.53 | $5.66 |
| Observed 2/11 mixed pattern | 2.48 h | 3.09 h | $5.88 | $7.35 |
| Fixed-output alias counterfactual 4/11 | 2.04 h | 2.66 h | $5.05 | $6.32 |

These are extrapolations from only two updates and 11 adaptively stopped groups
under two policies, not reliable population estimates. Later policies, source
lengths, stock, I/O and setup may differ. Ten mixed groups within one question pass
are not guaranteed. The margin is a planning allowance, not a confidence interval.
The optimistic case is not a certified runtime lower bound.

Generation volume is a separate constraint: extrapolating the observed pattern
gives about 695 training requests / 2.86 million reserved tokens, below the original
1,024 / 8,388,608 phase caps. The 24-episode evaluation extrapolation is about 76
requests / 312,012 reserved tokens, versus its 288 / 2,359,296 caps. Reservations
are neither actual output-token counts nor extra hosted-API charges for local
inference. These averages do not guarantee phase completion within any cap.
Exact formulas, stage receipt hashes, phase limits and caveats:
[cost projection](../examples/verifiers-port/reader-reward-review-20261024/cost-projection.json).

**Do not launch the unchanged ten-update protocol.** All three scenarios exceed
its one-hour training and two-hour total rental limits; the observed-pattern case
also exceeds the proposed $5 before adding margin. The integration timing override
did not authorize different training/evaluation limits. No budget was raised and
no paid resource was created during this audit.

## Next decision and verification

First review/version the training-only wording additions and resolve or hold the
ambiguous representation question. Keep the historical reward and original traces
intact; any adopted keys need a new identity and validation before collection.
Keep evaluation references locked. Then prepare a pilot whose update count,
workflow timing, evaluation allowance and explicit budget agree. A larger run
needs a fresh price/capacity check and its own paid authorization. Reducing
checkpoint/process overhead could help, but it has not been implemented or timed.

The audit checked complete coverage of the 22 wording rows, training-only
membership, original rollout/context hashes, the copied-key validator, and full
rescoring of all 44 healthy untrainable episodes. Frozen files and the user's
pre-existing edits are preserved. **55 focused tests passed** (three expected CPU
checkpoint warnings); receipt integrity and the 29 frozen runtime files also
passed verification. See the
[verification receipt](../examples/verifiers-port/reader-reward-review-20261024/verification.json)
and [artifact hashes](../examples/verifiers-port/reader-reward-review-20261024/manifest.json).
No production code,
native trainer, optimizer, model prompt or active reward implementation changed.
The earlier full suite of 2,204 passes / 16 optional skips remains the last full
suite on unchanged production sources; it was not rerun for this artifact audit.
