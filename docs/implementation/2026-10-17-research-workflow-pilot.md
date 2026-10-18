# Direct tools versus Qwen readers: paired development pilot

Status: prepared, not executed. Preparation makes no live model calls and starts
no infrastructure. This tests whether delegation earns its complexity in one
narrow workflow. It does not test or establish fine-tuning improvement.

## Hypothesis and scope

Hypothesis: the same host assistant can delegate paper investigation to trained
Qwen readers while preserving supported-answer quality and reducing measured
total cost or warm latency. Expected signal: comparable human-reviewed answers
with fewer expensive host operations and an overall measured efficiency gain.

The five questions are agent-authored development prompts. Their sources and
question concepts were inspected during the October 16 source smoke and pilot
preparation. They are not held out, independently validated gold labels, or a
representative benchmark. Do not train on them or present their results as a
general research-agent quality estimate.

Question data: `data/eval/research-workflow-pilot-v1.json`. The model receives
only each question's `id`, `question`, and `arxiv_ids`, plus the common task
instructions. Top-level `review_strata` is evaluator-only metadata. Never send
it, this protocol's reviewer notes, or review assignments to either model.
Frozen papers are CodeAct `2402.01030v4`, C2C `2510.03215v2`, and LatentMAS
`2511.20639v4`. Check the recorded extracted-text hashes before running; use the
existing cache at `out/arxiv-mcp-live-library`. A changed text is a new corpus,
not an invisible refresh. HTML extraction can omit tables, figures, or equations.

## Paired procedure

Run two fresh conversations per question, keeping the host model, common prompt,
answer format, frozen sources, decoding, and host limits equal:

- **Direct:** host uses `find_evidence` and `get_passage` on pre-opened papers.
- **Delegated:** host uses `investigate` with Qwen, then synthesizes. It may use
  the same source tools to check or repair results; record every such fallback.

Do not add discovery or outside browsing. Disable the optional reasoner. Use
harness V2 and the pinned Envoy v5 checkpoint, not an unspecified hosted Qwen.
Keep executable Python in Docker. Reader failures preserve other evidence and
remain visible; never silently rerun or omit a failed question.

Alternate arm order across the five questions and record the schedule before
inference. Do not reuse conversation history, answers, or evidence packets from
the other arm. A common pre-opened immutable paper cache is allowed. Record
first-start loading separately from warm investigation time. Do not describe a
warm comparison as end-to-end first-use latency.

## Execution limits

These are proposed pilot limits, not a statement that runtime enforcement exists:

| Scope | Limit |
| --- | --- |
| Host | 8 turns per question per arm |
| Qwen | 4 action steps per paper; constructor minimum is 2 |
| Model response | 1,024 reserved output tokens per request |
| Whole paired pilot | 96 model requests |
| Whole paired pilot | 300,000 locally counted input tokens |
| Whole paired pilot | 98,304 reserved output tokens |
| Reader admission | At most 2 active readers across pilot calls |
| Optional reasoner | Off |

Count host and worker requests, format re-asks, failed attempts, and requested
retries against the same aggregate limits. Reserve request/input/output budgets
before dispatch, including concurrent dispatches. Record tokenizer identity and
how input counts are computed. These are local accounting caps, not provider
token counts, a reliable dollar cap, or permission to start paid resources.
The aggregate cap may stop the pilot before every per-question maximum is used;
that is an incomplete pilot, not authorization to raise the cap.

The two-reader setting is a pilot gate. Application-wide admission control,
context budgeting, GPU startup, idle shutdown, and an independent cleanup
watchdog remain separate new runtime work. An existing per-investigation worker
limit does not enforce a process-wide limit across simultaneous investigations.

## Review and decision rule

Review all 10 answers with randomized identities concealed until review ends.
Human review remains blank until a human performs it; model assistance cannot
be relabeled independent human review. An agent may draft claim segmentation,
but the reviewer verifies it includes every factual claim and qualification.

Reviewer-only strata and scope:

| Question | Stratum | Scope to inspect |
| --- | --- | --- |
| P01 | Answerable | Python action representation, feedback, reported limitations; no zero-error guarantee |
| P02 | Answerable | Cache fusion, learned components, stated limitation; author report versus advice |
| P03 | Answerable | Final-answering-agent output-token statistic and exclusions; no whole-system cost extrapolation |
| P04 | Answerable | Evidence from both papers, communication and compatibility/training assumptions; no cross-paper score ranking |
| P05 | Insufficient deployment evidence | No unmeasured 24 GB, cost, or latency guarantee; concrete missing profiling |

For each answer record reviewer identity/date, blind ID, question ID, atomic
claims, materiality, cited evidence IDs/offsets, exact-provenance result, semantic
support, scope/qualification correctness, completeness, verdict, and notes.
Semantic support choices are `supported`, `partly_supported`, `unsupported`,
`contradicted`, and `not_assessable`. Provenance is a separate field.

Answer verdicts: `pass` answers the question with supported major claims and
honest scope; `partial` is useful but misses an important point or qualification;
`fail` is empty, wrong, misleading, or materially unsupported. An answer with
fewer claims is not automatically better; score completeness. Inspect the full
source context when needed. Report four answerable cases separately from the
single deployment-abstention case, then report paired wins/losses/ties.

Continue beyond this pilot only if delegated answers achieve **at least 4/5
human passes, including at least 3/4 answerable passes**, with **no paired quality
regression and no major unsupported deployment claim**, **and** either total
measured cost or median warm latency improves by **at least 20%** versus direct.
Paired quality uses `pass > partial > fail`; a lower delegated verdict on any
question fails the no-regression condition. Unreviewed answers, incomplete pairs,
or missing complete costs make the primary decision inconclusive. Report any
available latency signal separately; do not infer missing cost savings.

Cost includes both host and worker inference, GPU loading/active/idle time,
cleanup interval, and any storage charges attributable to the session. Record
actual charges and allocation method, never hardcode assumed dollar rates.
Do not double-count overlapping reader GPU time. Show batch cold-start cost
separately and state any amortization. Unknown host/subscription usage cost is
unknown, not zero. Keep unresolved failures and shutdown billing visible.

## Reproducibility record

Save Git commit and dirty-tree hash; question-file and corpus hashes; versioned
paper text hashes/parser/coverage; host model identifier and revision availability;
all prompts and hashes; Qwen base and adapter revisions/hashes and whether weights
were attested; effective harness prompt hashes; Docker image digest; decoding,
seeds or unknown; GPU type/count/precision; serving software/settings; limits,
tokenizers, concurrency and fallback policy; ordered requests, tools, errors,
usage with missing values explicit; startup/warm/end-to-end timing; billing and
cleanup records; review seed, concealed key, reviewer identity and dates.
Reward/training version is `not applicable`; no training is performed.

Existing `src/eval/research_review.py` can supply exact-span materialization and
blind IDs. Its trained-model-improvement decision wording and answer-only review
schema must not be reused unchanged. This comparison evaluates direct versus
delegated workflows; a separate matched base-versus-trained evaluation is needed
for any fine-tuning claim. Five paired questions are an engineering pilot, not
statistical evidence of general superiority or the historical product gates.
