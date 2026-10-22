# Matched V5 live smoke with hybrid-selected QASPER passages

User explicitly said “ok go test it” after being told that hybrid retrieval
improved saved coverage but Qwen had not yet read those passages. This authorizes
one new bounded live V5 attempt; it does not reset any earlier consumed allowance.

Hypothesis: replacing the two BM25 windows with frozen plain-hybrid windows
improves useful V5 answers on the same exposed three-question regression smoke.
Expected signal: supported humor cues rather than inability, specific LSTM/attention
facts rather than generic NMT, and a question-specific scoped gap on the original
unanswerable racial/ethnic-bias question. This is not a held-out/generalization test.

Decision rule, fixed before calls: both answerable cases must produce complete,
source-supported requested content, and the unanswerable case must return a useful
specific scoped gap without an invented capability. For humor, at least one complete
original reference cue list is required (four latent categories or three stylistic
features); report coverage of both lists separately. For translation, require LSTMs
and the attention mechanism, beyond generic NMT. Gap fields must identify what
is missing for the racial/ethnic question and propose a relevant concrete query;
prompt placeholders fail quality even if structurally valid. Require valid packets
and no Python execution errors. Report partial gains and every action separately.
These judgments are implementing-assistant review, not independent human grading.
No model judgment is inserted into a human-review field or used as a training reward.

Only the controller passage-scope metadata changes. Use exact previously saved
plain-hybrid windows, source SHA and query from the frozen offline comparison;
no new retrieval tuning, reranking, labels or gold facts in model input. Preserve
question order/IDs, entire source snapshots, Qwen V5 checkpoint/adapter, effective
system prompt, decoding, Docker image, four-step/request limit and harness bytes.
First requests must byte-match the old payloads after replacing only the passage
scope JSON. The same two disjoint 1600-character windows total 3200 characters.

Historical BM25 reference: examples/qasper-passage-v2 (nine saved requests,
27,936 tokens). Do not rerun or relabel that baseline. This is a historical matched
reference, not a fresh randomized/contemporaneous control; sampling/service
variation is not isolated and no causal or general fine-tuning gain is claimed.
Remote seed is not guaranteed by the queue adapter. No new training, RL, reward
change, base/Nemotron call, new questions or reserved research-reader pack access.
The retriever remains an experimental opt-in; product defaults stay unchanged.

Bounds: one existing A5000 endpoint activation, max 900 seconds including startup,
watchdog armed before activation, shared max 12 requests/98304 reserved tokens,
max 4 requests/32768 reserved tokens per case, no inference retries or replacement
cases. Fresh serverless rate must be <=$0.69/hour. Local allowance $0.50; the
15-minute compute estimate is <=$0.1725, NOT a provider invoice cap. No Nebius call.
Use an exclusive new global marker. Preserve all old artifacts and paid markers.

Implementation uses the existing real in-process evaluation MCP tool, Docker-only
investigator, packet verifier and bounded queue/lifecycle runner. Freeze new
implementation/source/baseline hashes and run scripted MCP/Docker preparation,
focused tests and the full relevant suite before activation. Save raw model output,
cleaned actions, source inspections, verified quotes, gaps, usage and failures.
Finish by parking workers 0/0, verifying the full original endpoint configuration,
zero workers/pending jobs and absence of owned Docker containers through fresh reads.


## Outcome

Implementation `7ba4e52`; exact matched first inputs frozen in `6ff9404` before
activation. [Saved live results](../../examples/qasper-hybrid-v5/README.md):
nine V5 requests / 28,270 tokens; zero execution/action/submission/gap errors,
six actual Docker inspections and three provenance-valid terminal packets.
Translation improved from generic NMT to supported LSTM/attention/dropout facts.
Humor still denied cues despite all seven reference terms being present in the
actual final model request and returned quotes. Bias returned question-specific
prose inability as evidence_found, with no terminal gap or follow-up query.
Complete answerable cases 0/2 → 1/2; useful gaps 0/1 → 0/1; overall gate not met.
Review is implementing-assistant, not independent human or official QASPER grading.
No model weight, prompt or harness was changed. No general model gain is claimed.

Fresh 16:35:47 UTC reads verified full configuration restoration, workers 0/0,
zero workers/pending jobs; six owned Docker containers absent. 804 prior hashes
and all old consumed markers unchanged. Focused 64 passed; full 2012 passed in
98.49s, no skips. The new hybrid V5 marker is consumed. No additional paid calls,
reserved-set access, model training/RL or reward change is included.
