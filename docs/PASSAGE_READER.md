# Optional Qwen passage reader in experiment mode

Use this when a larger assistant has selected a relevant paper and candidate
passages, and wants a small worker to extract requested facts and conditions.
The host retains discovery, semantic review and the final experiment decision.
This mode does not make Qwen an autonomous literature researcher.

The tool is opt-in. Ordinary experiment mode still starts without a model,
tokenizer or Docker and exposes the same twelve tools. With a reader configured,
it also exposes `read_experiment_passages`.

## Configure an already served model

Use the repository Python environment, a locally cached matching Qwen3-8B
tokenizer, Docker and the pinned sandbox image. For an OpenAI-compatible endpoint:

```sh
python scripts/experiment_mode_mcp.py \
  --store-root ~/.envoy/ml-experiments \
  --library-root ~/.envoy/arxiv-library \
  --qwen-endpoint https://YOUR-SERVED-MODEL/v1 \
  --qwen-model envoy \
  --qwen-checkpoint YOUR-PINNED-CHECKPOINT-AND-REVISION \
  --sandbox-image sha256:79bd8bf84bd4b7f2a32a0596cb1fd90164716c1cfca879f78829411193646382
```

Use `ENVOY_MODEL_API_KEY` in the host environment if required; keep secrets out
of arguments. This command neither provisions nor activates a GPU. A Runpod
queue URL is not an OpenAI-compatible endpoint. The development runner below
injects the existing queue adapter explicitly; it is not an always-on service.

Startup checks Docker and resolves tokenizer files locally without downloads.
It does not make model requests. A declared checkpoint/model alias is not
independent weight attestation for an arbitrary external endpoint.

## Conversation

1. Save or resume the experiment. The host chooses the current revision.
2. Use source tools to open a paper and choose candidate passage coordinates.
3. Explicitly invoke `read_experiment_passages` with the question, experiment
   ID, revision, new note ID, cached paper version and windows.
4. Inspect the returned candidate facts and source quotes. Exact quotations
   establish provenance; semantic support, conditions and usefulness need review.
5. Resume later to retrieve the Qwen-origin note, or retrieve the full note/report
   when the bounded recap omits detail. Origin is recorded by this server, not a
   cryptographic model attestation.

```json
{
  "experiment_id": "experiment_<actual generated ID>",
  "expected_revision": 1,
  "note_id": "reader_attempt_1",
  "question": "Which measurement and hardware conditions are stated here?",
  "arxiv_id": "<already opened paper version>",
  "windows": [{"start": 1000, "end": 1800}]
}
```

The coordinates above are placeholders. Supply offsets from the actual cached
paper. Source text is untrusted data. The call sends only the supplied question,
paper metadata and selected source text; it does not implicitly forward private
experiment setup. Invoking the tool authorizes that explicit external reading.

## Boundaries

One invocation permits four model requests/actions, each reserving at most 7168
input and 1024 output tokens, for 32,768 reserved tokens. Format corrections use
the same budget. There is no hidden reasoner, inference resubmission or provider
dollar cap. An uncertain transport outcome stops further dispatch for that call.

Select one to three ordered, disjoint windows, each at most 2000 characters and
at most 4000 characters in total. Metadata does not count as inspection. Qwen's
displayed Python is parsed conservatively before network-disabled Docker runs
it. Only literal `passage(doc_id, start, length)` reads within those windows are
allowed. The controller supplies ready-to-use read calls with explicit lengths.
Bounded constant integer addition/subtraction in passage arguments is parsed as
AST data and normalized into literals before Docker execution. No `eval`, host
Python execution, calls, exponentiation or unbounded arithmetic are allowed.
Raw and normalized actions are retained in the trace. Other document tools,
arbitrary Python and out-of-scope reads are
rejected before execution. Supported submissions can cite only inspected ranges.

For a gap, the model must inspect two distinct reads, report what remains
missing, and abstain. The gap is limited to selected passages; it does not prove
absence from the paper or all literature. A proposed next query is not executed.

Successful and incomplete reader reports are saved under the configured library's
`passage-reader-reports` directory. Verified evidence is attached as an unreviewed
research note. A result is retained if the brief changes before attachment; the
host can reattach it through the ordinary research-note tool without rerunning Qwen.
An interrupted invocation remains consumed rather than being retried automatically.
The same note ID and request return the saved report without new model calls;
changed input under that ID is rejected. New note IDs are explicit new attempts.

## Development attempt

The [three-case record](../examples/passage-reader-three-case/README.md) uses
familiar QLoRA, AWQ and PagedAttention cases with deliberately selected relevant
passages. It is a capability probe, not a held-out comparison or an RL reward.
The one-attempt runner has a durable aggregate twelve-request ceiling, native
startup verification and a 900-second GPU watchdog. Running its paid phase needs
a new plan-bound allowance; existing consumed attempt markers cannot be reset.

Focused local checks exercise source bounds, unread citation rejection, durable
attempts, stale revisions, MCP save/resume, prompt identity and actual Docker with
scripted model replies. A scripted success is never a live-Qwen quality result.

The first live attempt returned one supported AWQ answer and two incomplete cases.
The literal-only controller rejected valid end-minus-start read lengths, wasting
limited actions; one replacement also exceeded its source window. The arithmetic
normalization and explicit read-call metadata were added afterward.

The separately authorized [live rerun](../examples/passage-reader-rerun-1/README.md)
then returned supported QLoRA and AWQ answers under implementing-assistant source
review. All five passage reads succeeded with literal lengths. The missing-target
case read both windows, attempted abstention before reporting its gap, and used
its last action to record the gap after feedback. It remained incomplete and
returned no substantive partial findings. No target latency was invented.
The overall three-case rule still failed. Nine requests used 26,302 reported
tokens; full configuration restoration and zero workers/jobs were verified.
The same exposed inputs and unchanged model do not establish generalization,
fine-tuning improvement or an advantage over ordinary assistant reading. Live
literal reads did not exercise arithmetic normalization; both it and the new
read examples changed between attempts, so their effects are not isolated.

A separately authorized [base comparison](../examples/passage-reader-base/README.md)
used the same repaired prompt, source windows and four-action limit, changing
only the model alias in first requests. Base Qwen3-8B without V5 returned the
QLoRA facts correctly, but never inspected AWQ because mixed read/submission
outputs were interpreted as premature abstentions. Its gap case also remained
incomplete. V5 completed two answerable cases versus base's one in these familiar
examples, with fewer calls (9 versus 12). This suggests better protocol compliance
here, without establishing a general fine-tuning, factual-reading or cost gain.
Independent human review remains blank. Base and V5 still fail the gap procedure;
training should target reviewed, repeatable failures after interface choices
are stable and must be tested on separate questions.
