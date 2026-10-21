# Completed personal experiment workflow

The host-led local workflow is complete: question → research → recorded plan →
measurements → decision → resume. This example uses real saved observations and
actual MCP stdio calls. It replays previously collected data; it is not a new
live paper investigation or model-quality evaluation.

The project-derived example question is: **Can two Qwen v5 reader requests
share one A5000, and should we claim a latency win?** This question belongs to
the example. The general MCP accepts the user's own question and setup.

## What was connected

- The [earlier live paper walkthrough](../experiment-mode-walkthrough/README.md)
  discovered/opened PagedAttention and inspected source passages. This replay
  imports a selected saved finding, quotation and versioned source URL. The
  import verifies artifact bytes; it does not perform a new paper-span check.
- The [automatic live serving run](../two-reader-profile/automatic-live-1/README.md)
  completed ten Qwen calls and nine telemetry jobs. Its original model identity,
  raw timings, output signatures and uncertainty assessment remain unchanged.
- A new, explicitly retrospective analysis record links that research to the
  measured outcomes. Freezing this reconstruction does not imply preregistration
  or authorize another run. Replay resource ceilings are zero model/GPU calls.
- Fifteen real MCP calls across two server processes attach research, save and
  freeze the analysis, record measurements, retain the host decision, restart,
  resume and retrieve history. Identical result/note retries are safe. Two
  deliberate attempts to overwrite history return tool errors as expected.

The completed record is `experiment_00bb7ec56e354c9493e5a98c87b688f9`.
It contains two research notes, one immutable result, and a revision-2 frozen
analysis. The original walkthrough record stays unchanged.

## Reproduce locally

Use the repository Python environment with the `mcp` and `vault` extras:

```sh
python scripts/demo_experiment_mode.py --output out/my-experiment-demo
```

Choose a new directory. The command verifies original source hashes, derives
the small selected export, exercises MCP, then writes `RESUME.md`, `EXPERIMENT.md`,
`complete-record.json`, `report.json`, local records and all tool responses.
It needs no model, tokenizer, Docker daemon, network, credentials or GPU.
New record IDs/timestamps will differ; source bindings and assessment should agree.
Changed saved inputs or an existing destination fail without overwriting records.

To continue the new record in an MCP host, register this command using absolute
paths and the same Python environment:

```sh
python scripts/experiment_mode_mcp.py \
  --store-root out/my-experiment-demo/records \
  --library-root out/my-experiment-demo/library \
  --artifact-root out/my-experiment-demo/exports
```

Call `list_experiments`, then `resume_experiment` with the ID printed by the
demo. `get_experiment` retrieves full records; use `research_note_id:
"prior_research_context"` to inspect the older paper research. A fresh host
session must load the configured tools; this chat's tool inventory is unchanged.

For normal use, [configure personal store/cache roots](../../docs/EXPERIMENT_MODE.md)
and start with your own question. Source tools retrieve papers only when invoked;
local save/resume operations make no model or network calls.

## Result and decision

The record's mechanical assessment is **inconclusive**:

| Criterion | Recorded outcome |
| --- | --- |
| All four timed calls per dispatch mode completed | Met |
| Two requests observed running in the same engine | Met in both concurrent rounds |
| Resolvable pair-time nonregression | Unknown; original uncertainty gate failed |

Raw times remain available: round 1 was 3.423 s serial versus 3.355 s concurrent;
round 2 was 3.829 versus 2.850. Round 1's 0.068 s difference is inside the original
0.4-second margin. We retain these observations while withholding a comparison
value eligible for a latency verdict. The receipt proves that the unknown and
its explanation were recorded faithfully; it does not make the latency known.

The host decision is to retain a shared serving engine with up to two requests,
avoid claiming a speedup, and avoid choosing more optimizations from these short
outputs. Representative reader usefulness needs its own evidence before more
training. That recommendation is a host interpretation, not a human-approved
adoption decision. See the [small claim review](claim-review.json).

## Inspect and verify

- [Resume](RESUME.md): what a later conversation recovers.
- [Readable experiment](EXPERIMENT.md) and [full record](complete-record.json).
- [MCP report](report.json): all calls, response hashes, restart/retry checks.
- [Selected observations](exports/serving-observations.json): parent hashes,
  derivation, raw timings, measured completion/overlap and unknown latency.
- [Verification](validation.json): source/code/artifact hashes and test results.

Original research, serving attempts, consumed allowances and protected files
remain unchanged. This task made no new source requests, model calls, GPU
activations or training changes. Ordinary host-assistant usage is not metered
by this script and is not reported as free.

## Scope of completion

The personal MCP can now carry a question through researched context, a frozen
plan, supplied results and a later resumed decision. The host interprets evidence;
the MCP validates provenance, revisions and arithmetic. It does not execute
experiments or decide correctness from quotation matches.

Qwen paper research remains optional and separately configured. Its last full
live research smoke produced no verified evidence; this successful serving test
and record replay do not replace that negative result. Research usefulness,
independent human review, productivity advantage over ordinary web research,
fine-tuning gains and any RL benefit remain unestablished. No paid retry,
additional training, public deployment or hackathon submission occurred here.
