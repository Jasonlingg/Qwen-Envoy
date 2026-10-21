# Experiment mode, task 1: connected local context

The user approved the first task only: connect the existing experiment records
and research tools so an experiment can be resumed. No inference, GPU launch,
optimization experiment, benchmark result, reward change or training is included.

Implementation:

1. Add append-only, source-checked research notes to the existing experiment
   store. Bind every note to the exact brief revision inspected by its author.
   Keep frozen plans and recorded results unchanged. Preserve old stores.
2. Add a local resume response containing the brief, linked research, latest
   supplied result and explicit unknowns. Summaries never become semantic grades
   or automatic next-experiment recommendations.
3. Provide one experiment-mode MCP entry point with the existing record tools
   and five model-free paper tools. The host researches, then explicitly records
   its findings against the active experiment ID. Qwen remains an optional
   separate research worker; this task does not invoke it or silently import its
   reports. No process-global 'active experiment' may mix concurrent clients.

Verify real MCP start/save/source-inspect/note/restart/resume flows with fixture
sources, frozen-plan/result preservation, stale revision and retry handling,
concurrent notes, corruption and unsafe source rejection. These are component
checks, not evidence of research usefulness or model improvement.

Later, separately approved work: a real experiment walkthrough, then a bounded
paired optimization pilot with measured correctness and performance. No winner
or speedup is assumed in advance.

## Implemented behavior

The [experiment-mode guide](../EXPERIMENT_MODE.md) documents the single stdio
entry point and tool flow. Existing local record and research entry points keep
their original inventories. Experiment mode exposes those five record tools,
the five source tools, `record_experiment_research` and `resume_experiment`.
Source operations delegate through the public MCP SDK to the existing source
implementation. The experiment-discovery fallback remains bounded and explicit.

Research is an append-only note, not an edit to a frozen plan. Every note binds
to an existing experiment and the exact inspected brief revision/digest. Source
receipts are generated internally using the existing resolver. The notes remain
available after draft edits, result recording, restart and removal of original
sources. Exact retries do not re-read missing sources. Changed content under the
same note ID, stale new notes and cross-experiment bindings are rejected.

Resumption returns the latest note and supplied result in a bounded text recap,
plus IDs, counts, current revision and frozen-plan hash. Long sections explicitly
report truncation. Full and historical records remain available through
`get_experiment`. Older notes flag their original brief revision. No global
active-experiment state, automatic Qwen report import, paid inference, semantic
grader or execution loop was added.

## Focused verification

The 112-test focused run passed on October 21, 2026:

```sh
python -m pytest -q \
  tests/test_experiment_mode_mcp.py tests/test_ml_experiment_mcp.py \
  tests/test_ml_experiment_research_notes.py tests/test_ml_experiment_store.py \
  tests/test_ml_experiment_outcomes.py tests/test_ml_experiment_storage_boundaries.py \
  tests/test_ml_experiment_evidence.py tests/test_arxiv_research_mcp.py \
  tests/test_experiment_discovery.py
```

The integration flow uses real MCP clients with fixture paper data and explicitly
synthetic result measurements. It creates a question, inspects a passage, records
research, revises/freezes a plan, records supplied results, deletes the source,
restarts the store/client and resumes. A separate real stdio process restarts
from another working directory without an inference endpoint. Other checks
exercise concurrent notes, corrupt/hash-valid misbound records, stale writes,
safe retries, unsafe source links, rejected review claims and interrupted writes.
These are component checks, not native-assistant usability or live model tests.

Ruff and `git diff --check` passed. All 11 protected-file hashes, including the
active MuSiQue reward and the seven preexisting uncommitted files, matched their
pre-task values. No reader/judge dataset or historical evaluation was changed.

The final full suite passed **1850 tests, zero failures, zero skips**, in 114.61
seconds, with three existing SWIG deprecation warnings. It ran offline with the
existing pinned Docker image for tests that require real Docker:

```sh
PATH=/Users/jasonling/.local/bin:$PATH \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
ENVOY_ASSIGNED_TEST_DOCKER_IMAGE=sha256:79bd8bf84bd4b7f2a32a0596cb1fd90164716c1cfca879f78829411193646382 \
python -m pytest -q -rs
```

The [validation receipt](../../examples/experiment-mode/validation.json) binds
implementation/protected-file hashes and the local test log hash. Model requests,
paid model calls, GPU provisioning and actual optimization runs were all zero.
The next real walkthrough and benchmark pilot remain unperformed.
