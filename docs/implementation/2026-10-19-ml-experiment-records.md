# Experiment records — Task 2

Status: local Python foundation implemented. MCP wiring is Task 3; result
recording/comparison is Task 4. Neither task starts with this change.

The user begins with a question in an ordinary conversation. The host will
gradually populate a brief as it researches the setup and proposed change.
The storage layer has no built-in experiment recommendations, model calls,
network access, benchmark execution, or GPU provisioning.

## Local API

```python
from pathlib import Path
from src.product.ml_experiment_store import ExperimentStore

store = ExperimentStore(Path("~/.envoy/ml-experiments"))
draft = store.save_brief({"question": "Would a smaller model help my workload?"})
saved = store.get(draft["experiment_id"])
assert saved["status"] == "draft"
assert "hypothesis" in saved["missing_fields"]
```

`save_brief(brief, experiment_id, expected_revision)` appends a draft revision.
The host supplies the complete current brief, not a partial patch. A revision
conflict returns an error; the caller must retrieve the current version first.
`list(limit=20)` returns recent summaries and a truncation indicator (limit 1–100).

`freeze_plan(experiment_id, expected_revision)` requires a repeatable test:
setup/task/workload, hypothesis, proposed change, baseline and candidate,
evaluation samples, metrics, required criteria, procedure, applicability, and
resource limits. Exactly one condition is labeled the proposed change. Known
values marked held constant must match; other differences must be declared.
Unknown hardware/revisions retain explanations. A freeze records intent and
does not grant execution permission or establish independent preregistration.

A frozen plan cannot be edited through the API. Create a new experiment with
`supersedes_experiment_id` to change it. Repeating an identical freeze returns
the same plan digest. Eventual results will bind to that digest in Task 4.

## Validation and persistence

- Strict versioned Pydantic schemas reject extra fields, numeric coercion from
  strings/bools, nonfinite values, duplicate IDs, and dangling references.
- A one-question draft is valid. Missing planning fields remain explicit;
  unknown setup values carry reasons and never become zero measurements.
- Paper/artifact references are structurally validated declarations. This task
  does not verify source quotes or read artifact exports. Task 3 supplies that
  verification. Exact source provenance will still not prove semantic support.
- If both sample IDs and a sample-set hash are supplied, the hash must be the
  raw SHA-256 of UTF-8 JSON of the sorted IDs, using `ensure_ascii=False` and
  `separators=(",", ":")`. It identifies the set, not ordering or dataset bytes.
- Revision and frozen-plan objects are stored under their raw-byte SHA-256.
  A checksummed index binds the full revision chain and frozen plan. Readers
  check hashes, identities, schemas, and parent bindings on each operation.
- A process-local lock plus POSIX file lock protects concurrent updates.
  Objects are published before an atomic, fsynced index replacement; an
  interrupted operation can leave unreferenced objects but preserves the last
  committed record. There is no background object cleanup in this version.
- Files use mode 0600; newly created store roots use 0700. Store paths and
  records reject symlinks, traversal, and non-regular files. JSON has a 256 KiB
  bound, depth/element limits, and rejects duplicate keys. Drafts allow up to
  100 revisions. Corrupt experiments fail explicitly rather than being omitted
  silently from listings; healthy experiments remain individually accessible.

Freezing is enforced by this API. Hashes detect changed content; they do not
authenticate a local filesystem owner who deliberately rewrites all bindings.
Reviewer identity, historical timing, and caller-provided observations remain
declarations. This foundation establishes no model-quality or product-value result.

## Verification

Focused suite: 73 tests across `test_ml_experiment_models.py`,
`test_ml_experiment_store.py`, and `test_ml_experiment_storage_boundaries.py`.
It covers question-first drafts, restart persistence, concurrent revision
conflicts, interrupted save/freeze commits, immutable criteria, content/chain
tampering, sample identity, unsafe paths, symlinks/FIFOs, and bounded JSON.
All experiment values in these tests are explicitly synthetic.
