# Question-first ML experiment planning

Ask your normal engineering question, for example:

> Could a smaller open-source model handle this workload?
>
> Would fine-tuning address the failures in these runs?
>
> Which latency method is worth testing on my current setup?

The host assistant inspects the setup and selected results, asks for missing
details, researches methods with the existing paper tools or ordinary browsing,
and proposes a small test. It saves the brief behind that conversation. You do
not need to fill in a form. Hypotheses, methods, and thresholds are supplied per
experiment; the server has no built-in experiment recommendations.

This companion currently saves, retrieves, lists, and freezes plans. Result
recording/comparison is the separately approved Task 4. Research-to-decision
usefulness still needs the real development demonstration and user review in
Task 5. A connection check does not establish useful recommendations or Qwen
improvement.

## Start locally

Install the existing project dependencies, then run the stdio server:

```bash
pip install -e '.[mcp,vault]'
python scripts/ml_experiment_mcp.py \
  --store-root ~/.envoy/ml-experiments \
  --library-root ~/.envoy/arxiv-library \
  --artifact-root /absolute/path/to/selected-run-exports
```

Select a dedicated directory of sanitized JSON exports. The artifact directory
is optional: omit it if you only need question drafts or paper references.
Use the same paper-library root as your research MCP. Papers must already have
been opened there. This server never fetches papers, calls a model, starts a
GPU, runs code from a paper, or executes an experiment. The host assistant's own
usage remains separate. An optional Qwen reader belongs to the existing
[research MCP](ARXIV_RESEARCH_MCP.md) and requires its own available endpoint.

Register this companion alongside that research server in your local MCP
client. For Codex, an example project-local `.codex/config.toml` entry is:

```toml
[mcp_servers.ml-experiments]
command = "/absolute/path/to/python"
args = ["/absolute/path/to/rlm-explorer/scripts/ml_experiment_mcp.py", "--store-root", "/absolute/path/to/experiment-records", "--artifact-root", "/absolute/path/to/selected-run-exports", "--library-root", "/absolute/path/to/paper-cache"]
cwd = "/absolute/path/to/rlm-explorer"
startup_timeout_sec = 30
tool_timeout_sec = 30
enabled = true
```

Project configuration requires a trusted project. Restart the relevant client
or extension to load the new inventory; check it with `codex mcp get
ml-experiments --json`. See the [official Codex MCP configuration
instructions](https://learn.chatgpt.com/docs/extend/mcp).
Configuration alone is not proof that a native assistant completed research.

## Tools and conversation

| Tool | What it does |
| --- | --- |
| `save_experiment_brief` | Save a draft from just a question, or revise the complete current brief with an exact prior revision. Verify and copy attached evidence. |
| `get_experiment` | Return the saved brief, readable Markdown, evidence receipts, revision history, and frozen-plan identity. |
| `list_experiments` | List recent question/plan summaries; default 20, maximum 100, with truncation reported. |
| `freeze_experiment_plan` | Record the exact complete plan the user agreed to; preserve the criteria and evidence hashes. |

The host can first save `{"question": "Could a smaller model help here?"}`.
It fills in the baseline, candidate, hypothesis, evaluation sample, relevant
controlled conditions, metrics, procedure, budget, and stopping rule as the
investigation progresses. Missing details remain explicit. It explains the
proposal and freezes it only when you agree. This is workflow guidance, not
authenticated proof of human review. Freezing grants no execution or spending
permission. A changed frozen plan requires a new linked experiment.

## Evidence attachments

The brief's `references` list accepts these three kinds:

- `paper_span`: `reference_id`, a versioned `arxiv_id`, `start`, and `end`
  (1–1,200 characters). An optional `quote` or `document_sha256` must match.
  The server rebuilds the quote from cached text, checks cache integrity and
  identity, and records the document's raw-byte SHA-256 and extraction coverage.
- `local_artifact`: `reference_id`, `relative_path`, raw-byte `sha256`, and
  `json_pointer`. The server reads only inside the selected export directory,
  checks the exact file bytes, and copies the selected JSON value. Pointers use
  RFC 6901, e.g. `/metrics/answerable/passed`; `~1` escapes `/`, `~0` escapes `~`.
  Use a small object containing population, denominators, units, and reviewer
  kind where needed, rather than an unexplained score. A root pointer is `""`.
- `external_reference`: `reference_id`, HTTPS `url`, and `description`.
  Host-inspected model cards, implementation docs, and other URLs remain
  explicitly unverified references here; this server does not download them.

Example selected result attachment (digest must come from the actual file):

```json
{
  "kind": "local_artifact",
  "reference_id": "baseline_result",
  "relative_path": "baseline.json",
  "sha256": "<64 lowercase hex characters>",
  "json_pointer": "/answerable_summary"
}
```

Files are bounded to 2 MiB; selected values to 4,000 serialized JSON characters.
Large run files need a bounded export with declared parent hash and selection
recipe. An unchecked parent link remains a declaration. Symlinks, traversal,
special files, malformed/duplicate-key JSON, invalid pointers, and hash/quote
mismatches are rejected. No receipt is accepted from a caller as preverified.
Task 2 records with declared references need resaving with source verification
before a new freeze. Old frozen plans retain their original limitations.

Evidence receipts are stored in revision objects and copied into frozen plans.
They remain inspectable after the original source disappears. Paper checks
prove the quote came from the stored text; artifact checks prove a value came
from the selected export. Neither establishes truth, semantic support, faithful
parent-export derivation, fair benchmarking, or causal improvement. Paper
extraction may omit figures, table structure, or equations. Semantic support
remains `not_reviewed`. Human and model review remain separate.

## Limits and verification

Storage details are in the [record implementation
notes](implementation/2026-10-19-ml-experiment-records.md). This server uses that
256 KiB record bound, revision checks, file locking, and immutable content
objects. Receipts count toward the bound; oversized saves fail rather than
silently discarding evidence. Markdown display is capped at 20,000 characters
with an explicit truncation flag; the structured brief retains its full fields.

The project-local development registration uses ignored `out/ml-experiments`
and `out/ml-experiment-exports`, plus the existing `out/arxiv-mcp-live-library`.
It contains no credentials and preserves the `arxiv-research` entry.
The local diagnostic at `out/ml-experiment-mcp-local-20261019-v2.json` records the
configured server's real stdio initialization, four-tool inventory, and list
call. It performs no question answering or reader inference. Automated tests
also save a question, revise it with synthetic data and source receipts, freeze,
restart the subprocess, and retrieve the original plan/evidence.

Focused verification: 126 passing tests across the experiment modules and the
existing arXiv server/profile tests; Ruff and whitespace checks pass. Reproduce:

```bash
python -m pytest -q tests/test_ml_experiment*.py tests/test_arxiv_mcp_profiles.py tests/test_arxiv_research_mcp.py
```

The full relevant suite and independent release review remain Task 6.
