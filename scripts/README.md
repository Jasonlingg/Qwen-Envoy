# Script entry points

Run these commands from the repository root. Start with the main [README](../README.md)
and [repository map](../docs/REPOSITORY_LAYOUT.md) for the project boundaries.
The list below is an entry-point guide, not an inventory of every dated
experiment script.

| Task | Entry point | Effect |
| --- | --- | --- |
| Recount the published case study | `python -I -S scripts/reproduce_case_study.py` | Offline hash and arithmetic verification; no model call. |
| Open the local Studio | `python scripts/launch_studio.py` | Serves saved traces locally; live inference is opt-in. |
| Export research history to Obsidian | `python scripts/export_research_vault.py` | Writes a one-way reading copy under `out/`. |
| Serve the optional research MCP | `python scripts/research_justification_mcp.py` | Exposes bounded paper and experiment-record tools to an MCP client. |

The training and evaluation scripts are experiment-specific. Before running
`train_sft.py`, `validate_reader_rl.py`, or a provider-backed runner, read its
linked protocol and budget in the [technical case study](../docs/TECHNICAL_CASE_STUDY.md)
or [experiment atlas](../docs/EXPERIMENT_ATLAS.md). Some scripts make paid model
calls or require a GPU; their presence here does not make them part of the
offline case-study check.

Dated scripts without a current entry in this table preserve specialized
analyses and source-bound experiments. Superseded entry points are documented
in [archived prototypes](../docs/ARCHIVED_PROTOTYPES.md); Git retains their
original source. Do not use historical host-local teacher replay as a substitute
for the current Docker execution boundary.
