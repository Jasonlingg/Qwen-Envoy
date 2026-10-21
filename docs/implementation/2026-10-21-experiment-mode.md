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
