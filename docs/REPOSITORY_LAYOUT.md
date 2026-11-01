# Repository map

Main has two inspectable paths: QASPER Agent Studio and the optional experiment-research MCP. The full pre-cleanup tree is preserved at [`archive-full`](https://github.com/Jasonlingg/Qwen-Envoy/tree/archive-full). It contains dated experiments, one-off runners, and historical prototypes.

| Start here | Responsibility |
| --- | --- |
| [`../scripts/reproduce_case_study.py`](../scripts/reproduce_case_study.py) | Verify hashes and recompute saved paired counts offline. |
| [`../src/env/repl.py`](../src/env/repl.py), [`../src/env/tools.py`](../src/env/tools.py) | Docker action execution and paper tools. |
| [`../scripts/train_sft.py`](../scripts/train_sft.py) | Per-action v5 SFT preparation and prefix validation. |
| [`../src/training/`](../src/training/) | Experimental Prime-RL/Verifiers rollout and evaluation code. |
| [`../benchmarks/envoybench/`](../benchmarks/envoybench/) | Frozen split, scoring, and Studio API. |
| [`../scripts/launch_studio.py`](../scripts/launch_studio.py) | Serve saved Studio locally. |
| [`../scripts/research_justification_mcp.py`](../scripts/research_justification_mcp.py) | Optional stdio research MCP. |
| [`../src/product/experiment_atlas.py`](../src/product/experiment_atlas.py), [`EXPERIMENT_ATLAS.md`](EXPERIMENT_ATLAS.md) | Curated prior experiments and bounded source access. |
| [`../release/`](../release/) | Recorded traces, scoring, training cards, and public inspector assets. |

The [project guide](../PROJECT_GUIDE.md) has runnable commands. The passage-pilot files remain on main because the experimental Prime-RL continuation checks their frozen hashes. Standalone timing diagnostics and unpromoted v6 packaging scripts are available at `archive-full`. Frozen releases and replay hashes are unchanged by this organization work.
