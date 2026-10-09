# Two-minute Studio demo script

**Recording status:** draft. The current build has four saved runs, and an
NVIDIA Nemotron-3-Ultra run completed through Nebius in the local Studio.
Its answer had citation errors and its trace had execution failures. A
live test build for judges and the public video are still pending. The saved
viewer is [public](https://jasonlingg.github.io/Qwen-Envoy/); it supports the
first five scenes below without credentials. The complete video must be public,
under three minutes, and reflect what the shipped build does.

| Time | On screen | Suggested narration |
| --- | --- | --- |
| 0:00–0:15 | Open Studio **Runs**. | “A final answer does not show whether a research agent found the paper, ran valid code, or cited evidence. EnvoyBench keeps that path inspectable.” |
| 0:15–0:35 | Open the September 30 **base** run from **Runs**. | “The base and v5 agents received the same frozen paper questions and Python tools. Their grades are provisional: 5 of 40 passes for base and 15 for v5, but both passed only 3 of 20 answerable questions. The observed difference is mainly abstention and fewer execution failures.” |
| 0:35–0:55 | Choose a question and open **Trace**. Show a Python action, its tool observation, the answer, and a submitted source span. | “Here is one saved agent trace. The source span is checkable; whether it supports the claim still needs review.” |
| 0:55–1:05 | Return to **Runs**, select September 30 **v5**, then open the same question and its trace. | “Switching runs lets me inspect how the other agent handled this same question.” |
| 1:05–1:20 | Return to **Runs** and open an October 3 development-smoke run. Expand one turn's **Token likelihood** panel. | “The two-question smoke is unscored. Token likelihoods, including those for generated code, are debugging telemetry, not confidence in answer correctness.” |
| 1:20–1:50 | Show or rerun the local live case `run_3666f88fff934274b76c3c1e6b4e4017`: NVIDIA `nvidia/Nemotron-3-Ultra-550b-a55b` on Nebius Token Factory. Show the submitted answer, citation spans, two Python syntax errors, and usage. | “The live model completed a ten-step investigation. It named the six encoder and decoder layers, but one cited span points to sub-layer text and another truncates the decoder claim. This is an inspectable result, not a supported-answer success.” |
| 1:50–2:00 | Show source link and run identity, then finish. | “The point is to make agent behavior and its limits reviewable, not to turn a provisional score into a leaderboard.” |

Before filming, choose a public-paper question whose source passage and both
saved traces have been manually checked for the screen recording. The static
export includes the complete September 30 traces. Avoid showing API keys,
private vault content, local connection configuration, or an unsupported
quality claim. Recheck the live run and usage in the final judge-accessible
build before recording; the local result alone does not establish that the
public build works.
