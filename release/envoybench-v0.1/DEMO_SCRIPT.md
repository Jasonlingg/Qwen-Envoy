# Two-minute QASPER Agent Studio demo script

**Recording status:** draft. The historical release has four saved run rows,
and a separate single-paper NVIDIA Nemotron-3-Ultra investigation completed
through Nebius in the local Studio. Its answer had citation errors and its
trace had execution failures. The new 40-question Nebius attempt is incomplete: the cost guard stopped it
after 38 finished episodes, one interrupted episode, and one unattempted question
at $1.947862 estimated usage. No full-run score or completed-run row is claimed. A
live test build for judges and the public video are still pending. The saved
viewer is [public](https://jasonlingg.github.io/Qwen-Envoy/); it supports the
first five scenes below without credentials. The complete video must be public,
under three minutes, and reflect what the shipped build does.

| Time | On screen | Suggested narration |
| --- | --- | --- |
| 0:00–0:15 | Open **QASPER Agent Studio → Runs**; show the QASPER attribution. | “This independent Qwen Envoy study adapts 40 questions from Ai2's QASPER. Studio shows how a paper-reading agent reached its answer, including its code and evidence.” |
| 0:15–0:35 | Show **Official QASPER Answer F1 (0–100)** and the separate AI pass counts; open the September 30 **base** run. | “Overall answer F1 rose from 19.9 to 30.7, but on the 20 answerable questions it fell from 29.8 to 21.4. The gain comes from the unanswerable group.” |
| 0:35–0:55 | Choose a question and open **Trace**. Show a Python action, its tool observation, the answer, and a submitted source span. | “Here is one saved agent trace. The source span is checkable; whether it supports the claim still needs review.” |
| 0:55–1:05 | Return to **Runs**, select September 30 **v5**, then open the same question and its trace. | “Switching runs lets me inspect how the other agent handled this same question.” |
| 1:05–1:20 | Return to **Runs** and open an October 3 development-smoke run. Expand one turn's **Token likelihood** panel. | “The two-question smoke is unscored. Token likelihoods, including those for generated code, are debugging telemetry, not confidence in answer correctness.” |
| 1:20–1:50 | Show or rerun the local live case `run_3666f88fff934274b76c3c1e6b4e4017`: NVIDIA `nvidia/Nemotron-3-Ultra-550b-a55b` on Nebius Token Factory. Show the submitted answer, citation spans, two Python syntax errors, and usage. | “The live model completed a ten-step investigation. It named the six encoder and decoder layers, but one cited span points to sub-layer text and another truncates the decoder claim. This is an inspectable result, not a supported-answer success.” |
| 1:50–2:00 | Show source link and run identity, then finish. | “QASPER supplies the questions and official answer metric. Studio makes the agent's actions and evidence reviewable. This is a selected agent study, not a full QASPER leaderboard result.” |

Use the final QASPER-branded export for recording. The official Answer F1
column is normalized token overlap against all original answer annotations
for the selected questions. It is not a citation-support grade. Keep the
provisional AI support judgments and unscored development smoke distinct.

If the completed supplementary Nebius run replaces the single-paper live
scene, show its own run identity and saved trace. It uses the same frozen
questions and paper tools, but has a different provider, serving context,
model revision policy, and runtime. It is not a controlled extension of the
Qwen base-versus-v5 comparison. Its support verdicts remain unreviewed even
when official Answer F1 is available. Do not reuse the historical live case's
timing or usage for the new run.

Source credit: [QASPER, Dasigi et al. (2021)](https://aclanthology.org/2021.naacl-main.365/),
CC BY 4.0. The [pinned official evaluator](../../benchmarks/envoybench/vendor/qasper/README.md)
and [original-reference score artifact](../qasper-agent-study/qwen-official-score.json)
document the scoring. QASPER Agent Studio is independent of Ai2.

Before filming, choose a public-paper question whose source passage and both
saved traces have been manually checked for the screen recording. The static
export includes the complete September 30 traces. Avoid showing API keys,
private vault content, local connection configuration, or an unsupported
quality claim. Recheck the live run and usage in the final judge-accessible
build before recording; the local result alone does not establish that the
public build works.
