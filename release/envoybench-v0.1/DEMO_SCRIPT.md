# Two-minute QASPER Agent Studio demo script

**Recording status:** draft. Studio has four original Qwen run rows and a fifth,
**Incomplete** NVIDIA Nemotron-3-Ultra/Nebius row. The cost guard stopped that
40-question attempt after 38 finished episodes; question 39 was interrupted
and question 40 was unattempted. Its catalog-rate estimate is $1.947862, and
its quality scores are blank. A separate historical single-paper Nebius
investigation is documented in the live-validation record. A
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
| 1:20–1:50 | Select **Nemotron Ultra Nebius** in Runs. Show **Incomplete · budget stop**, **38/40 finished**, blank scores, and the recorded usage. Open **Unfinished** in Questions and inspect question 39's 11-action trace and question 40's no-attempt state. | “This NVIDIA model ran through Nebius on the same questions, but the $2 local cost guard interrupted question 39 and question 40 never ran. We can inspect the saved actions; this attempt has no comparable 40-question score.” |
| 1:50–2:00 | Show source link and run identity, then finish. | “QASPER supplies the questions and official answer metric. Studio makes the agent's actions and evidence reviewable. This is a selected agent study, not a full QASPER leaderboard result.” |

Use the final QASPER-branded export for recording. The official Answer F1
column is normalized token overlap against all original answer annotations
for the selected questions. It is not a citation-support grade. Keep the
provisional AI support judgments and unscored development smoke distinct.

The partial Nemotron run uses the same frozen questions and paper tools, but
has a different provider, serving context, model revision policy, and runtime.
It is not a controlled extension of the Qwen base-versus-v5 comparison. Do not
calculate a 40-question F1 from its finished subset or use the historical
single-paper case's timing and usage for this run.

Source credit: [QASPER, Dasigi et al. (2021)](https://aclanthology.org/2021.naacl-main.365/),
CC BY 4.0. The [pinned official evaluator](../../benchmarks/envoybench/vendor/qasper/README.md)
and [original-reference score artifact](../qasper-agent-study/qwen-official-score.json)
document the scoring. QASPER Agent Studio is independent of Ai2.

Before filming, choose a public-paper question whose source passage and both
Qwen traces have been manually checked for the screen recording. The static
export includes the complete September 30 traces and the saved Nemotron partial
traces. Avoid showing API keys,
private vault content, local connection configuration, or an unsupported
quality claim. Recheck the public viewer and saved usage before recording;
the local build alone does not establish that the public build works.
