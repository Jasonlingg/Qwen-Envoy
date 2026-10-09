# EnvoyBench v0.1 candidate release

EnvoyBench evaluates paper-reading agents that write bounded Python programs to
search, read, and cite passages. This release contains the frozen split recipe,
benchmark code, a recorded base-Qwen3-8B versus v5 case study, and a read-only
Studio snapshot. It is a **candidate evaluation protocol, not a validated
leaderboard**.

Read the [technical case study](../../docs/TECHNICAL_CASE_STUDY.md) for the
training objective, observed behavior changes, confounds, and concrete trace
examples. Original training records are in the separate
[training evidence bundle](../qwen-v5-training/README.md).

## Explore saved runs

Open the [public Studio](https://jasonlingg.github.io/Qwen-Envoy/), or download
[`studio.html`](studio.html) and open it directly in a browser. It is self-contained,
read-only, and makes no model or API calls. Its 40 paired questions, actions,
tool observations, answers, and provisional review labels come from the JSON
artifacts in this directory. No GPU is needed. The Studio flow is **Runs →
Questions → Trace**: choose a saved run, pick a question, then inspect that
agent's answer, Python actions, tool observations, and submitted source spans.
All 562 turns from the September 30 runs are embedded in full, including long
tool responses. The October 3 runs add optional token-likelihood panels.
The trace shows where an agent failed or abstained; an exact source span alone
does not prove that the answer is supported.

The September 30 run recorded 5/40 provisional passes for base Qwen3-8B and
15/40 for the v5 SFT adapter. Both passed only 3/20 questions labeled
answerable. V5's observed gain is mainly in abstention and execution
reliability. The labels and answers were **model-assisted reviewed, not
independently human reviewed**; the original planned Sonnet review was
incomplete. Exact citation IDs and source spans establish provenance, not
semantic support. The 40-question `test_candidate` split is not presented as
an independently validated held-out model-improvement result. See the
[case-study report](../../benchmarks/envoybench/COMPARISON_2026_09_30.md) for
the full interpretation and limitations.

The optional **October 3 token diagnostic** is a separate, two-question
development smoke with submissions from both models. It was not answer-quality
scored. Expand a turn's *Token likelihood* panel to inspect the saved
generated-token logprobs; these are likelihoods of tokens the model emitted,
not probabilities that its answer is correct. The interrupted 40-question
expansion supports only an [offline execution diagnostic](../../benchmarks/envoybench/LOGPROB_ANALYSIS.md):
base and v5 submitted answers on only 10 common questions, and logprobs did
not establish a useful error-warning signal. Neither diagnostic changes the
September 30 grades.

## Rebuild the corpus and verify the artifacts

To recompute the reported comparison first, from the repository root:

```bash
python scripts/reproduce_case_study.py
```

This standard-library command works offline without package installation. It
checks this release's hashes and review bindings, then recomputes passes,
answerability breakdown, error-affected episodes and submissions. `--json`
prints the audit as machine-readable output. It does not run models, download
papers, verify source spans against a reconstructed corpus, or independently
validate the model-assisted judgments. The fuller verification below checks
the corpus and spans as well.

From a fresh clone at the repository root, with Python 3.10+ and Docker
available for **new inference** (Docker is not needed for this saved run):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[viewer,qasper]'
python -m benchmarks.envoybench.build_data --output benchmarks/envoybench/data
python -m benchmarks.envoybench.verify \
  --run-dir release/envoybench-v0.1/run \
  --review-dir release/envoybench-v0.1/review-prepared \
  --judged-review release/envoybench-v0.1/review-model-assisted/review.json \
  --provisional
```

The builder fetches the pinned [AllenAI QASPER](https://huggingface.co/datasets/allenai/qasper)
revision if it is not cached, checks source hashes, and materializes paper text
locally. QASPER is by Dasigi et al. (2021), licensed CC BY 4.0; see the
[data provenance](../../benchmarks/envoybench/data/README.md). Full paper text
is intentionally not duplicated in this repository. The verifier checks the
frozen dataset and corpus, complete same-question results, exact spans, and
review bindings. It does **not** independently validate answer correctness.

To explore the full recorded traces locally after building the corpus:

```bash
python scripts/launch_studio.py
```

This launches the packaged runs and token diagnostic on port 8765. The equivalent
explicit command is:

```bash
python -m benchmarks.envoybench.demo \
  --run-dir release/envoybench-v0.1/run \
  --review-dir release/envoybench-v0.1/review-prepared \
  --judged-review release/envoybench-v0.1/review-model-assisted/review.json \
  --token-diagnostic-run release/envoybench-v0.1/token-diagnostic-smoke \
  --provisional --port 8765
```

Open <http://127.0.0.1:8765>. The live paper-reader page requires a separately
configured model endpoint; viewing and verifying the recorded comparison do
not. Instructions for running a new paid model comparison and for interpreting
the candidate split are in the [benchmark README](../../benchmarks/envoybench/README.md).
Omit `--token-diagnostic-run` to hide the separate October 3 smoke. The
[sanitized diagnostic JSON](logprob-diagnostic.json) records its provenance and
step-level summary without raw paper text, actions, answers, or token strings.

To enable the optional live Nebius paper reader, keep `NEBIUS_API_KEY` in your
local `.env` or environment, build the sandbox, and launch explicitly:

```bash
docker build -f benchmarks/envoybench/Dockerfile -t rlm-sandbox .
python scripts/launch_studio.py --live-nebius
```

Opening Studio does not trigger inference. New investigations begin only when
you click **Run** on `/papers`; they consume API credits and are saved separately
from the frozen benchmark. See the [reader guide](../../benchmarks/envoybench/PAPER_READER.md).

`run/` is the unedited inference record. `review-prepared/` contains the
prepared blind packets and mechanical checks. `review-model-assisted/`
contains the disclosed, provisional model-assisted verdicts and report.
`token-diagnostic-smoke/` is a byte-identical copy of the two-question saved
development smoke, with generated actions and public QASPER-derived text;
its source-file hashes are in [the analysis](../../benchmarks/envoybench/LOGPROB_ANALYSIS.md).
`SHA256SUMS` pins the release files. No API credentials or personal vault data
are included. For a short recorded walkthrough, see the [demo script](DEMO_SCRIPT.md).

The optional live path was exercised with NVIDIA Nemotron through Nebius Token
Factory. Ultra completed an investigation but selected incomplete evidence;
three Lightning attempts failed to submit. The [live validation record](LIVE_VALIDATION.md)
preserves all four outcomes. This demonstrates local integration, not a reliable
paper-answering model or a new benchmark score.

The saved viewer is hosted on GitHub Pages. Its deployed HTML was checked
against the packaged file; runs, question navigation, traces, export, and token
panels were exercised in a browser. The publication workflow verifies the
recorded artifact hashes before deployment. This public site runs no inference.

This release is **not a completed Nebius × NVIDIA hackathon submission**.
The live Nebius/NVIDIA test-build instructions still need validation for judge
access; the public video, sponsor feedback, and submission remain pending.
The [Devpost draft](DEVPOST_DRAFT.md) separates the published saved viewer from
those unfinished submission steps.
