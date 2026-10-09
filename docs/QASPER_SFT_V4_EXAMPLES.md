# What the new teacher examples teach

Update: the [v5 dataset](../data/sft/qasper-v5/README.md) now contains 44 reviewed
conversations. The examples below describe the original v4 seed, which is preserved.

September 18, 2026. We generated 24 Claude episodes and kept 21 after source review:
four unchanged and 17 after Codex edits followed by fresh execution. Three questions
were excluded because of missing source coverage or ambiguous/mismatched references.
All 21 accepted episodes replay exactly. Review was by the editing assistant, not an
independent human.

The files are in `data/sft/qasper-v4/`: 17 training conversations and four validation
conversations, with different papers in each split. This is a reviewed seed batch, not
a sufficient replacement training set or evidence that Qwen has improved.

## A straightforward answer

**Question:** How many instances does the dataset in *Speaker-independent classification
of phonetic segments from raw ultrasound in child speech* have?

The question supplies the paper ID. The first action searches using the question's
concepts:

```python
print(search_within("qasper_1907_01413", "dataset training examples"))
```

The actual tool result includes a passage at offset 8600 saying the sampling process
produces 10,700 training examples. The next action is:

```text
SUBMIT: The paper reports about 10,700 training examples after sampling. CITATIONS: ["qasper_1907_01413"]
```

The original teacher had put `10700` into its first search because it had seen the gold
answer in a private hint. The repaired example removes that shortcut. The full training
record contains the actual returned text, not just this summary of it.

## Knowing when a number is missing

**Question:** How much faster do natural-language agents converge in the experiments in
*Natural Language State Representation for Reinforcement Learning*?

First action:

```python
print(search_within("qasper_1910_02789", "convergence faster natural language agents"))
```

The returned abstract says the agents converge faster, but gives no amount. The next
action checks the paper's convergence paragraphs:

```python
print(extract("qasper_1910_02789", r"[^\n]*converg[^\n]*"))
```

The tool returns the abstract and a results paragraph. Both make a qualitative claim;
neither quantifies a speedup. The final action is:

```text
SUBMIT: The retrieved abstract and results paragraph say natural-language agents converge faster than the comparison agents, but give no numerical speedup. I cannot quantify how much faster from those passages. CITATIONS: ["qasper_1910_02789"]
```

This teaches a specific evidence gap. It does not teach that faster convergence is false
or that a couple of keyword searches prove something is absent from an entire paper.

## How SFT uses these records

For each action, Qwen receives the question and previous real tool observations and
learns to predict the next Python action or submission. The existing training loader
applies loss to that action; teacher-only answerability hints and reference answers are
not included in student inputs. The code-execution task remains multi-step.

The useful change is the quality of those targets: attainable searches, source-supported
answers, bounded uncertainty, and stopping once the requested facts are established.
We have not changed the QLoRA method or retrained Qwen in this pass.

The [experiment record](QWEN3_SFT_DATA_ITERATION.md) includes the failed generation gates,
usage counts, review limitations, and next evaluation criteria. The
[review file](../data/research/qasper_sft_v4_review.json) preserves the exact edits and
hashes so these examples can be audited.
