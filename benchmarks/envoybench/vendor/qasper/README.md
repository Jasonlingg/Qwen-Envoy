# Official QASPER evaluator

`evaluator.py` is an **unmodified** copy of AllenAI's official evaluator from
[`allenai/qasper-led-baseline`](https://github.com/allenai/qasper-led-baseline/blob/e996b6c7b1b5f95d9308a74e3586416c6e780df1/scripts/evaluator.py),
commit `e996b6c7b1b5f95d9308a74e3586416c6e780df1` (May 20, 2021).
The upstream Apache 2.0 [LICENSE](LICENSE) is included without modification.
The script credits the official SQuAD v1.1 evaluator for answer normalization
and token F1. No AllenNLP, Transformers, or dataset package is required to run it.

| File | SHA-256 |
| --- | --- |
| `evaluator.py` | `781aba7cd8e524bef4f0a1b4bf3504e5b02cb1d8d5bf32a8f0a89dfa83e86bfe` |
| `LICENSE` | `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` |

The surrounding `qasper_official.py` adapter reports only the evaluator's
**Answer F1**, on the selected 40-question agent-study subset. It preserves all
original answer annotations and evaluates the literal submitted answer text.
It does not report Evidence F1: the agent's character spans are a different
output format from QASPER's paragraph strings, and no conversion is claimed.

Dataset citation: Pradeep Dasigi, Kyle Lo, Iz Beltagy, Arman Cohan, Noah A.
Smith, and Matt Gardner. 2021. [A Dataset of Information-Seeking Questions and
Answers Anchored in Research Papers](https://aclanthology.org/2021.naacl-main.365/).
The QASPER dataset has its own CC BY 4.0 attribution in the reference export.
