import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from scripts import train_sft
from scripts.train_sft import (
    _checked_data,
    _expand_per_action,
    _load_conversations,
    _sanity_messages,
    _tokenize_per_action,
)

ROOT = Path(__file__).resolve().parents[1]
V5 = ROOT / "data/sft/qasper-v5"


MESSAGES = [
    {"role": "system", "content": "Use tools."},
    {"role": "user", "content": "Question: where?"},
    {"role": "assistant", "content": "print(search('where'))"},
    {"role": "user", "content": "Found d1"},
    {"role": "assistant", "content": 'SUBMIT: There CITATIONS: ["d1"]'},
]


def test_expand_per_action_preserves_history_and_one_target():
    examples = _expand_per_action([{"messages": MESSAGES}])

    assert len(examples) == 2
    assert examples[0] == {
        "prompt": MESSAGES[:2],
        "completion": [MESSAGES[2]],
    }
    assert examples[1] == {
        "prompt": MESSAGES[:4],
        "completion": [MESSAGES[4]],
    }


class _PrefixTokenizer:
    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt=False, enable_thinking=False
    ):
        assert tokenize is True
        assert enable_thinking is False
        ids = []
        for message in messages:
            ids += [len(message["role"]), len(message["content"])]
        if add_generation_prompt:
            ids += [999, 1000]
        elif messages[-1]["role"] == "assistant":
            ids[-2:-2] = [999, 1000]
        return ids


class _MismatchedTokenizer(_PrefixTokenizer):
    def apply_chat_template(self, messages, **kwargs):
        ids = super().apply_chat_template(messages, **kwargs)
        if not kwargs.get("add_generation_prompt") and messages[-1]["role"] == "assistant":
            ids[len(ids) - 4] = 998
        return ids


def test_tokenize_per_action_masks_prompt_and_counts_target():
    examples = _expand_per_action([{"messages": MESSAGES}])
    rows, stats = _tokenize_per_action(_PrefixTokenizer(), examples, 100, "train")

    assert len(rows) == 2
    for row in rows:
        first_target = next(i for i, label in enumerate(row["labels"]) if label != -100)
        assert all(label == -100 for label in row["labels"][:first_target])
        assert row["labels"][first_target:] == row["input_ids"][first_target:]
    assert stats == {
        "actions": 2,
        "input_tokens": 20,
        "supervised_tokens": 4,
        "max_tokens": 12,
    }


def test_tokenize_per_action_refuses_prefix_mismatch():
    examples = _expand_per_action([{"messages": MESSAGES}])
    with pytest.raises(ValueError, match="do not start with the inference generation prefix"):
        _tokenize_per_action(_MismatchedTokenizer(), examples, 100, "train")


def test_tokenize_per_action_refuses_truncation():
    examples = _expand_per_action([{"messages": MESSAGES}])
    with pytest.raises(ValueError, match="refusing silent target truncation"):
        _tokenize_per_action(_PrefixTokenizer(), examples, 5, "train")


def _write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_published_v5_preflight_and_sanity_prompt():
    train, val = _checked_data(
        V5 / "train.jsonl", V5 / "val.jsonl", V5 / "manifest.json"
    )
    assert (len(train), len(val)) == (35, 9)
    assert len(_expand_per_action(train)) == 106
    assert _sanity_messages("test")[0] == train[0]["messages"][0]


@pytest.mark.parametrize("contents, error", [
    ("{broken\n", "invalid JSON"),
    (json.dumps({"messages": MESSAGES[:-1]}) + "\n", "need system, user, assistant"),
    (
        json.dumps({"messages": [
            MESSAGES[0], MESSAGES[1], {"role": "assistant", "content": "  "},
        ]}) + "\n",
        "nonempty assistant",
    ),
])
def test_preflight_rejects_malformed_conversations(tmp_path, contents, error):
    path = tmp_path / "bad.jsonl"
    path.write_text(contents)
    with pytest.raises(ValueError, match=error):
        _load_conversations(path, "train")


def test_preflight_rejects_shared_question_or_paper(tmp_path):
    def row(question):
        return {"messages": [MESSAGES[0], {"role": "user", "content": question},
                             MESSAGES[2]]}

    train = _write_rows(tmp_path / "train.jsonl", [row('Question: A (doc_id: "d1")')])
    val = _write_rows(tmp_path / "val.jsonl", [row('Question: A (doc_id: "d1")')])
    with pytest.raises(ValueError, match="share an initial question"):
        _checked_data(train, val, None)

    _write_rows(val, [row('Question: B (doc_id: "d1")')])
    with pytest.raises(ValueError, match="share an assigned paper"):
        _checked_data(train, val, None)


def test_preflight_rejects_changed_manifest_hash(tmp_path):
    train = _write_rows(tmp_path / "train.jsonl", [{"messages": MESSAGES}])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"splits": {"train": {"sha256": "wrong"}}}))
    with pytest.raises(ValueError, match="train file does not match"):
        _checked_data(train, None, manifest)


def test_existing_output_is_rejected_before_model_load(tmp_path, monkeypatch):
    data = _write_rows(tmp_path / "train.jsonl", [{"messages": MESSAGES}])
    output = tmp_path / "existing"
    output.mkdir()
    (output / "checkpoint-1").write_text("keep")
    monkeypatch.setattr(train_sft, "_load_training_deps", lambda: pytest.fail("model loaded"))
    result = CliRunner().invoke(train_sft.app, [
        "train", "--data", str(data), "--out", str(output),
    ])
    assert isinstance(result.exception, ValueError)
    assert "output directory is not empty" in str(result.exception)
    assert (output / "checkpoint-1").read_text() == "keep"


def test_sanity_check_uses_pinned_base_and_non_thinking_prompt(monkeypatch):
    calls = {}

    class Tokenizer:
        eos_token_id = 7

        @classmethod
        def from_pretrained(cls, model, **kwargs):
            calls["tokenizer"] = (model, kwargs)
            return cls()

        def apply_chat_template(self, messages, **kwargs):
            calls["template"] = (messages, kwargs)
            return "rendered"

        def __call__(self, text, **kwargs):
            assert text == "rendered" and kwargs == {"return_tensors": "pt"}
            return SimpleNamespace(to=lambda device: {"input_ids": SimpleNamespace(shape=(1, 2))})

        def decode(self, tokens, **kwargs):
            assert tokens == [9] and kwargs == {"skip_special_tokens": True}
            return "print(search('x'))"

    class Base:
        @classmethod
        def from_pretrained(cls, model, **kwargs):
            calls["base"] = (model, kwargs)
            return cls()

    class Adapter:
        device = "cpu"

        @classmethod
        def from_pretrained(cls, base, path):
            assert isinstance(base, Base) and path == "adapter"
            return cls()

        def eval(self):
            pass

        def generate(self, **kwargs):
            assert kwargs["max_new_tokens"] == 8
            assert kwargs["do_sample"] is False
            assert kwargs["pad_token_id"] == 7
            return [[0, 0, 9]]

    torch = SimpleNamespace(bfloat16="bf16", no_grad=nullcontext)
    monkeypatch.setattr(train_sft, "_load_training_deps", lambda: (
        torch, None, None, Adapter, Base, Tokenizer, None, None, None,
    ))
    train_sft.sanity_check(
        model=Path("adapter"), question="where?", max_new_tokens=8,
        base_model=train_sft.BASE_MODEL, base_revision=train_sft.BASE_REVISION,
    )

    assert calls["tokenizer"] == (train_sft.BASE_MODEL, {
        "revision": train_sft.BASE_REVISION, "trust_remote_code": True,
    })
    assert calls["base"][1]["revision"] == train_sft.BASE_REVISION
    messages, settings = calls["template"]
    assert messages == train_sft._sanity_messages("where?")
    assert settings["enable_thinking"] is False
    assert settings["add_generation_prompt"] is True
