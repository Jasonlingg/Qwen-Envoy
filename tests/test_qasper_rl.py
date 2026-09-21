import copy
import json

import torch

from src.eval.qasper_reward import answer_f1, parse_strict, score_submission
from scripts.train_grpo_custom import _action_logits
from scripts.train_qasper_grpo import adapter_fingerprint


DOCS = {"p": {"text": "x"*1000}, "other": {"text": "x"*1000}}
REF = {"answer_type": "abstractive", "answer_text": "red red blue",
       "unanswerable": False, "evidence": [{"doc_id":"p", "start":100, "end":200}]}
QUESTION = {"answer_annotations": [REF]}


def submit(answer="red red blue", evidence=None, citations=None):
    if evidence is None:
        evidence = [{"doc_id":"p", "start":100, "end":200}]
    if citations is None:
        citations = sorted({e["doc_id"] for e in evidence})
    return f"SUBMIT: {answer} CITATIONS: {json.dumps(citations)} EVIDENCE: {json.dumps(evidence)}"


def score(action, q=QUESTION, investigated=True):
    return score_submission(action, q, DOCS, investigated=investigated)


def test_multiset_f1_and_all_references():
    assert answer_f1("red", "red red blue") == .5
    q = copy.deepcopy(QUESTION)
    q["answer_annotations"].insert(0, dict(REF, answer_text="green"))
    assert score(submit(), q)["reward"] == 1


def test_evidence_shortcuts_cannot_get_full_credit():
    good = score(submit())["reward"]
    for start, end in [(100,101), (0,1000), (500,600)]:
        bad = score(submit(evidence=[{"doc_id":"p", "start":start, "end":end}]))
        assert bad["reward"] < good
    assert score(submit("orange"))["reward"] == 0
    assert score(submit(), investigated=False)["reward"] == 0
    assert score(submit(evidence=[]))["reward"] == 0


def test_invalid_spans_and_mismatched_citations_are_zero():
    for start, end in [(-1,10), (10,1001), (10,10), (True,100)]:
        assert score(submit(evidence=[{"doc_id":"p", "start":start, "end":end}]))["reward"] == 0
    assert score(submit(citations=["other"]))["reward"] == 0
    assert parse_strict(submit()+" extra garbage") is None
    assert parse_strict("SUBMIT: CITATIONS: [] EVIDENCE: []") is None


def test_abstention_is_not_timeout_or_false_refusal():
    q = {"answer_annotations":[dict(REF, answer_type="unanswerable", unanswerable=True)]}
    abstain = submit("Unanswerable", evidence=[])
    assert score(abstain, q)["reward"] == 1
    assert score(abstain)["reason"] == "false_refusal"
    assert score("", q)["reward"] == 0
    assert score(submit(), q)["reward"] == 0


def test_boolean_cannot_hedge_with_yes_and_no():
    q = {"answer_annotations":[dict(REF, answer_type="boolean", answer_text="Yes")]}
    assert score(submit("Yes"), q)["reward"] == 1
    assert score(submit("Yes No"), q)["reward"] == 0


def test_adapter_fingerprint_is_scoped_and_detects_changes():
    class FakeModel:
        policy = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        reference = torch.nn.Parameter(torch.tensor([1.0, 2.0]), requires_grad=False)

        def named_parameters(self):
            return iter([
                ("layer.lora_A.default.weight", self.policy),
                ("layer.lora_A.kl_ref.weight", self.reference),
            ])

    model = FakeModel()
    policy_before, count = adapter_fingerprint(model, "default")
    reference_before, _ = adapter_fingerprint(model, "kl_ref")
    assert count == 2
    with torch.no_grad():
        model.policy.add_(1)
    assert adapter_fingerprint(model, "default")[0] != policy_before
    assert adapter_fingerprint(model, "kl_ref")[0] == reference_before


def test_efficient_logits_match_full_forward_and_gradient():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(20, 6)
            self.head = torch.nn.Linear(6, 20)
        def forward(self, ids, logits_to_keep=0):
            from types import SimpleNamespace
            h = self.emb(ids)
            if logits_to_keep:
                h = h[:, -logits_to_keep:]
            return SimpleNamespace(logits=self.head(h))
    m = Model()
    ids = torch.arange(10).unsqueeze(0)
    full = m(ids).logits[0, 5:9]
    efficient = _action_logits(m, ids, 6)
    torch.testing.assert_close(full, efficient)
    a = torch.autograd.grad(full.sum(), tuple(m.parameters()), retain_graph=True)
    b = torch.autograd.grad(efficient.sum(), tuple(m.parameters()))
    for left,right in zip(a,b):
        torch.testing.assert_close(left,right)


def test_qwen_saved_defaults_cannot_override_experiment_decoding():
    from transformers import Qwen3Config, Qwen3ForCausalLM, GenerationConfig
    from scripts.train_qasper_grpo import generation_kwargs
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=32, hidden_size=16,
        intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, head_dim=8))
    model.generation_config = GenerationConfig(do_sample=True, temperature=.6,
                                               top_k=20, top_p=.95)
    for temperature in (0.0, 1.0):
        prepared, _ = model._prepare_generation_config(**generation_kwargs(temperature, 12, 2))
        assert prepared.do_sample == (temperature>0)
        assert prepared.top_k == 0 and prepared.top_p == 1.0
        if temperature>0:
            assert prepared.temperature == temperature
        assert prepared.max_new_tokens == 12
