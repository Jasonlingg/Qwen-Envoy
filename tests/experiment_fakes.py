"""Scripted policies and signed fixture packets, not a live model evaluation."""
from src.env.tools_v2 import HARNESS_V2_SHA256
from src.policies.openai_compatible import OpenAICompatiblePolicy
from src.product.paper_assignment import assignment_hash, paper_assignment
from src.research.agent import load_snapshot
from tests.arxiv_fakes import FakeInvestigator
from tests.test_experiment_budget import Tokenizer

QUOTE = "Eviction saves 40 percent of memory"
FINISH = ('finish(findings=[{"claim":"The paper reports memory savings.",'
          '"evidence_ids":["E1"]}], uncertainties=["Our setup needs a local test."], '
          'proposed_test={"hypothesis":"Memory falls", "baseline":"Current model", '
          '"change":"Eviction", "metrics":"Memory and quality", '
          '"stop_rule":"Reject if quality regresses"})')


class ScriptedTransport:
    def __init__(self, actions):
        self.actions, self.calls = iter(actions), []

    def __call__(self, url, payload, headers, timeout):
        self.calls.append(payload)
        action = next(self.actions)
        if isinstance(action, Exception):
            raise action
        if isinstance(action, dict):
            return action
        return {"choices": [{"message": {"content": action}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10}}


class FixtureReader(FakeInvestigator):
    reader_mode = "assigned"
    reader_guidance = "baseline"
    repl_factory = None
    model_identity = {"checkpoint": "fixture-not-a-model"}

    def __init__(self, budget, quotes):
        super().__init__(quotes)
        self.budget = budget

    def policy_factory(self):
        return OpenAICompatiblePolicy(endpoint="http://localhost:8000/v1", model="fixture",
            system_prompt=self.system_prompt, transport=self.budget,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})

    def investigate(self, question, snapshot):
        # Two scripted inference calls share the controller's real budget wrapper.
        policy = self.policy_factory()
        policy.act("Fixture source inspection")
        policy.act("Fixture submission")
        packet = super().investigate(question, snapshot)
        manifest, docs = load_snapshot(snapshot)
        assignment = paper_assignment(manifest, docs)
        return {**packet, "harness_version": "v2", "harness_shim_sha256": HARNESS_V2_SHA256,
                "reader_mode": "assigned", "paper_assignment": assignment,
                "assignment_sha256": assignment_hash(assignment), "research_gaps": []}


def researcher(library, discovery, actions, quotes=None, **limits):
    from src.product.experiment_research import ExperimentResearcher
    transport = ScriptedTransport(actions)
    def factory(budget):
        return FixtureReader(budget, quotes if quotes is not None else {"2609.12345v2": QUOTE})
    return ExperimentResearcher(library, discovery, factory, Tokenizer(), transport,
                                **limits), transport
