"""Frozen scope for the candidate MCP test; never approval or inference."""
from src.product.assigned_guidance import FOCUSED


def candidate_contract() -> dict:
    return {"reader_guidance": FOCUSED, "execution_path": "mcp_in_process",
        "hypothesis": "Focused-v1 guidance lets the same v5 reader interpret displayed "
            "evidence correctly, recover from rejected actions and report a qualified gap.",
        "expected_signal": "A supported batch-size answer and an explicit A5000/batch8 "
            "gap with useful follow-up query, returned through the real MCP tool.",
        "decision_rule": "Both cases must meet the unchanged source/claim review "
            "requirements, pass MCP/packet checks, include any required gap report, and "
            "have zero Python execution errors, controller/submission/gap rejections, "
            "repeated calls or budget violations. Exact quotes alone do not pass a case.",
        "comparison": "Same two familiar development cases as the saved baseline; "
            "candidate prompt/feedback/gap gate and MCP invocation path change. This is "
            "a development regression, not a held-out or causal fine-tuning evaluation.",
        "e2e_scope": "In-process MCP client/protocol/tool, bounded remote Qwen transport, "
            "Docker document actions, packet verification, host-visible result and "
            "shutdown. Cached paper only; no live discovery, desktop client or stdio "
            "process-launch integration. No reasoner or additional model.",
        "execution_readiness": "Preparation only. Requires one separately approved "
            "attempt, fresh catalog/parked-state checks, a new exclusive marker and "
            "verified pinned worker identity before inference. No provider dollar cap."}
