"""Use the strengthened sandbox for new research-MCP reader episodes."""

from __future__ import annotations

from pathlib import Path

from src.env.hardened_repl import HardenedDockerREPL
from src.env.tools_v2 import HARNESS_V2_PREAMBLE
from src.product.qwen_investigator import QwenInvestigator


class HardenedQwenInvestigator(QwenInvestigator):
    def _repl(self, corpus: Path):
        if self.repl_factory is not None:
            return super()._repl(corpus)
        return HardenedDockerREPL(
            image=self.image,
            corpus_path=str(corpus),
            extra_preamble=HARNESS_V2_PREAMBLE if self.harness_version == "v2" else "",
        )
