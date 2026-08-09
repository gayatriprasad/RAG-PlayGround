"""
Factory for prompt strategies.
"""

from __future__ import annotations

from raglab.prompts.base import BasePromptStrategy


def get_prompt_strategy(cfg) -> BasePromptStrategy:
    """
    Instantiate prompt strategy based on config.

    Args:
        cfg: PromptCfg with strategy field

    Returns:
        BasePromptStrategy implementation
    """
    strategy = cfg.strategy

    match strategy:
        case "zero_shot":
            from raglab.prompts.zero_shot import ZeroShotPrompt
            return ZeroShotPrompt(cfg)
        case "few_shot":
            from raglab.prompts.few_shot import FewShotPrompt
            return FewShotPrompt(cfg)
        case "cot":
            from raglab.prompts.cot import ChainOfThoughtPrompt
            return ChainOfThoughtPrompt(cfg)
        case "self_consistency":
            from raglab.prompts.self_consistency import SelfConsistencyPrompt
            return SelfConsistencyPrompt(cfg)
        case "medprompt":
            from raglab.prompts.medprompt import MedpromptPrompt
            return MedpromptPrompt(cfg)
        case _:
            raise ValueError(
                f"Unknown prompt strategy: '{strategy}'. "
                f"Valid: zero_shot, few_shot, cot, self_consistency, medprompt"
            )
