from __future__ import annotations

from dataclasses import dataclass

from auto_trojan.evaluator.base import RewardSignal


@dataclass
class RewardWeights:
    eval: float = 1.0
    affirmative_likelihood: float = 0.5
    refusal_penalty: float = 0.3


class RewardModel:
    def __init__(self, weights: RewardWeights | None = None) -> None:
        self.weights = weights or RewardWeights()

    def compute(
        self,
        eval_score: float,
        affirmative_likelihood: float,
        refusal_penalty: float,
    ) -> RewardSignal:
        w = self.weights
        total = (
            w.eval * eval_score
            + w.affirmative_likelihood * affirmative_likelihood
            - w.refusal_penalty * refusal_penalty
        )
        return RewardSignal(
            eval_score=eval_score,
            affirmative_likelihood=affirmative_likelihood,
            refusal_penalty=refusal_penalty,
            total=total,
        )
