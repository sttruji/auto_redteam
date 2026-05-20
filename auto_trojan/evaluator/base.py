from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class RewardSignal:
    eval_score: float               # 1.0 = successful attack, 0.0 = failed
    affirmative_likelihood: float   # log-prob of affirmative tokens in response
    refusal_penalty: float          # penalty score for detected refusal keywords
    total: float                    # weighted composite reward


class EvaluationAgent(ABC):
    """Scores a (attack_prompt, victim_response) pair into a RewardSignal."""

    @abstractmethod
    def score(self, attack_prompt: str, victim_response: str) -> RewardSignal:
        ...
