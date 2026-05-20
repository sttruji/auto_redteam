from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from auto_trojan.conversation import Conversation


@dataclass
class RewardSignal:
    eval_score: float               # 1.0 = successful attack, 0.0 = failed
    affirmative_likelihood: float   # log-prob of affirmative tokens in response
    refusal_penalty: float          # penalty score for detected refusal keywords
    total: float                    # weighted composite reward


class EvaluationAgent(ABC):
    """Scores attack attempts and produces reward signals."""

    @abstractmethod
    def score(self, attack_prompt: str, victim_response: str) -> RewardSignal:
        """Score a single (attack_prompt, victim_response) pair."""
        ...

    def score_conversation(self, conversation: Conversation) -> RewardSignal:
        """
        Score a full multi-turn conversation.

        Default: score the final attacker→victim exchange. Override for
        richer conversation-aware scoring (e.g., partial credit per turn).
        """
        attack = conversation.last_attacker_turn()
        response = conversation.last_victim_response()
        if attack is None or response is None:
            return RewardSignal(
                eval_score=0.0,
                affirmative_likelihood=0.0,
                refusal_penalty=0.0,
                total=0.0,
            )
        return self.score(attack, response)
