from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from auto_trojan.conversation import Conversation
from auto_trojan.evaluator.base import RewardSignal


@dataclass
class Episode:
    seed_prompt: str
    attack_prompt: str
    victim_response: str
    reward: RewardSignal
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class MultiTurnEpisode:
    """
    A completed multi-turn red-team conversation.

    per_turn_rewards[i] is the reward signal after the victim's i-th response.
    success_turn is the 0-indexed turn at which eval_score first hit 1.0, or None
    if the conversation ended without a successful attack.
    """

    conversation: Conversation
    per_turn_rewards: list[RewardSignal]
    final_reward: RewardSignal
    success_turn: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.success_turn is not None

    @property
    def num_turns(self) -> int:
        return len(self.per_turn_rewards)


class AttackerAgent(ABC):
    """Generates adversarial attack prompts and fine-tunes itself via RL feedback."""

    @abstractmethod
    def generate_attacks(self, seed_prompt: str, k: int) -> list[str]:
        """Return k adversarial variants of seed_prompt (single-turn)."""
        ...

    @abstractmethod
    def fine_tune(self, episodes: list[Episode]) -> None:
        """Update attacker weights based on completed single-turn episodes."""
        ...
