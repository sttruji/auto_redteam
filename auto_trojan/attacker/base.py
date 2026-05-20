from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from auto_trojan.evaluator.base import RewardSignal


@dataclass
class Episode:
    seed_prompt: str
    attack_prompt: str
    victim_response: str
    reward: RewardSignal
    metadata: dict[str, object] = field(default_factory=dict)


class AttackerAgent(ABC):
    """Generates adversarial attack prompts and fine-tunes itself via RL feedback."""

    @abstractmethod
    def generate_attacks(self, seed_prompt: str, k: int) -> list[str]:
        """Return k adversarial variants of seed_prompt."""
        ...

    @abstractmethod
    def fine_tune(self, episodes: list[Episode]) -> None:
        """Update attacker weights based on completed episodes."""
        ...
