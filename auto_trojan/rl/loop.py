from dataclasses import dataclass, field

from rich.console import Console

from auto_trojan.attacker.base import AttackerAgent, Episode
from auto_trojan.evaluator.base import EvaluationAgent
from auto_trojan.victim.base import VictimLLM

console = Console()


@dataclass
class RLConfig:
    k: int = 10                  # attack prompts per episode
    episodes: int = 500
    seed: int = 42
    log_every: int = 10
    checkpoint_dir: str = "checkpoints"
    extra: dict[str, object] = field(default_factory=dict)


class RLLoop:
    def __init__(
        self,
        attacker: AttackerAgent,
        victim: VictimLLM,
        evaluator: EvaluationAgent,
        config: RLConfig,
    ) -> None:
        self.attacker = attacker
        self.victim = victim
        self.evaluator = evaluator
        self.config = config

    def run(self, seed_prompts: list[str]) -> None:
        for ep in range(self.config.episodes):
            episodes: list[Episode] = []

            for seed in seed_prompts:
                attacks = self.attacker.generate_attacks(seed, self.config.k)
                for attack in attacks:
                    response = self.victim.generate(attack)
                    reward = self.evaluator.score(attack, response)
                    episodes.append(
                        Episode(
                            seed_prompt=seed,
                            attack_prompt=attack,
                            victim_response=response,
                            reward=reward,
                        )
                    )

            self.attacker.fine_tune(episodes)

            if ep % self.config.log_every == 0:
                avg_reward = sum(e.reward.total for e in episodes) / len(episodes)
                console.log(f"Episode {ep:04d} | avg_reward={avg_reward:.4f}")
