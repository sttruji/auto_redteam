from __future__ import annotations

from dataclasses import dataclass

from rich.console import Console

from auto_trojan.attacker.base import MultiTurnEpisode
from auto_trojan.attacker.multiturn_attacker import MultiTurnAttackerAgent
from auto_trojan.conversation import Conversation
from auto_trojan.evaluator.base import EvaluationAgent
from auto_trojan.victim.base import VictimLLM

console = Console()


@dataclass
class MultiTurnRLConfig:
    max_turns: int = 6          # maximum attacker turns per conversation
    episodes: int = 500
    seed: int = 42
    log_every: int = 10
    checkpoint_dir: str = "checkpoints"


class MultiTurnRLLoop:
    """
    RL training loop for multi-turn red-teaming.

    Each episode is a conversation rollout:
      for t in range(max_turns):
        attacker generates next turn
        victim responds
        evaluator scores the exchange
        if success → stop early

    Rollouts are collected, then the attacker is updated via fine_tune_multiturn().
    """

    def __init__(
        self,
        attacker: MultiTurnAttackerAgent,
        victim: VictimLLM,
        evaluator: EvaluationAgent,
        config: MultiTurnRLConfig,
    ) -> None:
        self.attacker = attacker
        self.victim = victim
        self.evaluator = evaluator
        self.config = config

    def run(self, seed_prompts: list[str]) -> None:
        for ep_num in range(self.config.episodes):
            episodes = [self._rollout(seed) for seed in seed_prompts]
            self.attacker.fine_tune_multiturn(episodes)

            if ep_num % self.config.log_every == 0:
                self._log(ep_num, episodes)

    def _rollout(self, seed: str) -> MultiTurnEpisode:
        """
        Run one full conversation and collect per-turn rewards.

        Stops early as soon as eval_score == 1.0 to avoid wasting turns
        after a successful attack — and to assign credit to the turn that
        actually caused the breakthrough.
        """
        conversation = Conversation(seed_goal=seed)
        per_turn_rewards = []
        success_turn = None

        for t in range(self.config.max_turns):
            attack = self.attacker.generate_next_turn(conversation)
            conversation.add_turn("attacker", attack)

            response = self.victim.generate_with_history(conversation)
            conversation.add_turn("victim", response)

            reward = self.evaluator.score_conversation(conversation)
            per_turn_rewards.append(reward)

            if reward.eval_score == 1.0 and success_turn is None:
                success_turn = t
                break

        return MultiTurnEpisode(
            conversation=conversation,
            per_turn_rewards=per_turn_rewards,
            final_reward=per_turn_rewards[-1],
            success_turn=success_turn,
        )

    def _log(self, ep_num: int, episodes: list[MultiTurnEpisode]) -> None:
        n = len(episodes)
        avg_reward = sum(e.final_reward.total for e in episodes) / n
        success_rate = sum(1 for e in episodes if e.succeeded) / n
        avg_turns = sum(e.num_turns for e in episodes) / n
        console.log(
            f"Episode {ep_num:04d} | "
            f"avg_reward={avg_reward:.4f} | "
            f"success_rate={success_rate:.2%} | "
            f"avg_turns={avg_turns:.1f}"
        )
