"""
Tests for MultiTurnAttackerAgent, ValueHead, compute_discounted_returns,
and MultiTurnRLLoop. No GPU or real model weights required.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest
import torch
import torch.nn as nn

from auto_trojan.attacker.base import MultiTurnEpisode
from auto_trojan.attacker.multiturn_attacker import (
    MultiTurnAttackerAgent,
    MultiTurnAttackerConfig,
    compute_discounted_returns,
)
from auto_trojan.attacker.value_head import ValueHead
from auto_trojan.conversation import Conversation, Turn
from auto_trojan.evaluator.base import RewardSignal
from auto_trojan.rl.multiturn_loop import MultiTurnRLConfig, MultiTurnRLLoop


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _signal(total: float) -> RewardSignal:
    return RewardSignal(eval_score=float(total > 0), affirmative_likelihood=0.0,
                        refusal_penalty=0.0, total=total)


def _conv(*turn_pairs: tuple[str, str], seed: str = "goal") -> Conversation:
    c = Conversation(seed_goal=seed)
    for role, content in turn_pairs:
        c.add_turn(role, content)  # type: ignore[arg-type]
    return c


def _episode(rewards: list[float], success_turn: int | None = None) -> MultiTurnEpisode:
    turns = []
    for i, r in enumerate(rewards):
        turns.append(("attacker", f"attack {i}"))
        turns.append(("victim", f"response {i}"))
    c = _conv(*turns)
    per_turn = [_signal(r) for r in rewards]
    return MultiTurnEpisode(
        conversation=c,
        per_turn_rewards=per_turn,
        final_reward=per_turn[-1],
        success_turn=success_turn,
    )


def _make_agent(tmp_path: Path) -> MultiTurnAttackerAgent:
    """Build MultiTurnAttackerAgent with everything mocked."""
    config = MultiTurnAttackerConfig(
        model_id="fake/model",
        device="cpu",
        lora_rank=4,
        checkpoint_dir=str(tmp_path / "ckpts"),
        checkpoint_every=5,
        gamma=0.9,
        value_loss_coeff=0.5,
    )

    tok = MagicMock()
    tok.pad_token = "<pad>"
    tok.pad_token_id = 0
    tok.padding_side = "left"
    tok.decode = MagicMock(return_value="generated attack turn")

    seq_len = 10
    vocab = 32

    def tok_call(text, return_tensors=None, truncation=False, max_length=None, **kw):
        if isinstance(text, list):
            b = len(text)
            out = MagicMock()
            out.input_ids = torch.zeros(b, seq_len, dtype=torch.long)
            out.attention_mask = torch.ones(b, seq_len, dtype=torch.long)
            out.to = lambda d: out
            return out
        out = MagicMock()
        out.input_ids = torch.zeros(1, seq_len, dtype=torch.long)
        out.to = lambda d: out
        return out

    tok.side_effect = tok_call

    # Fake model
    hidden_size = 64
    model = MagicMock()
    model.generate = MagicMock(return_value=torch.zeros(1, seq_len + 5, dtype=torch.long))
    model.eval = MagicMock()
    model.train = MagicMock()
    model.to = MagicMock(return_value=model)
    model.save_pretrained = MagicMock()

    # Model forward: returns logits + hidden states
    def model_forward(*args, output_hidden_states=False, **kw):
        inp = args[0] if args else kw.get("input_ids", torch.zeros(1, seq_len))
        B, T = inp.shape if inp.dim() == 2 else (1, seq_len)
        logits = torch.randn(B, T, vocab, requires_grad=True)
        out = MagicMock()
        out.logits = logits
        if output_hidden_states:
            out.hidden_states = (torch.randn(B, T, hidden_size, requires_grad=True),) * 4
        return out

    model.side_effect = model_forward
    model.parameters = MagicMock(return_value=iter([torch.nn.Parameter(torch.randn(4))]))

    # Fake config with hidden_size
    model.config = MagicMock()
    model.config.hidden_size = hidden_size

    agent = MultiTurnAttackerAgent.__new__(MultiTurnAttackerAgent)
    agent.config = config
    agent._step = 0
    agent._baseline = 0.0
    agent.tokenizer = tok
    agent.model = model

    # Real value head (uses hidden_size from mock config)
    agent.value_head = ValueHead(hidden_size, dropout=0.0)
    agent.value_head.to("cpu")

    # Mock optimizers
    agent.optimizer = MagicMock()
    agent.optimizer.zero_grad = MagicMock()
    agent.optimizer.step = MagicMock()
    agent.value_optimizer = MagicMock()
    agent.value_optimizer.zero_grad = MagicMock()
    agent.value_optimizer.step = MagicMock()

    return agent


# ---------------------------------------------------------------------------
# ValueHead
# ---------------------------------------------------------------------------

class TestValueHead:
    def test_output_shape_3d_input(self):
        vh = ValueHead(hidden_size=32)
        x = torch.randn(4, 10, 32)
        out = vh(x)
        assert out.shape == (4,)

    def test_output_shape_2d_input(self):
        vh = ValueHead(hidden_size=32)
        x = torch.randn(4, 32)
        out = vh(x)
        assert out.shape == (4,)

    def test_uses_last_token_for_3d(self):
        """Verify value only changes when last token changes, not earlier ones."""
        vh = ValueHead(hidden_size=16, dropout=0.0)
        base = torch.randn(1, 5, 16)

        perturbed_early = base.clone()
        perturbed_early[0, 0, :] += 100.0  # change first token
        perturbed_last = base.clone()
        perturbed_last[0, -1, :] += 100.0  # change last token

        v_base = vh(base)
        v_early = vh(perturbed_early)
        v_last = vh(perturbed_last)

        assert torch.isclose(v_base, v_early, atol=1e-5).all()
        assert not torch.isclose(v_base, v_last, atol=1e-5).all()

    def test_gradients_flow(self):
        vh = ValueHead(hidden_size=16, dropout=0.0)
        x = torch.randn(2, 8, 16, requires_grad=True)
        loss = vh(x).sum()
        loss.backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# compute_discounted_returns
# ---------------------------------------------------------------------------

class TestDiscountedReturns:
    def test_single_reward(self):
        assert compute_discounted_returns([5.0], gamma=0.9) == pytest.approx([5.0])

    def test_two_rewards_discounted(self):
        # G_0 = 1.0 + 0.9*2.0 = 2.8, G_1 = 2.0
        result = compute_discounted_returns([1.0, 2.0], gamma=0.9)
        assert result == pytest.approx([2.8, 2.0])

    def test_gamma_zero_means_immediate_only(self):
        result = compute_discounted_returns([3.0, 5.0, 7.0], gamma=0.0)
        assert result == pytest.approx([3.0, 5.0, 7.0])

    def test_gamma_one_means_undiscounted_sum(self):
        result = compute_discounted_returns([1.0, 2.0, 3.0], gamma=1.0)
        assert result == pytest.approx([6.0, 5.0, 3.0])

    def test_empty(self):
        assert compute_discounted_returns([], gamma=0.9) == []

    def test_length_preserved(self):
        rewards = [0.1, 0.5, 0.2, 0.9]
        result = compute_discounted_returns(rewards, gamma=0.95)
        assert len(result) == len(rewards)

    def test_monotone_for_positive_rewards(self):
        """Later turns have smaller returns when all rewards are positive."""
        result = compute_discounted_returns([1.0, 1.0, 1.0], gamma=0.9)
        # G_0 > G_1 > G_2 for gamma < 1
        assert result[0] > result[1] > result[2]


# ---------------------------------------------------------------------------
# MultiTurnAttackerAgent: prompt formatting
# ---------------------------------------------------------------------------

class TestMultiTurnPromptFormatting:
    def _agent(self) -> MultiTurnAttackerAgent:
        agent = MultiTurnAttackerAgent.__new__(MultiTurnAttackerAgent)
        agent.config = MultiTurnAttackerConfig(model_id="x", device="cpu")
        return agent

    def test_format_includes_seed_goal(self):
        agent = self._agent()
        result = agent._format_multiturn_prompt("steal data", [])
        assert "steal data" in result

    def test_format_includes_history_turns(self):
        agent = self._agent()
        history = [Turn("attacker", "first msg"), Turn("victim", "response")]
        result = agent._format_multiturn_prompt("goal", history)
        assert "first msg" in result
        assert "response" in result

    def test_format_ends_with_user_prompt(self):
        agent = self._agent()
        result = agent._format_multiturn_prompt("goal", [])
        assert result.strip().endswith("User:")

    def test_attacker_role_maps_to_user(self):
        agent = self._agent()
        history = [Turn("attacker", "my message")]
        result = agent._format_multiturn_prompt("goal", history)
        assert "User: my message" in result

    def test_victim_role_maps_to_assistant(self):
        agent = self._agent()
        history = [Turn("victim", "victim reply")]
        result = agent._format_multiturn_prompt("goal", history)
        assert "Assistant: victim reply" in result


# ---------------------------------------------------------------------------
# MultiTurnAttackerAgent: generate_next_turn
# ---------------------------------------------------------------------------

class TestGenerateNextTurn:
    def test_returns_string(self, tmp_path):
        agent = _make_agent(tmp_path)
        c = _conv(("attacker", "hi"), ("victim", "hey"))
        result = agent.generate_next_turn(c)
        assert isinstance(result, str)

    def test_switches_to_eval_then_train(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent.generate_next_turn(_conv())
        agent.model.eval.assert_called()
        agent.model.train.assert_called()

    def test_passes_history_in_prompt(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent.tokenizer.side_effect = None  # capture calls

        calls = []
        def tok_capture(text, **kw):
            calls.append(text)
            out = MagicMock()
            out.input_ids = torch.zeros(1, 5, dtype=torch.long)
            out.to = lambda d: out
            return out

        agent.tokenizer.side_effect = tok_capture
        c = _conv(("attacker", "UNIQUE_ATTACK_MSG"), ("victim", "victim_response"), seed="MY_GOAL")
        agent.generate_next_turn(c)

        combined = " ".join(str(c) for c in calls)
        assert "UNIQUE_ATTACK_MSG" in combined or "MY_GOAL" in combined


# ---------------------------------------------------------------------------
# MultiTurnAttackerAgent: fine_tune_multiturn
# ---------------------------------------------------------------------------

class TestFineTuneMultiturn:
    def test_empty_episodes_is_noop(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent.fine_tune_multiturn([])
        agent.optimizer.step.assert_not_called()
        assert agent._step == 0

    def test_step_increments(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent.fine_tune_multiturn([_episode([1.0])])
        assert agent._step == 1

    def test_both_optimizers_step(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent.fine_tune_multiturn([_episode([0.5, 1.0])])
        agent.optimizer.step.assert_called_once()
        agent.value_optimizer.step.assert_called_once()

    def test_both_optimizers_zero_grad(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent.fine_tune_multiturn([_episode([1.0])])
        agent.optimizer.zero_grad.assert_called_once()
        agent.value_optimizer.zero_grad.assert_called_once()

    def test_checkpoint_saved_at_interval(self, tmp_path):
        agent = _make_agent(tmp_path)  # checkpoint_every=5
        for _ in range(4):
            agent.fine_tune_multiturn([_episode([1.0])])
        agent.model.save_pretrained.assert_not_called()

        agent.fine_tune_multiturn([_episode([1.0])])  # step 5
        agent.model.save_pretrained.assert_called_once()

    def test_processes_multiturn_episode(self, tmp_path):
        """Ensure multi-turn episodes (>1 reward) process all turns."""
        agent = _make_agent(tmp_path)
        ep = _episode([0.2, 0.5, 1.0], success_turn=2)
        agent.fine_tune_multiturn([ep])
        # All 3 turns should have been processed — model was called at least 3 times
        assert agent.model.call_count >= 3


# ---------------------------------------------------------------------------
# MultiTurnRLLoop
# ---------------------------------------------------------------------------

class TestMultiTurnRLLoop:
    def _make_loop(self, max_turns: int = 3) -> tuple[MultiTurnRLLoop, MagicMock, MagicMock, MagicMock]:
        attacker = MagicMock()
        attacker.generate_next_turn = MagicMock(return_value="attack message")

        victim = MagicMock()
        victim.generate_with_history = MagicMock(return_value="victim response")

        evaluator = MagicMock()
        # Default: always fail (eval_score=0, total=0)
        evaluator.score_conversation = MagicMock(
            return_value=RewardSignal(0.0, 0.0, 0.0, 0.0)
        )

        config = MultiTurnRLConfig(max_turns=max_turns, episodes=1, log_every=999)
        loop = MultiTurnRLLoop(attacker, victim, evaluator, config)
        return loop, attacker, victim, evaluator

    def test_rollout_builds_correct_turn_count(self):
        loop, attacker, victim, evaluator = self._make_loop(max_turns=3)
        ep = loop._rollout("seed")
        # 3 turns = 3 attacker + 3 victim = 6 conversation turns
        assert ep.conversation.num_turns == 6
        assert ep.num_turns == 3

    def test_early_stopping_on_success(self):
        loop, attacker, victim, evaluator = self._make_loop(max_turns=5)
        # Succeed on turn 2 (0-indexed)
        evaluator.score_conversation.side_effect = [
            RewardSignal(0.0, 0.0, 0.0, 0.0),
            RewardSignal(0.0, 0.0, 0.0, 0.0),
            RewardSignal(1.0, 0.5, 0.0, 1.0),  # success at turn 2
            RewardSignal(1.0, 0.5, 0.0, 1.0),  # should not be reached
        ]
        ep = loop._rollout("seed")
        assert ep.succeeded
        assert ep.success_turn == 2
        assert ep.num_turns == 3  # stopped after turn 2

    def test_no_success_runs_all_turns(self):
        loop, _, _, _ = self._make_loop(max_turns=4)
        ep = loop._rollout("seed")
        assert not ep.succeeded
        assert ep.num_turns == 4

    def test_run_calls_fine_tune_multiturn(self):
        loop, attacker, _, _ = self._make_loop()
        config = MultiTurnRLConfig(max_turns=2, episodes=3, log_every=999)
        loop.config = config
        loop.run(["seed1", "seed2"])
        # fine_tune_multiturn called once per RL episode (3 times)
        assert attacker.fine_tune_multiturn.call_count == 3

    def test_fine_tune_receives_correct_episode_count(self):
        loop, attacker, _, _ = self._make_loop()
        config = MultiTurnRLConfig(max_turns=2, episodes=1, log_every=999)
        loop.config = config
        loop.run(["seed1", "seed2", "seed3"])
        episodes_passed = attacker.fine_tune_multiturn.call_args[0][0]
        assert len(episodes_passed) == 3  # one episode per seed

    def test_episode_final_reward_is_last_per_turn_reward(self):
        loop, _, _, evaluator = self._make_loop(max_turns=2)
        last_signal = RewardSignal(0.5, 0.2, 0.0, 0.5)
        evaluator.score_conversation.side_effect = [
            RewardSignal(0.0, 0.0, 0.0, 0.0),
            last_signal,
        ]
        ep = loop._rollout("seed")
        assert ep.final_reward.total == pytest.approx(0.5)
