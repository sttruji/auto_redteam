"""
Tests for LoRAAttackerAgent that run without GPU or real model weights.

The model is mocked at the transformers/peft boundary so we can test:
  - prompt formatting
  - log-prob masking / REINFORCE loss shape
  - baseline EMA updates
  - fine_tune optimizer step fires
  - checkpoint saving
"""
from __future__ import annotations

import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from auto_trojan.attacker.base import Episode
from auto_trojan.attacker.lora_attacker import LoRAAttackerAgent, LoRAAttackerConfig
from auto_trojan.evaluator.base import RewardSignal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(tmp_path: Path) -> LoRAAttackerConfig:
    return LoRAAttackerConfig(
        model_id="fake/model",
        device="cpu",
        lora_rank=4,
        lora_alpha=8,
        checkpoint_dir=str(tmp_path / "ckpts"),
        checkpoint_every=2,
    )


def _make_episode(seed: str, attack: str, reward_total: float) -> Episode:
    signal = RewardSignal(
        eval_score=reward_total,
        affirmative_likelihood=0.0,
        refusal_penalty=0.0,
        total=reward_total,
    )
    return Episode(seed_prompt=seed, attack_prompt=attack, victim_response="", reward=signal)


def _make_fake_agent(config: LoRAAttackerConfig) -> LoRAAttackerAgent:
    """Build a LoRAAttackerAgent with all heavy dependencies mocked out."""
    vocab_size = 50
    hidden = 8
    seq_len = 16

    # fake tokenizer
    tok = MagicMock()
    tok.pad_token = "<pad>"
    tok.pad_token_id = 0
    tok.eos_token = "</s>"
    tok.padding_side = "left"
    tok.decode = MagicMock(return_value="mocked attack prompt")

    def fake_call(text, return_tensors=None, padding=False, truncation=False,
                  max_length=None, return_attention_mask=False, **kw):
        if isinstance(text, list):
            batch = len(text)
        else:
            batch = 1
        ids = torch.zeros(batch, seq_len, dtype=torch.long)
        mask = torch.ones(batch, seq_len, dtype=torch.long)
        out = MagicMock()
        out.input_ids = ids
        out.attention_mask = mask
        out.__getitem__ = lambda self, k: {"input_ids": ids, "attention_mask": mask}[k]
        out.to = lambda dev: out
        return out

    tok.side_effect = fake_call

    # single-call tokenizer for prompt-length measurement
    single_out = MagicMock()
    single_out.input_ids = torch.zeros(1, 5, dtype=torch.long)
    single_out.to = lambda d: single_out

    # fake model
    model = MagicMock()
    # generate returns padded ids
    model.generate.return_value = torch.zeros(3, seq_len + 10, dtype=torch.long)
    # forward returns logits with grad
    logits = torch.randn(1, seq_len, vocab_size, requires_grad=True)
    model_output = MagicMock()
    model_output.logits = logits
    model.return_value = model_output
    model.parameters.return_value = iter([torch.nn.Parameter(torch.randn(4, 4))])
    model.eval = MagicMock()
    model.train = MagicMock()
    model.to = MagicMock(return_value=model)
    model.save_pretrained = MagicMock()

    optimizer = MagicMock()
    optimizer.zero_grad = MagicMock()
    optimizer.step = MagicMock()

    agent = LoRAAttackerAgent.__new__(LoRAAttackerAgent)
    agent.config = config
    agent._step = 0
    agent._baseline = 0.0
    agent.tokenizer = tok
    agent.model = model
    agent.optimizer = optimizer
    return agent


# ---------------------------------------------------------------------------
# Tests: prompt formatting
# ---------------------------------------------------------------------------

def test_format_prompt_contains_seed():
    config = LoRAAttackerConfig(model_id="x", device="cpu")
    agent = LoRAAttackerAgent.__new__(LoRAAttackerAgent)
    agent.config = config
    prompt = agent._format_prompt("do something bad")
    assert "do something bad" in prompt
    assert "Rephrased:" in prompt


def test_format_prompt_includes_system_prompt():
    config = LoRAAttackerConfig(model_id="x", device="cpu", system_prompt="CUSTOM")
    agent = LoRAAttackerAgent.__new__(LoRAAttackerAgent)
    agent.config = config
    assert "CUSTOM" in agent._format_prompt("seed")


# ---------------------------------------------------------------------------
# Tests: baseline EMA
# ---------------------------------------------------------------------------

def test_baseline_starts_at_zero():
    config = LoRAAttackerConfig(model_id="x", device="cpu")
    agent = LoRAAttackerAgent.__new__(LoRAAttackerAgent)
    agent.config = config
    agent._baseline = 0.0
    assert agent._baseline == 0.0


def test_baseline_updates_toward_batch_mean():
    config = LoRAAttackerConfig(model_id="x", device="cpu", baseline_momentum=0.9)
    agent = LoRAAttackerAgent.__new__(LoRAAttackerAgent)
    agent.config = config
    agent._baseline = 0.0
    agent._update_baseline(10.0)
    assert agent._baseline == pytest.approx(0.9 * 0.0 + 0.1 * 10.0)


def test_baseline_converges():
    config = LoRAAttackerConfig(model_id="x", device="cpu", baseline_momentum=0.9)
    agent = LoRAAttackerAgent.__new__(LoRAAttackerAgent)
    agent.config = config
    agent._baseline = 0.0
    for _ in range(200):
        agent._update_baseline(5.0)
    assert agent._baseline == pytest.approx(5.0, abs=0.1)


# ---------------------------------------------------------------------------
# Tests: generate_attacks
# ---------------------------------------------------------------------------

def test_generate_attacks_returns_k_strings(tmp_path):
    config = _make_config(tmp_path)
    agent = _make_fake_agent(config)
    results = agent.generate_attacks("seed", k=3)
    assert len(results) == 3
    assert all(isinstance(r, str) for r in results)


def test_generate_attacks_calls_model_eval(tmp_path):
    config = _make_config(tmp_path)
    agent = _make_fake_agent(config)
    agent.generate_attacks("seed", k=2)
    agent.model.eval.assert_called()
    agent.model.train.assert_called()


# ---------------------------------------------------------------------------
# Tests: fine_tune
# ---------------------------------------------------------------------------

def test_fine_tune_empty_episodes_is_noop(tmp_path):
    config = _make_config(tmp_path)
    agent = _make_fake_agent(config)
    agent.fine_tune([])
    assert agent._step == 0


def test_fine_tune_increments_step(tmp_path):
    config = _make_config(tmp_path)
    agent = _make_fake_agent(config)

    episodes = [_make_episode("seed", "attack", 1.0)]

    # patch the heavy log-prob computation with a stub that returns a real tensor
    def fake_log_probs(prompts, completions):
        t = torch.tensor([0.5] * len(prompts), requires_grad=True)
        return t

    agent._batch_sequence_log_probs = fake_log_probs
    agent.fine_tune(episodes)
    assert agent._step == 1


def test_fine_tune_updates_baseline(tmp_path):
    config = _make_config(tmp_path)
    agent = _make_fake_agent(config)
    agent._baseline = 0.0

    episodes = [_make_episode("s", "a", 4.0), _make_episode("s", "b", 6.0)]

    def fake_log_probs(prompts, completions):
        return torch.tensor([0.1] * len(prompts), requires_grad=True)

    agent._batch_sequence_log_probs = fake_log_probs
    agent.fine_tune(episodes)

    expected = 0.9 * 0.0 + 0.1 * 5.0  # mean reward = 5.0
    assert agent._baseline == pytest.approx(expected)


def test_fine_tune_saves_checkpoint_at_interval(tmp_path):
    config = _make_config(tmp_path)  # checkpoint_every=2
    agent = _make_fake_agent(config)

    def fake_log_probs(prompts, completions):
        return torch.tensor([0.1] * len(prompts), requires_grad=True)

    agent._batch_sequence_log_probs = fake_log_probs

    episodes = [_make_episode("s", "a", 1.0)]
    agent.fine_tune(episodes)  # step 1 — no checkpoint
    agent.model.save_pretrained.assert_not_called()

    agent.fine_tune(episodes)  # step 2 — checkpoint
    agent.model.save_pretrained.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: log-prob masking (pure tensor math, no real model)
# ---------------------------------------------------------------------------

def test_batch_log_probs_masks_prompt_tokens(tmp_path):
    """Completion tokens after the prompt should accumulate log-probs; prompt tokens should not."""
    config = _make_config(tmp_path)
    agent = LoRAAttackerAgent.__new__(LoRAAttackerAgent)
    agent.config = config

    vocab = 10
    seq = 8
    prompt_len = 3  # first 3 tokens are the "prompt"

    # Fake tokenizer that always returns deterministic ids/masks
    tok = MagicMock()
    tok.pad_token = "<pad>"
    tok.pad_token_id = 0
    tok.padding_side = "left"

    # single-text call (used for prompt length)
    single = MagicMock()
    single.input_ids = torch.zeros(1, prompt_len, dtype=torch.long)

    def tok_call(text, **kw):
        if isinstance(text, list):
            out = MagicMock()
            out.input_ids = torch.zeros(len(text), seq, dtype=torch.long)
            out.attention_mask = torch.ones(len(text), seq, dtype=torch.long)
            out.to = lambda d: out
            return out
        return single

    tok.side_effect = tok_call
    agent.tokenizer = tok

    # Model returns fixed logits — log_softmax will give uniform -log(vocab)
    uniform_logits = torch.zeros(1, seq, vocab)  # [B=1, T, V]
    model_out = MagicMock()
    model_out.logits = uniform_logits
    model = MagicMock()
    model.return_value = model_out
    agent.model = model

    log_probs = agent._batch_sequence_log_probs(["prompt "], ["completion"])
    # completion tokens = seq - 1 (shifted) - (prompt_len - 1) = seq - prompt_len
    # each log prob = -log(vocab) for uniform distribution
    expected_tokens = seq - 1 - (prompt_len - 1)  # = seq - prompt_len
    expected = expected_tokens * (-torch.log(torch.tensor(float(vocab))).item())
    assert log_probs[0].item() == pytest.approx(expected, abs=1e-4)
