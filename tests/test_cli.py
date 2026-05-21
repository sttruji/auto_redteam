"""
Tests for the CLI — argument parsing, factory functions, and wiring.
No real models are instantiated; all adapters and agents are mocked.
"""
from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent
from unittest.mock import MagicMock, patch

import pytest

from auto_trojan.cli import (
    _build_evaluator,
    _build_multiturn_loop,
    _build_parser,
    _build_single_turn_loop,
    _build_victim,
    _run,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_cfg(yaml_text: str) -> object:
    from omegaconf import OmegaConf
    return OmegaConf.create(yaml_text)


def _minimal_cfg(mode: str = "single") -> object:
    base = dedent("""\
        seed_prompts:
          - "test prompt"
        victim:
          adapter: huggingface
          model_id: fake/model
          device: cpu
        attacker:
          model_id: fake/attacker
          device: cpu
          k: 2
        evaluator:
          type: default
          weights:
            eval: 1.0
            affirmative_likelihood: 0.5
            refusal_penalty: 0.3
        rl:
          episodes: 1
          seed: 0
          log_every: 1
          checkpoint_dir: checkpoints
    """)
    if mode == "multi":
        base += dedent("""\
              max_turns: 3
              gamma: 0.9
              value_head_dropout: 0.0
              value_loss_coeff: 0.5
              value_lr: 1e-4
              max_context_tokens: 512
        """)
    return _load_cfg(base)


# ---------------------------------------------------------------------------
# Parser structure
# ---------------------------------------------------------------------------

class TestParser:
    def test_run_subcommand_exists(self):
        parser = _build_parser()
        args = parser.parse_args(["run", "--config", "cfg.yaml"])
        assert args.command == "run"

    def test_default_mode_is_single_turn(self):
        parser = _build_parser()
        args = parser.parse_args(["run", "--config", "cfg.yaml"])
        assert args.mode == "single-turn"

    def test_mode_multi_turn(self):
        parser = _build_parser()
        args = parser.parse_args(["run", "--config", "cfg.yaml", "--mode", "multi-turn"])
        assert args.mode == "multi-turn"

    def test_seed_prompts_parsed_as_list(self):
        parser = _build_parser()
        args = parser.parse_args(["run", "--config", "cfg.yaml", "--seed-prompts", "a", "b"])
        assert args.seed_prompts == ["a", "b"]

    def test_checkpoint_optional(self):
        parser = _build_parser()
        args = parser.parse_args(["run", "--config", "cfg.yaml"])
        assert args.checkpoint is None

    def test_checkpoint_parsed(self):
        parser = _build_parser()
        args = parser.parse_args(["run", "--config", "cfg.yaml", "--checkpoint", "/path/to/ckpt"])
        assert args.checkpoint == "/path/to/ckpt"

    def test_config_required(self):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["run"])


# ---------------------------------------------------------------------------
# _build_evaluator
# ---------------------------------------------------------------------------

class TestBuildEvaluator:
    def test_returns_default_evaluator(self):
        from auto_trojan.evaluator.default import DefaultEvaluator
        cfg = _minimal_cfg()
        ev = _build_evaluator(cfg)
        assert isinstance(ev, DefaultEvaluator)

    def test_weights_passed_through(self):
        cfg = _minimal_cfg()
        ev = _build_evaluator(cfg)
        assert ev.reward_model.weights.eval == pytest.approx(1.0)
        assert ev.reward_model.weights.affirmative_likelihood == pytest.approx(0.5)

    def test_unknown_evaluator_type_exits(self):
        cfg = _load_cfg("evaluator:\n  type: unknown\n  weights: {}\n")
        with pytest.raises(SystemExit):
            _build_evaluator(cfg)


# ---------------------------------------------------------------------------
# _build_victim
# ---------------------------------------------------------------------------

class TestBuildVictim:
    def test_huggingface_adapter(self):
        cfg = _minimal_cfg()
        with patch("auto_trojan.victim.adapters.huggingface.HuggingFaceVictim") as MockHF:
            MockHF.return_value = MagicMock()
            victim = _build_victim(cfg)
            MockHF.assert_called_once_with(model_id="fake/model", device="cpu")

    def test_openai_adapter(self):
        cfg = _load_cfg(dedent("""\
            victim:
              adapter: openai
              model: gpt-4o
        """))
        with patch("auto_trojan.victim.adapters.openai.OpenAIVictim") as MockOAI:
            MockOAI.return_value = MagicMock()
            victim = _build_victim(cfg)
            MockOAI.assert_called_once_with(model="gpt-4o")

    def test_unknown_adapter_exits(self):
        cfg = _load_cfg("victim:\n  adapter: unknown\n  model_id: x\n")
        with pytest.raises(SystemExit):
            _build_victim(cfg)

    def test_adapter_key_not_passed_to_constructor(self):
        cfg = _minimal_cfg()
        with patch("auto_trojan.victim.adapters.huggingface.HuggingFaceVictim") as MockHF:
            MockHF.return_value = MagicMock()
            _build_victim(cfg)
            call_kwargs = MockHF.call_args[1]
            assert "adapter" not in call_kwargs


# ---------------------------------------------------------------------------
# _build_single_turn_loop
# ---------------------------------------------------------------------------

class TestBuildSingleTurnLoop:
    def test_returns_rl_loop(self):
        from auto_trojan.rl.loop import RLLoop
        cfg = _minimal_cfg()
        attacker = MagicMock()
        victim = MagicMock()
        evaluator = MagicMock()
        loop = _build_single_turn_loop(attacker, victim, evaluator, cfg)
        assert isinstance(loop, RLLoop)

    def test_k_from_attacker_config(self):
        cfg = _minimal_cfg()
        loop = _build_single_turn_loop(MagicMock(), MagicMock(), MagicMock(), cfg)
        assert loop.config.k == 2

    def test_episodes_from_rl_config(self):
        cfg = _minimal_cfg()
        loop = _build_single_turn_loop(MagicMock(), MagicMock(), MagicMock(), cfg)
        assert loop.config.episodes == 1

    def test_components_wired(self):
        cfg = _minimal_cfg()
        attacker, victim, evaluator = MagicMock(), MagicMock(), MagicMock()
        loop = _build_single_turn_loop(attacker, victim, evaluator, cfg)
        assert loop.attacker is attacker
        assert loop.victim is victim
        assert loop.evaluator is evaluator


# ---------------------------------------------------------------------------
# _build_multiturn_loop
# ---------------------------------------------------------------------------

class TestBuildMultiTurnLoop:
    def test_returns_multiturn_rl_loop(self):
        from auto_trojan.rl.multiturn_loop import MultiTurnRLLoop
        cfg = _load_cfg(dedent("""\
            rl:
              episodes: 2
              seed: 1
              log_every: 1
              checkpoint_dir: ckpts
              max_turns: 4
        """))
        loop = _build_multiturn_loop(MagicMock(), MagicMock(), MagicMock(), cfg)
        assert isinstance(loop, MultiTurnRLLoop)

    def test_max_turns_from_config(self):
        cfg = _load_cfg("rl:\n  max_turns: 5\n  episodes: 1\n  seed: 0\n  log_every: 1\n  checkpoint_dir: ckpts\n")
        loop = _build_multiturn_loop(MagicMock(), MagicMock(), MagicMock(), cfg)
        assert loop.config.max_turns == 5

    def test_components_wired(self):
        cfg = _load_cfg("rl:\n  max_turns: 3\n  episodes: 1\n  seed: 0\n  log_every: 1\n  checkpoint_dir: ckpts\n")
        attacker, victim, evaluator = MagicMock(), MagicMock(), MagicMock()
        loop = _build_multiturn_loop(attacker, victim, evaluator, cfg)
        assert loop.attacker is attacker
        assert loop.victim is victim
        assert loop.evaluator is evaluator


# ---------------------------------------------------------------------------
# _run — end-to-end wiring with everything mocked
# ---------------------------------------------------------------------------

class TestRun:
    def _write_config(self, tmp_path: Path, mode: str = "single") -> Path:
        content = dedent(f"""\
            seed_prompts:
              - "test prompt"
            victim:
              adapter: huggingface
              model_id: fake/model
              device: cpu
            attacker:
              model_id: fake/attacker
              device: cpu
              k: 1
              lora_rank: 4
              lora_alpha: 8
              lora_dropout: 0.0
              use_4bit: false
              max_new_tokens: 10
              temperature: 1.0
              top_p: 0.9
              learning_rate: 0.0001
              baseline_momentum: 0.9
              checkpoint_every: 100
              gamma: 0.9
              value_head_dropout: 0.0
              value_loss_coeff: 0.5
              value_lr: 0.0001
              max_context_tokens: 128
            evaluator:
              type: default
              weights:
                eval: 1.0
                affirmative_likelihood: 0.5
                refusal_penalty: 0.3
            rl:
              episodes: 1
              seed: 0
              log_every: 1
              checkpoint_dir: {tmp_path}/ckpts
              max_turns: 2
        """)
        cfg_path = tmp_path / "test.yaml"
        cfg_path.write_text(content)
        return cfg_path

    def _make_args(self, cfg_path: Path, mode: str = "single-turn",
                   seed_prompts: list[str] | None = None) -> object:
        import argparse
        ns = argparse.Namespace(
            command="run",
            config=str(cfg_path),
            mode=mode,
            seed_prompts=seed_prompts,
            checkpoint=None,
        )
        return ns

    def test_single_turn_calls_loop_run(self, tmp_path):
        cfg_path = self._write_config(tmp_path)
        args = self._make_args(cfg_path, mode="single-turn")

        mock_loop = MagicMock()
        with (
            patch("auto_trojan.cli._build_victim", return_value=MagicMock()),
            patch("auto_trojan.cli._build_evaluator", return_value=MagicMock()),
            patch("auto_trojan.cli._build_single_turn_attacker", return_value=MagicMock()),
            patch("auto_trojan.cli._build_single_turn_loop", return_value=mock_loop),
        ):
            _run(args)

        mock_loop.run.assert_called_once_with(["test prompt"])

    def test_multi_turn_calls_loop_run(self, tmp_path):
        cfg_path = self._write_config(tmp_path, mode="multi")
        args = self._make_args(cfg_path, mode="multi-turn")

        mock_loop = MagicMock()
        with (
            patch("auto_trojan.cli._build_victim", return_value=MagicMock()),
            patch("auto_trojan.cli._build_evaluator", return_value=MagicMock()),
            patch("auto_trojan.cli._build_multiturn_attacker", return_value=MagicMock()),
            patch("auto_trojan.cli._build_multiturn_loop", return_value=mock_loop),
        ):
            _run(args)

        mock_loop.run.assert_called_once_with(["test prompt"])

    def test_cli_seed_prompts_override_config(self, tmp_path):
        cfg_path = self._write_config(tmp_path)
        args = self._make_args(cfg_path, seed_prompts=["override prompt"])

        mock_loop = MagicMock()
        with (
            patch("auto_trojan.cli._build_victim", return_value=MagicMock()),
            patch("auto_trojan.cli._build_evaluator", return_value=MagicMock()),
            patch("auto_trojan.cli._build_single_turn_attacker", return_value=MagicMock()),
            patch("auto_trojan.cli._build_single_turn_loop", return_value=mock_loop),
        ):
            _run(args)

        mock_loop.run.assert_called_once_with(["override prompt"])

    def test_no_seed_prompts_exits(self, tmp_path):
        content = dedent("""\
            victim:
              adapter: huggingface
              model_id: x
            attacker:
              model_id: x
            evaluator:
              type: default
              weights: {}
            rl:
              episodes: 1
              seed: 0
              log_every: 1
              checkpoint_dir: ckpts
        """)
        cfg_path = tmp_path / "empty.yaml"
        cfg_path.write_text(content)
        args = self._make_args(cfg_path, seed_prompts=None)

        with pytest.raises(SystemExit):
            _run(args)

    def test_single_turn_uses_single_turn_attacker(self, tmp_path):
        cfg_path = self._write_config(tmp_path)
        args = self._make_args(cfg_path, mode="single-turn")

        mock_build_st = MagicMock(return_value=MagicMock())
        mock_build_mt = MagicMock(return_value=MagicMock())
        with (
            patch("auto_trojan.cli._build_victim", return_value=MagicMock()),
            patch("auto_trojan.cli._build_evaluator", return_value=MagicMock()),
            patch("auto_trojan.cli._build_single_turn_attacker", mock_build_st),
            patch("auto_trojan.cli._build_multiturn_attacker", mock_build_mt),
            patch("auto_trojan.cli._build_single_turn_loop", return_value=MagicMock()),
        ):
            _run(args)

        mock_build_st.assert_called_once()
        mock_build_mt.assert_not_called()

    def test_multi_turn_uses_multiturn_attacker(self, tmp_path):
        cfg_path = self._write_config(tmp_path)
        args = self._make_args(cfg_path, mode="multi-turn")

        mock_build_st = MagicMock(return_value=MagicMock())
        mock_build_mt = MagicMock(return_value=MagicMock())
        with (
            patch("auto_trojan.cli._build_victim", return_value=MagicMock()),
            patch("auto_trojan.cli._build_evaluator", return_value=MagicMock()),
            patch("auto_trojan.cli._build_single_turn_attacker", mock_build_st),
            patch("auto_trojan.cli._build_multiturn_attacker", mock_build_mt),
            patch("auto_trojan.cli._build_multiturn_loop", return_value=MagicMock()),
        ):
            _run(args)

        mock_build_mt.assert_called_once()
        mock_build_st.assert_not_called()
