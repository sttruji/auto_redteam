from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING, Any

from rich.console import Console

if TYPE_CHECKING:
    from auto_trojan.evaluator.base import EvaluationAgent
    from auto_trojan.rl.loop import RLLoop
    from auto_trojan.rl.multiturn_loop import MultiTurnRLLoop
    from auto_trojan.victim.base import VictimLLM

console = Console()


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "run":
        _run(args)
    else:
        parser.print_help()
        sys.exit(1)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="auto-trojan",
        description="Modular LLM red-teaming framework",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_p = subparsers.add_parser("run", help="Run a red-team campaign")
    run_p.add_argument("--config", required=True, help="Path to experiment YAML config")
    run_p.add_argument(
        "--mode",
        choices=["single-turn", "multi-turn"],
        default="single-turn",
        help="Attack mode (default: single-turn)",
    )
    run_p.add_argument(
        "--seed-prompts",
        nargs="+",
        metavar="PROMPT",
        help="Seed prompts to red-team with; overrides seed_prompts in config",
    )
    run_p.add_argument(
        "--checkpoint",
        metavar="PATH",
        help="Resume attacker from a saved LoRA checkpoint directory",
    )
    return parser


def _run(args: argparse.Namespace) -> None:
    import torch
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(args.config)
    console.rule("[bold]auto-trojan[/bold]")
    console.print(f"Config  : [cyan]{args.config}[/cyan]")
    console.print(f"Mode    : [cyan]{args.mode}[/cyan]")
    console.print(OmegaConf.to_yaml(cfg))

    seed = int(cfg.rl.get("seed", 42))
    torch.manual_seed(seed)
    console.print(f"Seed    : {seed}")

    seed_prompts: list[str] = list(args.seed_prompts or [])
    if not seed_prompts:
        from omegaconf import OmegaConf as _OC
        raw = cfg.get("seed_prompts", None)
        if raw is not None:
            seed_prompts = list(_OC.to_container(raw, resolve=True))  # type: ignore[arg-type]
    if not seed_prompts:
        console.print(
            "[red]Error: no seed prompts provided. "
            "Use --seed-prompts or add a seed_prompts list to the config.[/red]"
        )
        sys.exit(1)

    n = len(seed_prompts)
    preview = seed_prompts[:2]
    suffix = "..." if n > 2 else ""
    console.print(f"Prompts : {n} seed prompt(s) — {preview}{suffix}")

    victim = _build_victim(cfg)
    evaluator = _build_evaluator(cfg)

    if args.mode == "multi-turn":
        attacker = _build_multiturn_attacker(cfg, args.checkpoint)
        loop: RLLoop | MultiTurnRLLoop = _build_multiturn_loop(attacker, victim, evaluator, cfg)
    else:
        attacker = _build_single_turn_attacker(cfg, args.checkpoint)
        loop = _build_single_turn_loop(attacker, victim, evaluator, cfg)

    console.rule(f"[bold green]Starting {args.mode} campaign[/bold green]")
    loop.run(seed_prompts)
    console.rule("[bold green]Campaign complete[/bold green]")


# ---------------------------------------------------------------------------
# Factory functions — each defers its heavy imports so the CLI is fast to start
# ---------------------------------------------------------------------------

def _build_victim(cfg: Any) -> VictimLLM:
    from omegaconf import OmegaConf

    victim_dict: dict[str, Any] = OmegaConf.to_container(cfg.victim, resolve=True)  # type: ignore[arg-type]
    adapter = victim_dict.pop("adapter")

    if adapter == "huggingface":
        from auto_trojan.victim.adapters.huggingface import HuggingFaceVictim
        return HuggingFaceVictim(**victim_dict)
    elif adapter == "openai":
        from auto_trojan.victim.adapters.openai import OpenAIVictim
        return OpenAIVictim(**victim_dict)
    else:
        console.print(f"[red]Unknown victim adapter: {adapter!r}. Choose 'huggingface' or 'openai'.[/red]")
        sys.exit(1)


def _build_evaluator(cfg: Any) -> EvaluationAgent:
    from omegaconf import OmegaConf
    from auto_trojan.evaluator.default import DefaultEvaluator
    from auto_trojan.reward.model import RewardWeights

    ev_dict: dict[str, Any] = OmegaConf.to_container(cfg.evaluator, resolve=True)  # type: ignore[arg-type]
    weights = RewardWeights(**ev_dict.get("weights", {}))
    ev_type = ev_dict.get("type", "default")

    if ev_type == "default":
        return DefaultEvaluator(weights)
    else:
        console.print(f"[red]Unknown evaluator type: {ev_type!r}.[/red]")
        sys.exit(1)


def _build_single_turn_attacker(cfg: Any, checkpoint_path: str | None = None) -> Any:
    from omegaconf import OmegaConf
    from auto_trojan.attacker.lora_attacker import LoRAAttackerAgent, LoRAAttackerConfig

    a: dict[str, Any] = OmegaConf.to_container(cfg.attacker, resolve=True)  # type: ignore[arg-type]
    rl: dict[str, Any] = OmegaConf.to_container(cfg.rl, resolve=True)  # type: ignore[arg-type]

    lora_cfg = LoRAAttackerConfig(
        model_id=a["model_id"],
        device=a.get("device", "cuda"),
        lora_rank=a.get("lora_rank", 16),
        lora_alpha=a.get("lora_alpha", 32),
        lora_dropout=a.get("lora_dropout", 0.05),
        target_modules=a.get("target_modules", ["q_proj", "v_proj"]),
        use_4bit=a.get("use_4bit", False),
        max_new_tokens=a.get("max_new_tokens", 256),
        temperature=a.get("temperature", 1.0),
        top_p=a.get("top_p", 0.9),
        learning_rate=a.get("learning_rate", 1e-4),
        baseline_momentum=a.get("baseline_momentum", 0.9),
        checkpoint_every=a.get("checkpoint_every", 10),
        checkpoint_dir=rl.get("checkpoint_dir", "checkpoints"),
    )
    return LoRAAttackerAgent(lora_cfg, checkpoint_path=checkpoint_path)


def _build_multiturn_attacker(cfg: Any, checkpoint_path: str | None = None) -> Any:
    from omegaconf import OmegaConf
    from auto_trojan.attacker.multiturn_attacker import (
        MultiTurnAttackerAgent,
        MultiTurnAttackerConfig,
    )

    a: dict[str, Any] = OmegaConf.to_container(cfg.attacker, resolve=True)  # type: ignore[arg-type]
    rl: dict[str, Any] = OmegaConf.to_container(cfg.rl, resolve=True)  # type: ignore[arg-type]

    mt_cfg = MultiTurnAttackerConfig(
        model_id=a["model_id"],
        device=a.get("device", "cuda"),
        lora_rank=a.get("lora_rank", 16),
        lora_alpha=a.get("lora_alpha", 32),
        lora_dropout=a.get("lora_dropout", 0.05),
        target_modules=a.get("target_modules", ["q_proj", "v_proj"]),
        use_4bit=a.get("use_4bit", False),
        max_new_tokens=a.get("max_new_tokens", 256),
        temperature=a.get("temperature", 1.0),
        top_p=a.get("top_p", 0.9),
        learning_rate=a.get("learning_rate", 1e-4),
        baseline_momentum=a.get("baseline_momentum", 0.9),
        checkpoint_every=a.get("checkpoint_every", 10),
        checkpoint_dir=rl.get("checkpoint_dir", "checkpoints"),
        gamma=a.get("gamma", 0.95),
        value_head_dropout=a.get("value_head_dropout", 0.1),
        value_loss_coeff=a.get("value_loss_coeff", 0.5),
        value_lr=a.get("value_lr", 1e-4),
        max_context_tokens=a.get("max_context_tokens", 2048),
    )
    return MultiTurnAttackerAgent(mt_cfg, checkpoint_path=checkpoint_path)


def _build_single_turn_loop(
    attacker: Any, victim: Any, evaluator: Any, cfg: Any
) -> RLLoop:
    from omegaconf import OmegaConf
    from auto_trojan.rl.loop import RLConfig as LoopRLConfig, RLLoop

    a: dict[str, Any] = OmegaConf.to_container(cfg.attacker, resolve=True)  # type: ignore[arg-type]
    rl: dict[str, Any] = OmegaConf.to_container(cfg.rl, resolve=True)  # type: ignore[arg-type]

    loop_cfg = LoopRLConfig(
        k=a.get("k", 10),
        episodes=rl.get("episodes", 500),
        seed=rl.get("seed", 42),
        log_every=rl.get("log_every", 10),
        checkpoint_dir=rl.get("checkpoint_dir", "checkpoints"),
    )
    return RLLoop(attacker, victim, evaluator, loop_cfg)


def _build_multiturn_loop(
    attacker: Any, victim: Any, evaluator: Any, cfg: Any
) -> MultiTurnRLLoop:
    from omegaconf import OmegaConf
    from auto_trojan.rl.multiturn_loop import MultiTurnRLConfig as LoopMTConfig, MultiTurnRLLoop

    rl: dict[str, Any] = OmegaConf.to_container(cfg.rl, resolve=True)  # type: ignore[arg-type]

    loop_cfg = LoopMTConfig(
        max_turns=rl.get("max_turns", 6),
        episodes=rl.get("episodes", 500),
        seed=rl.get("seed", 42),
        log_every=rl.get("log_every", 10),
        checkpoint_dir=rl.get("checkpoint_dir", "checkpoints"),
    )
    return MultiTurnRLLoop(attacker, victim, evaluator, loop_cfg)
