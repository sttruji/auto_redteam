# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**auto_trojan** is an open-source, modular LLM red-teaming framework. Given a malicious input prompt, an attacker agent generates up to `k` adversarial attack prompts, passes them through a victim LLM, and an evaluation agent scores success. These scores (plus affirmative token likelihood and refusal keyword penalties) form a reward signal used to fine-tune the attacker via RL with QLoRA/LoRA — teaching it a policy that reliably elicits harmful outputs from victim models.

Planned extension: a critic agent that critiques attacker outputs and builds reusable attack strategies on the fly, plus multi-turn conversation attacks across up to `k` turns.

---

## Architecture

```
malicious_prompt
      │
      ▼
 AttackerAgent  ──── generates k candidate attack prompts
      │
      ▼
 VictimLLM (modular, swappable)
      │
      ▼
 EvaluationAgent
      │  ├─ binary success/failure
      │  ├─ affirmative token likelihood score
      │  └─ refusal keyword penalty
      ▼
 RewardModel  ──── RL policy reward signal
      │
      ▼
 FineTuner (QLoRA/LoRA)  ──── adapts AttackerAgent weights
```

### Key Design Principles

- **Victim modularity** — The victim LLM is behind a clean interface so any model (local, API-based, HuggingFace, OpenAI-compatible) can be swapped in with minimal config.
- **Attacker adaptability** — The attacker is a fine-tunable LLM; the RL loop updates it between episodes.
- **Evaluation composability** — The reward signal is a weighted combination of multiple sub-scores; each scorer is independently configurable.
- **Research-first** — Prioritize reproducibility (seed control, logging of all prompts/responses/rewards) over production hardening.

### Planned Module Layout

```
auto_trojan/
├── attacker/          # AttackerAgent: prompt generation + LoRA fine-tuning
├── victim/            # VictimLLM interface + concrete adapters (HF, OpenAI, etc.)
├── evaluator/         # EvaluationAgent + sub-scorers (token likelihood, keyword penalty)
├── reward/            # Reward model composition and normalization
├── rl/                # RL training loop, episode management, policy updates
├── critic/            # (future) CriticAgent + strategy library
├── config/            # YAML/TOML configs for experiments
└── experiments/       # Entry-point scripts for running red-team campaigns
```

---

## Development Commands

> Commands will be added here once the project scaffolding is in place.

```bash
# Install dependencies (expected)
pip install -e ".[dev]"

# Run tests
pytest

# Run a single test
pytest tests/path/to/test_file.py::test_name

# Lint
ruff check .

# Type check
mypy auto_trojan/
```

---

## Reward Signal

The RL reward `r` for a given attack prompt is:

```
r = w_eval * eval_score
  + w_aff  * affirmative_token_likelihood
  - w_ref  * refusal_keyword_penalty
```

Weights (`w_*`) are experiment-configurable. The evaluation agent determines `eval_score` (did the victim produce a harmful/target output?).

---

## Modularity Contract

**VictimLLM** must implement:
```python
def generate(prompt: str, **kwargs) -> str: ...
```

**AttackerAgent** must implement:
```python
def generate_attacks(seed_prompt: str, k: int) -> list[str]: ...
def fine_tune(episodes: list[Episode]) -> None: ...
```

**EvaluationAgent** must implement:
```python
def score(attack_prompt: str, victim_response: str) -> RewardSignal: ...
```

New victim adapters or scorers should satisfy these interfaces without touching core RL logic.

---

## Research Conventions

- Every experiment run logs: seed prompt, all `k` attack prompts, victim responses, per-scorer rewards, final reward, episode number, and model checkpoint path.
- Seeds are always fixed and logged for reproducibility.
- Model checkpoints are saved per RL iteration with metadata (config hash, episode range).
