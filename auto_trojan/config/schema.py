from __future__ import annotations

from pydantic import BaseModel, Field


class VictimConfig(BaseModel):
    adapter: str = "huggingface"
    model_id: str
    device: str = "cuda"


class AttackerConfig(BaseModel):
    model_id: str
    device: str = "cuda"
    k: int = Field(default=10, ge=1)
    # LoRA / QLoRA
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])
    use_4bit: bool = False
    # Generation
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 0.9
    # Training
    learning_rate: float = 1e-4
    baseline_momentum: float = 0.9
    checkpoint_every: int = 10


class RewardWeightsConfig(BaseModel):
    eval: float = 1.0
    affirmative_likelihood: float = 0.5
    refusal_penalty: float = 0.3


class EvaluatorConfig(BaseModel):
    type: str = "default"
    weights: RewardWeightsConfig = Field(default_factory=RewardWeightsConfig)


class RLConfig(BaseModel):
    episodes: int = 500
    seed: int = 42
    log_every: int = 10
    checkpoint_dir: str = "checkpoints"


class MultiTurnAttackerConfig(AttackerConfig):
    gamma: float = 0.95
    value_head_dropout: float = 0.1
    value_loss_coeff: float = 0.5
    value_lr: float = 1e-4
    max_context_tokens: int = 2048


class MultiTurnRLConfig(RLConfig):
    max_turns: int = 6


class ExperimentConfig(BaseModel):
    victim: VictimConfig
    attacker: AttackerConfig
    evaluator: EvaluatorConfig = Field(default_factory=EvaluatorConfig)
    rl: RLConfig = Field(default_factory=RLConfig)


class MultiTurnExperimentConfig(BaseModel):
    victim: VictimConfig
    attacker: MultiTurnAttackerConfig
    evaluator: EvaluatorConfig = Field(default_factory=EvaluatorConfig)
    rl: MultiTurnRLConfig = Field(default_factory=MultiTurnRLConfig)
