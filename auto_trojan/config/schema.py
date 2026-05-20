from pydantic import BaseModel, Field


class VictimConfig(BaseModel):
    adapter: str = "huggingface"
    model_id: str
    device: str = "cuda"


class AttackerConfig(BaseModel):
    model_id: str
    device: str = "cuda"
    k: int = Field(default=10, ge=1)


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
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


class ExperimentConfig(BaseModel):
    victim: VictimConfig
    attacker: AttackerConfig
    evaluator: EvaluatorConfig = Field(default_factory=EvaluatorConfig)
    rl: RLConfig = Field(default_factory=RLConfig)
