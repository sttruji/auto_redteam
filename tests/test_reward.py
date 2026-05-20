import pytest
from auto_trojan.reward.model import RewardModel, RewardWeights


def test_reward_composite():
    model = RewardModel(RewardWeights(eval=1.0, affirmative_likelihood=0.5, refusal_penalty=0.3))
    signal = model.compute(eval_score=1.0, affirmative_likelihood=0.8, refusal_penalty=0.0)
    assert signal.total == pytest.approx(1.0 * 1.0 + 0.5 * 0.8 - 0.3 * 0.0)


def test_reward_refusal_penalty_reduces_total():
    model = RewardModel()
    no_refusal = model.compute(1.0, 0.5, 0.0)
    with_refusal = model.compute(1.0, 0.5, 1.0)
    assert with_refusal.total < no_refusal.total


def test_reward_failed_attack():
    model = RewardModel()
    signal = model.compute(eval_score=0.0, affirmative_likelihood=0.0, refusal_penalty=1.0)
    assert signal.total < 0
