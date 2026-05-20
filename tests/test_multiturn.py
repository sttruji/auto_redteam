"""
Tests for multi-turn data structures, VictimLLM.generate_with_history,
and EvaluationAgent.score_conversation. No GPU or real models required.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from auto_trojan.attacker.base import MultiTurnEpisode
from auto_trojan.conversation import Conversation, Turn
from auto_trojan.evaluator.base import EvaluationAgent, RewardSignal
from auto_trojan.victim.base import VictimLLM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _signal(total: float = 1.0) -> RewardSignal:
    return RewardSignal(eval_score=total, affirmative_likelihood=0.0, refusal_penalty=0.0, total=total)


def _conv(seed: str = "goal", turns: list[tuple[str, str]] | None = None) -> Conversation:
    c = Conversation(seed_goal=seed)
    for role, content in (turns or []):
        c.add_turn(role, content)  # type: ignore[arg-type]
    return c


# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------

class TestConversation:
    def test_starts_empty(self):
        c = Conversation(seed_goal="goal")
        assert c.is_empty
        assert c.num_turns == 0

    def test_add_turn(self):
        c = _conv(turns=[("attacker", "hi"), ("victim", "hello")])
        assert c.num_turns == 2
        assert not c.is_empty

    def test_last_attacker_turn(self):
        c = _conv(turns=[("attacker", "first"), ("victim", "resp"), ("attacker", "second")])
        assert c.last_attacker_turn() == "second"

    def test_last_victim_response(self):
        c = _conv(turns=[("attacker", "hi"), ("victim", "bye"), ("attacker", "again")])
        assert c.last_victim_response() == "bye"

    def test_last_attacker_turn_none_when_empty(self):
        assert Conversation(seed_goal="x").last_attacker_turn() is None

    def test_last_victim_response_none_when_no_victim(self):
        c = _conv(turns=[("attacker", "hi")])
        assert c.last_victim_response() is None

    def test_attacker_turns_property(self):
        c = _conv(turns=[("attacker", "a"), ("victim", "v"), ("attacker", "b")])
        assert [t.content for t in c.attacker_turns] == ["a", "b"]

    def test_victim_turns_property(self):
        c = _conv(turns=[("attacker", "a"), ("victim", "v1"), ("attacker", "b"), ("victim", "v2")])
        assert [t.content for t in c.victim_turns] == ["v1", "v2"]

    def test_to_messages_role_mapping(self):
        c = _conv(turns=[("attacker", "hello"), ("victim", "hi there")])
        msgs = c.to_messages()
        assert msgs == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]

    def test_to_messages_empty(self):
        assert Conversation(seed_goal="x").to_messages() == []

    def test_seed_goal_not_in_messages(self):
        c = _conv(seed="SECRET GOAL", turns=[("attacker", "msg")])
        for m in c.to_messages():
            assert "SECRET GOAL" not in m["content"]


# ---------------------------------------------------------------------------
# Turn
# ---------------------------------------------------------------------------

class TestTurn:
    def test_turn_role_and_content(self):
        t = Turn(role="attacker", content="test")
        assert t.role == "attacker"
        assert t.content == "test"


# ---------------------------------------------------------------------------
# MultiTurnEpisode
# ---------------------------------------------------------------------------

class TestMultiTurnEpisode:
    def test_succeeded_when_success_turn_set(self):
        c = _conv(turns=[("attacker", "a"), ("victim", "v")])
        ep = MultiTurnEpisode(
            conversation=c,
            per_turn_rewards=[_signal(1.0)],
            final_reward=_signal(1.0),
            success_turn=0,
        )
        assert ep.succeeded is True

    def test_not_succeeded_when_success_turn_none(self):
        c = _conv(turns=[("attacker", "a"), ("victim", "I cannot")])
        ep = MultiTurnEpisode(
            conversation=c,
            per_turn_rewards=[_signal(0.0)],
            final_reward=_signal(0.0),
            success_turn=None,
        )
        assert ep.succeeded is False

    def test_num_turns_matches_per_turn_rewards(self):
        c = _conv(turns=[("attacker", "a"), ("victim", "r1"), ("attacker", "b"), ("victim", "r2")])
        ep = MultiTurnEpisode(
            conversation=c,
            per_turn_rewards=[_signal(0.3), _signal(1.0)],
            final_reward=_signal(1.0),
            success_turn=1,
        )
        assert ep.num_turns == 2

    def test_metadata_defaults_empty(self):
        ep = MultiTurnEpisode(
            conversation=_conv(),
            per_turn_rewards=[],
            final_reward=_signal(0.0),
        )
        assert ep.metadata == {}


# ---------------------------------------------------------------------------
# VictimLLM default generate_with_history (flat-string fallback)
# ---------------------------------------------------------------------------

class ConcreteVictim(VictimLLM):
    """Minimal concrete victim for testing the base class default."""

    def __init__(self) -> None:
        self.last_prompt: str = ""

    def generate(self, prompt: str, **kwargs: object) -> str:
        self.last_prompt = prompt
        return "response"


class TestVictimLLMBase:
    def test_generate_with_history_calls_generate(self):
        victim = ConcreteVictim()
        c = _conv(turns=[("attacker", "hello")])
        result = victim.generate_with_history(c)
        assert result == "response"
        assert "hello" in victim.last_prompt

    def test_format_conversation_includes_all_turns(self):
        c = _conv(turns=[("attacker", "hi"), ("victim", "hey"), ("attacker", "more")])
        formatted = VictimLLM._format_conversation(c)
        assert "User: hi" in formatted
        assert "Assistant: hey" in formatted
        assert "User: more" in formatted

    def test_format_conversation_ends_with_assistant_prompt(self):
        c = _conv(turns=[("attacker", "hi"), ("victim", "hey")])
        formatted = VictimLLM._format_conversation(c)
        assert formatted.endswith("Assistant:")

    def test_generate_with_history_empty_conversation(self):
        victim = ConcreteVictim()
        result = victim.generate_with_history(_conv())
        assert result == "response"
        assert "Assistant:" in victim.last_prompt


# ---------------------------------------------------------------------------
# HuggingFaceVictim.generate_with_history (mocked)
# ---------------------------------------------------------------------------

class TestHuggingFaceVictimMultiTurn:
    def _make_victim(self, has_chat_template: bool = True) -> object:
        from auto_trojan.victim.adapters.huggingface import HuggingFaceVictim

        victim = HuggingFaceVictim.__new__(HuggingFaceVictim)
        victim.device = "cpu"
        victim.generation_kwargs = {}

        tok = MagicMock()
        tok.chat_template = "<template>" if has_chat_template else None
        tok.apply_chat_template = MagicMock(return_value="<formatted>")
        tok.return_value = MagicMock(
            input_ids=__import__("torch").zeros(1, 5, dtype=__import__("torch").long),
            to=lambda d: MagicMock(
                input_ids=__import__("torch").zeros(1, 5, dtype=__import__("torch").long),
                **{"__iter__": lambda s: iter(["input_ids"])}
            ),
        )
        tok.decode = MagicMock(return_value="victim reply")

        model = MagicMock()
        import torch
        model.generate = MagicMock(return_value=torch.zeros(1, 10, dtype=torch.long))

        victim.tokenizer = tok
        victim.model = model
        return victim

    def test_uses_chat_template_when_available(self):
        victim = self._make_victim(has_chat_template=True)
        c = _conv(turns=[("attacker", "hi"), ("victim", "hey"), ("attacker", "more")])
        victim.generate_with_history(c)
        victim.tokenizer.apply_chat_template.assert_called_once()

    def test_skips_chat_template_when_absent(self):
        victim = self._make_victim(has_chat_template=False)
        c = _conv(turns=[("attacker", "hi")])
        victim.generate_with_history(c)
        victim.tokenizer.apply_chat_template.assert_not_called()

    def test_chat_template_receives_correct_messages(self):
        victim = self._make_victim(has_chat_template=True)
        c = _conv(turns=[("attacker", "prompt")])
        victim.generate_with_history(c)
        call_args = victim.tokenizer.apply_chat_template.call_args
        messages = call_args[0][0]
        assert messages == [{"role": "user", "content": "prompt"}]


# ---------------------------------------------------------------------------
# OpenAIVictim.generate_with_history (mocked)
# ---------------------------------------------------------------------------

class TestOpenAIVictimMultiTurn:
    def _make_victim(self) -> object:
        from auto_trojan.victim.adapters.openai import OpenAIVictim

        victim = OpenAIVictim.__new__(OpenAIVictim)
        victim.model = "gpt-4o"
        victim.extra = {}

        client = MagicMock()
        choice = MagicMock()
        choice.message.content = "openai reply"
        client.chat.completions.create.return_value = MagicMock(choices=[choice])
        victim.client = client
        return victim

    def test_passes_messages_list(self):
        victim = self._make_victim()
        c = _conv(turns=[("attacker", "hi"), ("victim", "hey"), ("attacker", "more")])
        result = victim.generate_with_history(c)
        assert result == "openai reply"
        call_kwargs = victim.client.chat.completions.create.call_args[1]
        assert call_kwargs["messages"] == c.to_messages()

    def test_passes_model_name(self):
        victim = self._make_victim()
        victim.generate_with_history(_conv(turns=[("attacker", "x")]))
        call_kwargs = victim.client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4o"


# ---------------------------------------------------------------------------
# EvaluationAgent.score_conversation default
# ---------------------------------------------------------------------------

class ConcreteEvaluator(EvaluationAgent):
    def score(self, attack_prompt: str, victim_response: str) -> RewardSignal:
        success = "sorry" not in victim_response.lower()
        return RewardSignal(
            eval_score=1.0 if success else 0.0,
            affirmative_likelihood=0.0,
            refusal_penalty=0.0,
            total=1.0 if success else 0.0,
        )


class TestEvaluationAgentScoreConversation:
    def test_scores_last_exchange(self):
        ev = ConcreteEvaluator()
        c = _conv(turns=[("attacker", "attack"), ("victim", "Sure, here you go")])
        signal = ev.score_conversation(c)
        assert signal.eval_score == 1.0

    def test_refusal_in_last_turn(self):
        ev = ConcreteEvaluator()
        c = _conv(turns=[("attacker", "attack"), ("victim", "Sorry I cannot help")])
        signal = ev.score_conversation(c)
        assert signal.eval_score == 0.0

    def test_empty_conversation_returns_zero(self):
        ev = ConcreteEvaluator()
        signal = ev.score_conversation(_conv())
        assert signal.total == 0.0

    def test_no_victim_turn_returns_zero(self):
        ev = ConcreteEvaluator()
        c = _conv(turns=[("attacker", "hi")])
        signal = ev.score_conversation(c)
        assert signal.total == 0.0
