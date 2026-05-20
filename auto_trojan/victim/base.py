from __future__ import annotations

from abc import ABC, abstractmethod

from auto_trojan.conversation import Conversation


class VictimLLM(ABC):
    """Interface all victim model adapters must implement."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs: object) -> str:
        """Generate a response to a single prompt."""
        ...

    def generate_with_history(self, conversation: Conversation, **kwargs: object) -> str:
        """
        Generate the victim's next response given the full conversation so far.

        The default implementation formats the conversation into a flat string
        and delegates to generate(). Adapters with native multi-turn support
        (OpenAI messages API, HuggingFace chat templates) should override this.
        """
        return self.generate(self._format_conversation(conversation), **kwargs)

    @staticmethod
    def _format_conversation(conversation: Conversation) -> str:
        """Flat text fallback for models without a chat template."""
        lines = []
        role_label = {"attacker": "User", "victim": "Assistant"}
        for turn in conversation.turns:
            lines.append(f"{role_label[turn.role]}: {turn.content}")
        lines.append("Assistant:")
        return "\n".join(lines)
