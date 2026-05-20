from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


Role = Literal["attacker", "victim"]
_ROLE_TO_API = {"attacker": "user", "victim": "assistant"}


@dataclass
class Turn:
    role: Role
    content: str


@dataclass
class Conversation:
    """
    A multi-turn exchange between the attacker (user) and victim (assistant).

    seed_goal is the attacker's internal objective — it is NOT sent to the victim.
    It is used by the attacker agent to condition each new turn it generates.
    """

    seed_goal: str
    turns: list[Turn] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_turn(self, role: Role, content: str) -> None:
        self.turns.append(Turn(role=role, content=content))

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def to_messages(self) -> list[dict[str, str]]:
        """Return OpenAI-compatible messages list (attacker=user, victim=assistant)."""
        return [{"role": _ROLE_TO_API[t.role], "content": t.content} for t in self.turns]

    def last_attacker_turn(self) -> str | None:
        for turn in reversed(self.turns):
            if turn.role == "attacker":
                return turn.content
        return None

    def last_victim_response(self) -> str | None:
        for turn in reversed(self.turns):
            if turn.role == "victim":
                return turn.content
        return None

    @property
    def num_turns(self) -> int:
        return len(self.turns)

    @property
    def attacker_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.role == "attacker"]

    @property
    def victim_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.role == "victim"]

    @property
    def is_empty(self) -> bool:
        return len(self.turns) == 0
