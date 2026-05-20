from abc import ABC, abstractmethod


class VictimLLM(ABC):
    """Interface all victim model adapters must implement."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs: object) -> str:
        """Generate a response from the victim model."""
        ...
