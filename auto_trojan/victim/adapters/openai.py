from __future__ import annotations

from auto_trojan.victim.base import VictimLLM


class OpenAIVictim(VictimLLM):
    """Victim adapter for OpenAI-compatible chat APIs."""

    def __init__(self, model: str = "gpt-4o", base_url: str | None = None, **kwargs: object) -> None:
        from openai import OpenAI  # type: ignore[import]

        self.model = model
        self.client = OpenAI(base_url=base_url)
        self.extra = kwargs

    def generate(self, prompt: str, **kwargs: object) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **{**self.extra, **kwargs},
        )
        return response.choices[0].message.content or ""
