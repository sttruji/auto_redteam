from __future__ import annotations

from auto_trojan.conversation import Conversation
from auto_trojan.victim.base import VictimLLM


class HuggingFaceVictim(VictimLLM):
    """Victim adapter for any HuggingFace causal LM."""

    def __init__(self, model_id: str, device: str = "cuda", **generation_kwargs: object) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import]

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
        self.device = device
        self.generation_kwargs = generation_kwargs

    def generate(self, prompt: str, **kwargs: object) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        merged = {**self.generation_kwargs, **kwargs}
        output_ids = self.model.generate(**inputs, **merged)
        prompt_len = inputs["input_ids"].shape[1]
        return self.tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)

    def generate_with_history(self, conversation: Conversation, **kwargs: object) -> str:
        """
        Use the tokenizer's chat template when available (Llama, Mistral, etc.),
        falling back to flat-string formatting for models without one.
        """
        messages = conversation.to_messages()
        merged = {**self.generation_kwargs, **kwargs}

        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            input_text: str = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        else:
            formatted = self._format_conversation(conversation)
            inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)

        output_ids = self.model.generate(**inputs, **merged)
        prompt_len = inputs["input_ids"].shape[1]
        return self.tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)
