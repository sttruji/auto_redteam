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
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
