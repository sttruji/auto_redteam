# auto_trojan

Open-source, modular LLM red-teaming framework. Feed it a malicious seed prompt; it learns a policy to reliably jailbreak your target model — so you can patch before release.

## How it works

```
seed prompt → AttackerAgent (generates k variants)
                    ↓
              VictimLLM  (modular, swap any model)
                    ↓
           EvaluationAgent (success score + token likelihood + refusal penalty)
                    ↓
             RewardModel  →  RL loop  →  fine-tune AttackerAgent (LoRA/QLoRA)
```

The attacker is a fine-tunable LLM. The RL loop updates its weights after each episode so it progressively learns what prompt patterns break the victim.

**Coming soon:** a critic agent that builds reusable attack strategies on the fly, and multi-turn conversation attacks.

## Quickstart

```bash
pip install -e ".[dev]"

# Run a red-team campaign
auto-trojan run --config config/default.yaml

# Run tests
pytest
```

## Configuration

Copy `config/default.yaml` and edit:

```yaml
victim:
  adapter: huggingface          # or: openai, anthropic
  model_id: meta-llama/Llama-2-7b-chat-hf

attacker:
  model_id: mistralai/Mistral-7B-v0.1
  k: 10                         # attack prompts per episode

evaluator:
  weights:
    eval: 1.0
    affirmative_likelihood: 0.5
    refusal_penalty: 0.3

rl:
  episodes: 500
  lora_rank: 16
```

## Swapping the Victim

Implement the `VictimLLM` interface and register your adapter:

```python
from auto_trojan.victim.base import VictimLLM

class MyModelAdapter(VictimLLM):
    def generate(self, prompt: str, **kwargs) -> str:
        ...
```

## Research & Ethics

This tool is built for **defensive security research** — helping model developers find and patch vulnerabilities before release. All usage should be authorized. Do not use against models or systems you do not own or have explicit permission to test.

## License

MIT
