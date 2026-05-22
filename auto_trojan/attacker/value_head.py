from __future__ import annotations

import torch
import torch.nn as nn


class ValueHead(nn.Module):
    """
    Projects an LM's last hidden state to a scalar value estimate V(s).

    Structurally identical to a single-output sequence classifier, but trained
    against discounted Monte Carlo returns rather than fixed supervised labels.
    """

    def __init__(self, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, T, H] (full sequence) or [B, H] (single token)
        Returns:
            [B] scalar value estimates, one per sample
        """
        if hidden_states.dim() == 3:
            x = hidden_states[:, -1, :]  # last token — causal LM convention
        else:
            x = hidden_states
        x = x.to(dtype=self.net[1].weight.dtype)  # align with Linear layer dtype (fp16/bf16/fp32)
        return self.net(x).squeeze(-1)
