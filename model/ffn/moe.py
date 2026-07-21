import torch
import torch.nn as nn
from config import (
    dropout,
)
class Expert(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
  
class MoE(nn.Module):
    def __init__(self, n_embd, num_experts):
        super().__init__()
        self.router = nn.Linear(n_embd, num_experts)
        self.experts = nn.ModuleList(
            [Expert(n_embd) for _ in range(num_experts)]
        )

    def forward(self, x):
        B, T, C = x.shape
        tokens = x.reshape(B * T, C)

        expert_idx = self.router(tokens).argmax(dim=-1)

        out = torch.zeros_like(tokens)

        for i, expert in enumerate(self.experts):
            mask = expert_idx == i
            if mask.any():
                out[mask] = expert(tokens[mask])

        return out.reshape(B, T, C)
