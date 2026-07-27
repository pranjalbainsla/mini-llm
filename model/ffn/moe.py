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
    def __init__(self, n_embd, num_experts, k):
        super().__init__()
        self.router = nn.Linear(n_embd, num_experts)
        self.experts = nn.ModuleList(
            [Expert(n_embd) for _ in range(num_experts)]
        )
        self.k = k

    def forward(self, x):
        B, T, C = x.shape
        tokens = x.reshape(B * T, C)
        
        # argmax returns the index of the largest value
        # expert_idx = self.router(tokens).argmax(dim=-1) # (B*T, 1) ? no it's (B*T,) 

        expert_weights, expert_indices = self.router(tokens).topk(self.k, dim=-1) # (B*T, k)
        # softmax for weights

        out = torch.zeros_like(tokens)

        # for i, expert in enumerate(self.experts):
        #     if self.k == 1:
        #         mask = expert_idx == i # boolean tensor of (B*T,) shape
        #         # mask.any() returns True if at least one element in the boolean tensor is True
        #         if mask.any(): 
        #             out[mask] = expert(tokens[mask])
        #     else:
        #         mask = (expert_indices == i).any(dim=-1) # (B*T,) -> batch for an expert
        #         if mask.any():
        #             weight = expert_weights[expert_indices == i].unsqueeze(-1)
        #             out[mask] += weight * expert(tokens[mask])

        for expert_id, expert in enumerate(self.experts):

            # Find every (token, slot) pair routed to this expert
            token_idx, slot_idx = (expert_indices == expert_id).nonzero(as_tuple=True)

            if token_idx.numel() == 0:
                continue

            # Gather tokens for this expert
            expert_input = tokens[token_idx]

            # Forward once on the whole mini-batch
            expert_output = expert(expert_input)

            # Corresponding routing weights
            weights = expert_weights[token_idx, slot_idx].unsqueeze(-1)

            # Scatter-add back into output
            out[token_idx] += weights * expert_output
            

        return out.reshape(B, T, C)
