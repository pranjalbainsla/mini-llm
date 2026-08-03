import torch
import torch.nn as nn
import torch.nn.functional as F
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
  
class MoEDeepSeek(nn.Module):
    """ Adds always-active shared experts alongside routed experts """

    def __init__(self, n_embd, num_experts, num_shared_experts, k):
        super().__init__()
        self.router = nn.Linear(n_embd, num_experts)
        self.experts = nn.ModuleList(
            [Expert(n_embd) for _ in range(num_experts)]
        )
        self.shared_experts = nn.ModuleList(
            [Expert(n_embd) for _ in range(num_shared_experts)]
        )
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.k = k

    def forward(self, x):
        B, T, C = x.shape
        tokens = x.reshape(B * T, C)
        
        router_logits = self.router(tokens) # (B*T, num_experts)
        probs = F.softmax(router_logits, dim=-1)
        topk_probs, topk_idx = torch.topk(probs, self.k, dim=-1) # (B*T, k)
        topk_probs /= topk_probs.sum(dim=-1, keepdim=True)

        routed_out = torch.zeros_like(tokens)

        for expert_id, expert in enumerate(self.experts):

            token_idx, slot_idx = (topk_idx == expert_id).nonzero(as_tuple=True)

            if token_idx.numel() == 0:
                continue

            expert_input = tokens[token_idx]
            expert_output = expert(expert_input)
            weights = topk_probs[token_idx, slot_idx].unsqueeze(-1)

            routed_out[token_idx] += weights * expert_output
        
        shared_out = torch.zeros_like(tokens)
        for expert in self.shared_experts:
            shared_out += expert(tokens)

        out = routed_out + shared_out

        # Load balancing loss     
        P = probs.mean(dim=0)
        mask = F.one_hot(topk_idx, num_classes=self.num_experts).float()
        f = mask.sum(dim=1).float().mean(dim=0) / self.k
        aux_loss = self.num_experts * (P * f).sum()

        return out.reshape(B, T, C), aux_loss