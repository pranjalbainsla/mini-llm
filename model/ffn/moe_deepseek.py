import torch
import torch.nn as nn
import torch.nn.functional as F

def ffn_hidden_dim(config, active_experts=1, multiple=1):
    """Iso-active SwiGLU width: active_experts * 3 * C * d == 8 * C^2."""
    d = getattr(config, "moe_intermediate_size", None)
    if d is None:
        d = 8 * config.n_embd / (3 * active_experts)
        d = max(multiple, round(d / multiple) * multiple)
    return d

class Expert(nn.Module):
    def __init__(self, config, hidden_dim):
        super().__init__()
        self.gate = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.up = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.down = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        # Note: Refer moe_scaling_practice.ipynb (point 3 - granularity) for theory/math

    def forward(self, x):
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))
  
class MoEDeepSeek(nn.Module):
    """ Adds always-active shared experts alongside routed experts """

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_shared_experts = config.num_shared_experts
        self.k = config.k
        self.target_fraction = 1.0 / config.num_experts
        self.bias_update_speed = config.bias_update_speed
        self.router = nn.Linear(config.n_embd, config.num_experts, bias=False)
        # Fine-grained segmentation: width is iso-active across routed (k) + shared experts.
        d = ffn_hidden_dim(config, active_experts=self.k + self.num_shared_experts)
        self.experts = nn.ModuleList(
            [Expert(config, d) for _ in range(config.num_experts)]
        )
        self.shared_experts = nn.ModuleList(
            [Expert(config, d) for _ in range(config.num_shared_experts)]
        )
        self.register_buffer("expert_bias", torch.zeros(config.num_experts))

    def forward(self, x):
        B, T, C = x.shape
        tokens = x.reshape(B * T, C)
        
        router_logits = self.router(tokens) # (B*T, num_experts)
        scores = torch.sigmoid(router_logits) 

        # Bias only affects expert selection.
        biased_scores = scores + self.expert_bias

        _, topk_idx = torch.topk(biased_scores, self.k, dim=-1) # (B*T, k)

        # Gating uses the original affinity scores.
        # Each row of topk_idx lists the columns (dim=1) to read from the 
        # same row of scores, so the output has topk_idx's shape (B*T, k)
        gates = scores.gather(1, topk_idx)
        gates /= gates.sum(dim=-1, keepdim=True)
        
        routed_out = torch.zeros_like(tokens)

        for expert_id, expert in enumerate(self.experts):

            token_idx, slot_idx = (topk_idx == expert_id).nonzero(as_tuple=True)

            if token_idx.numel() == 0:
                continue

            expert_input = tokens[token_idx]
            expert_output = expert(expert_input)
            weights = gates[token_idx, slot_idx].unsqueeze(-1)

            routed_out[token_idx] += weights * expert_output
        
        shared_out = torch.zeros_like(tokens)
        for expert in self.shared_experts:
            shared_out += expert(tokens)

        out = routed_out + shared_out

        return out.reshape(B, T, C), topk_idx
    
    @torch.no_grad()
    def update_expert_bias(self, topk_idx):
        """
        topk_idx: (num_tokens, k)
        """

        # Count how many times each expert was selected.
        expert_counts = torch.bincount(
            topk_idx.reshape(-1),
            minlength=self.num_experts
        ).float()

        # Fraction of routing decisions assigned to each expert.
        expert_fraction = expert_counts / topk_idx.numel()

        # Experts used too much -> decrease bias.
        overused = expert_fraction > self.target_fraction

        # Experts used too little -> increase bias.
        underused = expert_fraction < self.target_fraction

        self.expert_bias[overused] -= self.bias_update_speed
        self.expert_bias[underused] += self.bias_update_speed