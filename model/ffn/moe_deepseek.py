import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout),
        )
        # TODO: add moe_intermediate_size config knob to introduce DeepSeekMoe's
        # "fine grained expert segmentation" (which is the idea that as as num_experts
        # goes up, we deliberately shrink each expert's hidden dim so that P_total (model size) 
        # doesn't explode and each expert specializes on a narrower slice of behavior)
        # Note: Refer moe_scaling_practice.ipynb (point 3 - granularity) for theory/math

    def forward(self, x):
        return self.net(x)
  
class MoEDeepSeek(nn.Module):
    """ Adds always-active shared experts alongside routed experts """

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_shared_experts = config.num_shared_experts
        self.k = config.k
        self.target_fraction = config.k / config.num_experts
        self.bias_update_speed = config.bias_update_speed
        self.router = nn.Linear(config.n_embd, config.num_experts)
        self.experts = nn.ModuleList(
            [Expert(config) for _ in range(config.num_experts)]
        )
        self.shared_experts = nn.ModuleList(
            [Expert(config) for _ in range(config.num_shared_experts)]
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