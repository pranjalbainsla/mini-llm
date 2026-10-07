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

    def forward(self, x):
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))

class MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts, self.k = config.num_experts, config.k
        d = ffn_hidden_dim(config, active_experts=self.k)
        self.router = nn.Linear(config.n_embd, self.num_experts, bias=False)
        self.experts = nn.ModuleList(Expert(config, d) for _ in range(self.num_experts))
        self.aux_loss = None  # load-balancing loss from the latest forward
        self.load = None  # per-expert token fraction, for logging

    def forward(self, x):
        B, T, C = x.shape
        tokens = x.reshape(B * T, C)
        
        router_logits = self.router(tokens) # (B*T, num_experts)
        probs = F.softmax(router_logits, dim=-1)
        topk_probs, topk_idx = torch.topk(probs, self.k, dim=-1) # both (B*T, k)
        if self.k > 1:
            topk_probs /= topk_probs.sum(dim=-1, keepdim=True)

        out = torch.zeros_like(tokens)

        for expert_id, expert in enumerate(self.experts):

            # Find every (token, slot) pair routed to this expert

            # (topk_idx == expert_id) makes a (B*T, k) boolean mask of where 
            # this expert was picked, and .nonzero(as_tuple=True) returns the 
            # row indices (token_idx, which tokens) and column indices 
            # (slot_idx, which of the token's k choices) of the True entries
            token_idx, slot_idx = (topk_idx == expert_id).nonzero(as_tuple=True)

            if token_idx.numel() == 0:
                continue

            # Gather tokens for this expert
            expert_input = tokens[token_idx]

            # Forward once on the whole mini-batch
            expert_output = expert(expert_input)

            # Corresponding routing weights
            weights = topk_probs[token_idx, slot_idx].unsqueeze(-1)

            # Scatter-add back into output
            out[token_idx] += weights * expert_output
            
        P = probs.mean(dim=0) # average probability of each expert being selected across all tokens
        mask = F.one_hot(topk_idx, num_classes=self.num_experts).float()
        f = mask.sum(dim=1).float().mean(dim=0) / self.k # average fraction of tokens routed to each expert

        aux_loss = self.num_experts * (P * f).sum()

        self.aux_loss = aux_loss   # GPT adds config.alpha * sum over layers when use_aux_loss is on
        self.load = f.detach()

        return out.reshape(B, T, C), None  # None = no routing info, same as the dense FFNs
