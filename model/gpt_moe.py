import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import Block

class GPT(nn.Module):

    def __init__(self, vocab_size, config):
        super().__init__()

        self.config = config

        self.token_embedding_table = nn.Embedding(vocab_size, config.n_embd)
        # Positional embeddings live inside the attention layer
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, vocab_size)

    def forward(self, idx, targets=None, **kwargs):
        B, T = idx.shape
        x = self.token_embedding_table(idx) # (B,T,C)

        total_aux = None  # aux loss (e.g. MoE load-balancing) not wired up yet
        routing_info = []
        for block in self.blocks:
            x, topk_idx = block(x, **kwargs)
            routing_info.append(topk_idx)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            total_loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            total_loss = F.cross_entropy(logits, targets)
            if total_aux is not None:
                total_loss = total_loss + self.config.alpha * total_aux

        return logits, total_loss, routing_info

    def reset_cache(self):
        for block in self.blocks:
          if hasattr(block.attn, "reset_cache"):
            block.attn.reset_cache()

    def generate(self, idx, max_new_tokens, use_cache=False, use_weight_absorption=False, temperature=1.0, top_k=None):
        self.reset_cache()

        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

        logits, _, _ = self(idx_cond, use_cache=use_cache, use_weight_absorption=use_weight_absorption)

        for _ in range(max_new_tokens):
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            logits, _, _ = self(idx_next, use_cache=use_cache, use_weight_absorption=use_weight_absorption)

        return idx

    @torch.no_grad()
    def update_expert_bias(self, routing_info):
        for block, topk_idx in zip(self.blocks, routing_info):
            if hasattr(block.ffn, "update_expert_bias"):
                block.ffn.update_expert_bias(topk_idx)