import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import Block
from .builder import build_norm

class GPT(nn.Module):

    def __init__(self, vocab_size, config):
        super().__init__()

        self.config = config
        self.block_size = config.block_size

        self.token_embedding_table = nn.Embedding(vocab_size, config.n_embd)
        # Positional embeddings live inside the attention layer
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = build_norm(config)
        self.lm_head = nn.Linear(config.n_embd, vocab_size, bias=config.bias)

    def forward(self, idx, targets=None, **kwargs):
        B, T = idx.shape
        x = self.token_embedding_table(idx) # (B,T,C)

        total_aux = None
        routing_info = []
        for block in self.blocks:
            x, topk_idx = block(x, **kwargs)
            routing_info.append(topk_idx)
        # Load-balancing aux loss, training only: eval/val loss stays pure cross-entropy so
        # runs with use_aux_loss on/off are comparable.
        if self.training and getattr(self.config, 'use_aux_loss', False):
            auxes = [b.ffn.aux_loss for b in self.blocks if getattr(b.ffn, 'aux_loss', None) is not None]
            if auxes:
                total_aux = torch.stack(auxes).sum()
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            total_loss = None
        else:
            # flatten only for the loss; `logits` itself stays (B,T,vocab_size)
            total_loss = F.cross_entropy(logits.view(B*T, -1), targets.view(B*T))
            if total_aux is not None:
                total_loss = total_loss + self.config.alpha * total_aux

        return logits, total_loss, routing_info

    def reset_cache(self):
        for block in self.blocks:
          if hasattr(block.attn, "reset_cache"):
            block.attn.reset_cache()
            
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, use_cache=False, use_weight_absorption=False, temperature=1.0, top_k=None):
        self.reset_cache()

        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]

        logits, _, _ = self(idx_cond, use_cache=use_cache, use_weight_absorption=use_weight_absorption)

        for _ in range(max_new_tokens):
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            # note: Avoid repeated torch.cat since every gen step reallocates memory. 
            # Instead, pre-allocate a buffer of size (B, T+max_new_tokens) and write into it.

            if use_cache:
                idx_cond = idx_next
            else:
                idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits, _, _ = self(idx_cond, use_cache=use_cache, use_weight_absorption=use_weight_absorption)

        return idx

    @torch.no_grad()
    def update_expert_bias(self, routing_info):
        for block, topk_idx in zip(self.blocks, routing_info):
            if hasattr(block.ffn, "update_expert_bias"):
                block.ffn.update_expert_bias(topk_idx)