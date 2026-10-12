import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import Block
from .builder import build_norm
from .attention.rope import RotaryEmbedding

class GPT(nn.Module):

    def __init__(self, vocab_size, config):
        super().__init__()

        self.config = config
        self.block_size = config.block_size

        self.token_embedding_table = nn.Embedding(vocab_size, config.n_embd)
        # RoPE cos/sin are computed once per forward here and passed to every layer
        head_dim = config.n_embd // config.n_head
        rope_dim = int(head_dim * config.rotary_ratio) if config.attention.startswith("mla") else head_dim
        self.rotary = RotaryEmbedding(rope_dim)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = build_norm(config)
        self.lm_head = nn.Linear(config.n_embd, vocab_size, bias=config.bias)

    def forward(self, idx, targets=None, position_ids=None, **kwargs):
        B, T = idx.shape
        x = self.token_embedding_table(idx) # (B,T,C)
        if position_ids is None:
            position_ids = torch.arange(T, device=idx.device).unsqueeze(0).expand(B, -1)
        cos, sin = self.rotary(position_ids)

        total_aux = None
        routing_info = []
        for block in self.blocks:
            x, topk_idx = block(x, cos, sin, **kwargs)
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
        """Autoregressively sample `max_new_tokens` tokens after the prompt `idx` (B, T).

        use_cache=True: prefill the prompt once, then feed only the newest token each step
        (the attention layers keep K/V state). Positions must then be passed explicitly for RoPE.
        use_cache=False: re-run the full (cropped to block_size) sequence every step.
        Returns (B, T + max_new_tokens) token ids.
        """
        self.reset_cache()  # drop K/V state left over from any previous call
        fwd_kwargs = dict(use_cache=use_cache, use_weight_absorption=use_weight_absorption)

        # Prefill: process the whole prompt (cropped to the context window) in one pass.
        idx_cond = idx[:, -self.block_size:]
        logits, _, _ = self(idx_cond, **fwd_kwargs)
        pos = idx_cond.size(1)  # absolute position of the next token (used only with the cache)

        for step in range(max_new_tokens):
            logits = logits[:, -1, :] / temperature  # (B, vocab_size)
            if top_k is not None:
                # keep the k largest logits, mask the rest out before softmax
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # note: repeated torch.cat reallocates every step
            # pre-allocating a (B, T+max_new_tokens) buffer
            # and writing into it would avoid that.
            idx = torch.cat((idx, idx_next), dim=1)

            if step == max_new_tokens - 1:
                break  # last token sampled, another forward pass would be wasted

            # Build the next model input.
            if use_cache:
                # cache already holds everything before idx_next, so feed just the new token
                # and tell RoPE its true absolute position
                idx_cond = idx_next
                position_ids = torch.full((idx.size(0), 1), pos, device=idx.device)
                pos += 1
            else:
                # no cache: re-encode the whole sequence, positions default to 0..T-1 of the window
                idx_cond = idx[:, -self.block_size:]
                position_ids = None
            logits, _, _ = self(idx_cond, position_ids=position_ids, **fwd_kwargs)

        return idx

    @torch.no_grad()
    def update_expert_bias(self, routing_info):
        for block, topk_idx in zip(self.blocks, routing_info):
            if hasattr(block.ffn, "update_expert_bias"):
                block.ffn.update_expert_bias(topk_idx)