import torch
import torch.nn as nn
import torch.nn.functional as F
from config import (
    n_embd,
    n_head,
    n_layer,
    alpha
)
from .block import Block

class GPT(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None, use_cache=False):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        # pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        # x = tok_emb + pos_emb # (B,T,C)
        x = tok_emb
        # x = self.blocks(x) # (B,T,C)
        total_aux = 0
        routing_info = []
        for block in self.blocks:
          x, topk_idx = block(x, use_cache)
          routing_info.append(topk_idx)
          # total_aux += aux
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            total_loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            ce_loss = F.cross_entropy(logits, targets)

            total_loss = ce_loss

            if total_aux is not None:
                total_loss = ce_loss + alpha * total_aux

        return logits, total_loss, routing_info
    
    def reset_cache(self):
        for block in self.blocks:
            block.attn.reset_cache()  

    # generation before kv cache
    # def generate(self, idx, max_new_tokens):
    #     # idx is (B, T) array of indices in the current context 
    #     for _ in range(max_new_tokens):
    #         # crop idx to the last block_size tokens
    #         idx_cond = idx[:, -block_size:]
    #         # get the predictions
    #         logits, loss = self(idx_cond)
    #         # focus only on the last time step
    #         logits = logits[:, -1, :] # becomes (B, C)
    #         # apply softmax to get probabilities
    #         probs = F.softmax(logits, dim=-1) # (B, C)
    #         # sample from the distribution
    #         idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
    #         # append sampled index to the running sequence
    #         idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
    #     return idx

    def generate(self, idx, max_new_tokens, use_cache):
        # idx: (B, T_prompt)
        self.reset_cache()

        # Prefill: process the entire prompt once and populate the KV cache
        logits, _ = self(idx, use_cache=use_cache)

        # Decode (T = 1 each iteration)
        for _ in range(max_new_tokens):
            # Use the last token's logits to sample the next token
            logits = logits[:, -1, :]          # (B, vocab_size)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append to the generated sequence
            idx = torch.cat((idx, idx_next), dim=1)

            # Feed ONLY the new token; K/V for previous tokens are already cached
            logits, _ = self(idx_next, use_cache=use_cache)

        return idx
    
    @torch.no_grad()
    def update_expert_bias(self, routing_info):
        for block, topk_idx in zip(self.blocks, routing_info):
            if hasattr(block.ffwd, "update_expert_bias"):
                block.ffwd.update_expert_bias(topk_idx)
