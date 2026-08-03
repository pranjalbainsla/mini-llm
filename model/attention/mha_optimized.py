import torch, math
import torch.nn as nn
import torch.nn.functional as F
from .rope import apply_rope, precompute_freqs
from config import (
    block_size,
    dropout,
    max_seq_len,
    device
)

class MultiHeadAttentionOptimized(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        cos, sin = precompute_freqs(self.head_dim, max_seq_len, device)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

        self.k_cache = None
        self.v_cache = None
        self.cache_pos = 0

        self.q_proj = nn.Linear(n_embd, n_embd)
        self.k_proj = nn.Linear(n_embd, n_embd)
        self.v_proj = nn.Linear(n_embd, n_embd)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, use_cache):
        B, T, C = x.shape
        n, head_dim = self.n_head, self.head_dim
        if use_cache:
            start = self.cache_pos
        else:
            start = 0
        end = start + T

        # (B, T, C)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # (B, T, n_head, head_dim)
        Q = Q.view(B, T, n, head_dim)
        K = K.view(B, T, n, head_dim)
        V = V.view(B, T, n, head_dim)

        # (B, n_head, T, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # apply RoPE
        cos = self.cos[start:end].unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim/2)
        sin = self.sin[start:end].unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim/2)
        Q = apply_rope(Q, cos, sin)
        K = apply_rope(K, cos, sin)

        if use_cache:
            if self.k_cache is None:
                self.k_cache = K
                self.v_cache = V
            else:
                self.k_cache = torch.cat([self.k_cache, K], dim=2)
                self.v_cache = torch.cat([self.v_cache, V], dim=2)
                # Keep only the last block_size tokens.
                # The model was trained with a context window of block_size, so letting the
                # cache grow beyond this hurts generation quality.
                if self.k_cache.size(2) > block_size:
                    self.k_cache = self.k_cache[:, :, -block_size:, :]
                    self.v_cache = self.v_cache[:, :, -block_size:, :]
        if use_cache:
            self.cache_pos += T
        
        # attention
        if use_cache:
            K = self.k_cache
            V = self.v_cache
        wei = Q @ K.transpose(-2,-1) / math.sqrt(head_dim) # (B, H, T, D) @ (B, H, D, T) -> (B, H, T, T)
        if not use_cache or T>1:
            # don't need mask for generation (generating one token at a time)
            wei = wei.masked_fill(self.tril[:T, :K.size(2)] == 0, float('-inf')) # (B, H, T, T)
        wei = F.softmax(wei, dim=-1) # (B, H, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        out = wei @ V # (B, H, T, T) @ (B, H, T, D) -> (B, H, T, D)

        # (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        out = self.proj(out) # lets the model mix information across different heads
        out = self.dropout(out)

        return out
    
    def reset_cache(self):
        self.v_cache = None
        self.k_cache = None
        self.cache_pos = 0