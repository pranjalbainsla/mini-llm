import torch, math
import torch.nn as nn
import torch.nn.functional as F
from .rope import apply_rope, precompute_freqs
from config import (
    dropout,
    max_seq_len,
    device
)

class MultiheadLatentAttention(nn.Module):

    def __init__(self, n_embd, n_head, latent_dim):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.kv_cache = None
        self.cache_pos = 0
        cos, sin = precompute_freqs(self.head_dim, max_seq_len, device)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

        self.down = nn.Linear(
            n_embd,
            latent_dim,
        )
        self.up_k = nn.Linear(
            latent_dim,
            n_head * self.head_dim,
        )
        self.up_v = nn.Linear(
            latent_dim,
            n_head * self.head_dim,
        )
        self.q_proj = nn.Linear(n_embd, n_embd)
        self.register_buffer('tril', torch.tril(torch.ones(max_seq_len, max_seq_len)))
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, use_cache):
        B, T, C = x.shape
        n, head_dim = self.n_head, self.head_dim

        Q = self.q_proj(x) # (B, T, C)
        latent = self.down(x)

        if use_cache:
            if self.kv_cache is None:
                self.kv_cache = latent
            else:
                self.kv_cache = torch.cat([self.kv_cache, latent], dim=1)
            K = self.up_k(self.kv_cache) 
            V = self.up_v(self.kv_cache)
        else:
            K = self.up_k(latent)
            V = self.up_v(latent)
        
        L = K.size(1)

        Q = Q.view(B, T, n, head_dim) # (B, T, n_head, head_dim)
        K = K.view(B, L, n, head_dim) # (B, L, n_head, head_dim)
        V = V.view(B, L, n, head_dim) # (B, L, n_head, head_dim)

        Q = Q.transpose(1, 2) # (B, n_head, T, head_dim)
        K = K.transpose(1, 2) # (B, n_head, L, head_dim)
        V = V.transpose(1, 2) # (B, n_head, L, head_dim)

        # apply RoPE
        if use_cache:
            start = self.cache_pos
        else:
            start = 0
        end = start + T
        cos_q = self.cos[start:end].unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim/2)
        sin_q = self.sin[start:end].unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim/2)
        Q = apply_rope(Q, cos_q, sin_q)

        cos_k = self.cos[:L].unsqueeze(0).unsqueeze(0)
        sin_k = self.sin[:L].unsqueeze(0).unsqueeze(0)
        K = apply_rope(K, cos_k, sin_k)

        if use_cache:
            self.cache_pos += T
        
        wei = Q @ K.transpose(-2,-1) / math.sqrt(head_dim) # (B, H, T, D) @ (B, H, D, T) -> (B, H, T, T)
        if not use_cache or T>1:
            wei = wei.masked_fill(self.tril[:T, :K.size(2)] == 0, float('-inf')) # (B, H, T, T)
        wei = F.softmax(wei, dim=-1) # (B, H, T, T)
        wei = self.dropout(wei)
        out = wei @ V # (B, H, T, T) @ (B, H, T, D) -> (B, H, T, D)

        # (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        out = self.proj(out) # lets the model mix information across different heads
        out = self.dropout(out)

        return out

    def reset_cache(self):
        self.kv_cache = None
        self.cache_pos = 0