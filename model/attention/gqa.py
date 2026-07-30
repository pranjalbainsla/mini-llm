import torch, math
import torch.nn as nn
import torch.nn.functional as F
from .rope import apply_rope
from config import (
    block_size,
    dropout
)

class GroupedQueryAttention(nn.Module):

    def __init__(self, n_embd, n_head, n_kv_heads):
        super().__init__()
        assert n_embd % n_head == 0
        assert n_head % n_kv_heads == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.n_kv_heads = n_kv_heads
        self.repeat = n_head // n_kv_heads
        self.q_proj = nn.Linear(n_embd, n_embd)
        self.k_proj = nn.Linear(n_embd, n_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(n_embd, n_kv_heads * self.head_dim)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        n, head_dim, n_kv_heads, repeat = self.n_head, self.head_dim, self.n_kv_heads, self.repeat

        Q = self.q_proj(x) # (B, T, C)
        K = self.k_proj(x) # (B, T, n_kv_heads * head_dim)
        V = self.v_proj(x) # (B, T, n_kv_heads * head_dim)

        Q = Q.view(B, T, n, head_dim) # (B, T, n_head, head_dim)
        K = K.view(B, T, n_kv_heads, head_dim) # (B, T, n_kv_heads, head_dim)
        V = V.view(B, T, n_kv_heads, head_dim) # (B, T, n_kv_heads, head_dim)

        Q = Q.transpose(1, 2) # (B, n_head, T, head_dim)
        K = K.transpose(1, 2) # (B, n_kv_heads, T, head_dim)
        V = V.transpose(1, 2) # (B, n_kv_heads, T, head_dim)

        K = K.repeat_interleave(repeat, dim=1)
        V = V.repeat_interleave(repeat, dim=1)

        # apply RoPE
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim/2)
        sin = sin.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim/2)
        Q = apply_rope(Q, cos, sin)
        K = apply_rope(K, cos, sin)

        # attention
        wei = Q @ K.transpose(-2,-1) / math.sqrt(head_dim) # (B, H, T, D) @ (B, H, D, T) -> (B, H, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, H, T, T)
        wei = F.softmax(wei, dim=-1) # (B, H, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        out = wei @ V # (B, H, T, T) @ (B, H, T, D) -> (B, H, T, D)

        # (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        out = self.proj(out) # lets the model mix information across different heads
        out = self.dropout(out)

        return out