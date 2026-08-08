import torch
import torch.nn as nn
import torch.nn.functional as F
from .rope import apply_rope, precompute_freqs
from config import (
    n_embd,
    block_size,
    dropout,
    max_seq_len,
    device
)

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_dim):
        super().__init__()
        self.head_dim = head_dim
        self.key = nn.Linear(n_embd, head_dim, bias=False)
        self.query = nn.Linear(n_embd, head_dim, bias=False)
        self.value = nn.Linear(n_embd, head_dim, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cos, sin):
        B,T,C = x.shape

        k = self.key(x)   # (B,T,head_dim)
        q = self.query(x) # (B,T,head_dim)
        cos = cos.unsqueeze(0) # (1, T, head_dim/2)
        sin = sin.unsqueeze(0)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * self.head_dim**-0.5 # (B, T, head_dim) @ (B, head_dim, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,head_dim)
        out = wei @ v # (B, T, T) @ (B, T, head_dim) -> (B, T, head_dim)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, n_head, head_dim):
        super().__init__()
        cos, sin = precompute_freqs(head_dim, max_seq_len, device)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
        self.heads = nn.ModuleList([Head(head_dim) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x, self.cos, self.sin) for h in self.heads], dim=-1) # concatenate over the channel dimension
        out = self.dropout(self.proj(out)) # (B, T, num_heads * head_dim) -> (B, T, n_embd)
        return out