import torch
import torch.nn as nn
import torch.nn.functional as F
from .rope import apply_rope
from config import (
    n_embd,
    block_size,
    dropout,
)

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_dim):
        super().__init__()
        self.key = nn.Linear(n_embd, head_dim, bias=False)
        self.query = nn.Linear(n_embd, head_dim, bias=False)
        self.value = nn.Linear(n_embd, head_dim, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cos, sin):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        cos = cos.unsqueeze(0) # (1, T, head_dim/2)
        sin = sin.unsqueeze(0)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cos, sin):
        out = torch.cat([h(x, cos, sin) for h in self.heads], dim=-1) # concatenate over the channel dimension
        out = self.dropout(self.proj(out))
        return out