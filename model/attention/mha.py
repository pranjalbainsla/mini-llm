import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import apply_rope

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, config):

        super().__init__()

        self.head_dim = config.n_embd // config.n_head
        self.key = nn.Linear(config.n_embd, self.head_dim, bias=config.bias)
        self.query = nn.Linear(config.n_embd, self.head_dim, bias=config.bias)
        self.value = nn.Linear(config.n_embd, self.head_dim, bias=config.bias)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, cos, sin):
        B,T,C = x.shape

        k = self.key(x)   # (B,T,head_dim)
        q = self.query(x) # (B,T,head_dim)
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

    def __init__(self, config):
        super().__init__()
        self.head_dim = config.n_embd // config.n_head
        self.heads = nn.ModuleList([Head(config) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, cos, sin, **kwargs):
        # cos/sin: (B, T, head_dim/2), computed once per forward by GPT
        out = torch.cat([h(x, cos, sin) for h in self.heads], dim=-1) # concatenate over the channel dimension
        out = self.dropout(self.proj(out)) # (B, T, num_heads * head_dim) -> (B, T, n_embd)
        return out