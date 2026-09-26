import torch, math
import torch.nn as nn
import torch.nn.functional as F

from .rope import apply_rope, precompute_freqs

class GroupedQueryAttention(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config
        
        assert config.n_embd % config.n_head == 0
        assert config.n_head % config.n_kv_heads == 0
        self.head_dim = config.n_embd // config.n_head
        self.repeat = config.n_head // config.n_kv_heads

        cos, sin = precompute_freqs(
            self.head_dim,
            config.max_seq_len
        )
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

        self.k_cache = None
        self.v_cache = None
        self.cache_pos = 0

        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_kv_heads * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_kv_heads * self.head_dim, bias=config.bias)

        self.register_buffer(
            "tril",
            torch.tril(
                torch.ones(config.max_seq_len, config.max_seq_len)
            ),
        )
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, use_cache):
        B, T, C = x.shape

        if use_cache:
            start = self.cache_pos
        else:
            start = 0
        end = start + T

        Q = self.q_proj(x) # (B, T, C)
        K = self.k_proj(x) # (B, T, n_kv_heads * head_dim)
        V = self.v_proj(x) # (B, T, n_kv_heads * head_dim)

        Q = Q.view(B, T, self.config.n_head, self.head_dim) # (B, T, n_head, head_dim)
        K = K.view(B, T, self.config.n_kv_heads, self.head_dim) # (B, T, n_kv_heads, head_dim)
        V = V.view(B, T, self.config.n_kv_heads, self.head_dim) # (B, T, n_kv_heads, head_dim)

        Q = Q.transpose(1, 2) # (B, n_head, T, head_dim)
        K = K.transpose(1, 2) # (B, n_kv_heads, T, head_dim)
        V = V.transpose(1, 2) # (B, n_kv_heads, T, head_dim)

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
                if self.k_cache.size(2) > self.config.block_size:
                    self.k_cache = self.k_cache[:, :, -self.config.block_size:, :]
                    self.v_cache = self.v_cache[:, :, -self.config.block_size:, :]
        if use_cache:
            self.cache_pos += T

        # attention
        if use_cache:
            K = self.k_cache
            V = self.v_cache
        K = K.repeat_interleave(self.repeat, dim=1)
        V = V.repeat_interleave(self.repeat, dim=1)

        wei = Q @ K.transpose(-2,-1) / math.sqrt(self.head_dim) # (B, H, T, D) @ (B, H, D, T) -> (B, H, T, T)
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
        self.v_cache = None
        self.k_cache = None
        self.cache_pos = 0