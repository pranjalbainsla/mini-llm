import torch, math
import torch.nn as nn
import torch.nn.functional as F

from .rope import apply_rope


class MultiheadLatentAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.block_size = config.block_size

        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # K/V are compressed into one shared latent, then expanded back to full per-head K/V
        self.down = nn.Linear(config.n_embd, config.latent_kv_dim, bias=config.bias)
        self.up_k = nn.Linear(config.latent_kv_dim, config.n_head * self.head_dim, bias=config.bias)
        self.up_v = nn.Linear(config.latent_kv_dim, config.n_head * self.head_dim, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

        self.register_buffer("tril", torch.tril(torch.ones(config.block_size, config.block_size)))
        self.reset_cache()

    def reset_cache(self):
        self.latent_cache = None
        self.cos_cache = None  # RoPE is applied to the up-projected K, so keep each cached token's cos/sin
        self.sin_cache = None

    def forward(self, x, cos, sin, use_cache=False, **kwargs):
        B, T, C = x.shape

        Q = self._split_heads(self.q_proj(x))  # (B, n_head, T, head_dim)
        latent = self.down(x)  # (B, T, latent_kv_dim)

        if use_cache:
            latent, cos_k, sin_k = self._update_cache(latent, cos, sin)
        else:
            cos_k, sin_k = cos, sin

        # expand the latent into per-head K/V: (B, n_head, S, head_dim), where S = keys available
        K = self._split_heads(self.up_k(latent))
        V = self._split_heads(self.up_v(latent))

        # rotary embeddings; cos/sin are (B, T, head_dim/2), broadcast over heads
        # (K is rotated after up-projection, so it needs the cos/sin of every cached position)
        Q = apply_rope(Q, cos.unsqueeze(1), sin.unsqueeze(1))
        K = apply_rope(K, cos_k.unsqueeze(1), sin_k.unsqueeze(1))

        # scaled dot-product attention: (B, n_head, T, S)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)
        if T > 1:
            scores = scores.masked_fill(self.tril[:T, :K.size(2)] == 0, float('-inf'))
        attn = self.dropout(F.softmax(scores, dim=-1))

        # weighted sum of values, merge heads: (B, T, C)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, C)

        # output projection mixes information across heads
        return self.dropout(self.proj(out))

    def _split_heads(self, x):
        # (B, T, n_head * head_dim) -> (B, n_head, T, head_dim)
        B, T, _ = x.shape
        return x.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

    def _update_cache(self, latent, cos, sin):
        if self.latent_cache is not None:
            latent = torch.cat([self.latent_cache, latent], dim=1)
            cos = torch.cat([self.cos_cache, cos], dim=1)
            sin = torch.cat([self.sin_cache, sin], dim=1)
        # The model was trained on block_size tokens of context, so keep only the last block_size.
        self.latent_cache = latent[:, -self.block_size:]
        self.cos_cache = cos[:, -self.block_size:]
        self.sin_cache = sin[:, -self.block_size:]
        return self.latent_cache, self.cos_cache, self.sin_cache
