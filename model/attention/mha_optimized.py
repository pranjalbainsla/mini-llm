import torch, math
import torch.nn as nn
import torch.nn.functional as F

from .rope import apply_rope


class MultiHeadAttentionOptimized(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.block_size = config.block_size

        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

        self.register_buffer("tril", torch.tril(torch.ones(config.block_size, config.block_size)))
        self.reset_cache()

    def reset_cache(self):
        self.k_cache = None
        self.v_cache = None

    def forward(self, x, cos, sin, use_cache=False, **kwargs):
        B, T, C = x.shape

        # project and split into heads: (B, n_head, T, head_dim)
        Q = self._split_heads(self.q_proj(x))
        K = self._split_heads(self.k_proj(x))
        V = self._split_heads(self.v_proj(x))

        # rotary embeddings; cos/sin are (B, T, head_dim/2), broadcast over heads
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        Q, K = apply_rope(Q, cos, sin), apply_rope(K, cos, sin)

        if use_cache:
            K, V = self._update_cache(K, V)

        # scaled dot-product attention: (B, n_head, T, S), where S = keys available
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)
        if T > 1:  # a single new token can see everything, so it needs no mask
            scores = scores.masked_fill(self.tril[:T, :K.size(2)] == 0, float('-inf'))
        attn = self.dropout(F.softmax(scores, dim=-1))

        # weighted sum of values, merge heads: (B, T, C)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, C)

        # output projection mixes information across heads
        return self.dropout(self.proj(out))

    def _split_heads(self, x):
        # (B, T, C) -> (B, n_head, T, head_dim)
        B, T, _ = x.shape
        return x.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

    def _update_cache(self, K, V):
        if self.k_cache is not None:
            K = torch.cat([self.k_cache, K], dim=2)
            V = torch.cat([self.v_cache, V], dim=2)
        # The model was trained on block_size tokens of context, so keep only the last block_size.
        self.k_cache = K[:, :, -self.block_size:]
        self.v_cache = V[:, :, -self.block_size:]
        return self.k_cache, self.v_cache