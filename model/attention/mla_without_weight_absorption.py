import torch, math
import torch.nn as nn
import torch.nn.functional as F

from .rope import apply_rope


class MultiheadLatentAttentionDeepSeek(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.rope_dim = int(self.head_dim * config.rotary_ratio)  # per-head dims that carry RoPE
        self.non_rope_dim = self.head_dim - self.rope_dim 
        self.block_size = config.block_size

        # Q and KV are each compressed into a latent
        # note: Compressing Q doesn't reduce inference memory, but it cuts the Q projection's parameters and FLOPs, which helps training.
        self.down_proj_q = nn.Linear(config.n_embd, config.latent_q_dim, bias=config.bias)
        self.down_proj_kv = nn.Linear(config.n_embd, config.latent_kv_dim, bias=config.bias)

        # content (non-RoPE) parts, expanded from the latents
        self.up_q = nn.Linear(config.latent_q_dim, config.n_head * self.non_rope_dim, bias=config.bias)
        self.up_k = nn.Linear(config.latent_kv_dim, config.n_head * self.non_rope_dim, bias=config.bias)
        self.up_v = nn.Linear(config.latent_kv_dim, config.n_head * self.head_dim, bias=config.bias)

        # decoupled RoPE parts: per-head rotary query, single rotary key shared by all heads
        self.q_rotary = nn.Linear(config.latent_q_dim, config.n_head * self.rope_dim, bias=config.bias)
        self.k_rotary = nn.Linear(config.n_embd, self.rope_dim, bias=config.bias)

        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

        self.register_buffer("tril", torch.tril(torch.ones(config.block_size, config.block_size)))
        self.reset_cache()

    def reset_cache(self):
        self.kv_cache = None  # latent KV
        self.kr_cache = None  # rotary key, cached after RoPE (shared across heads, so tiny)

    def forward(self, x, cos, sin, use_cache=False, **kwargs):
        B, T, C = x.shape

        cq = self.down_proj_q(x)    # (B, T, latent_q_dim)
        ckv = self.down_proj_kv(x)  # (B, T, latent_kv_dim)

        # rotary parts; cos/sin are (B, T, rope_dim/2)
        QR = apply_rope(self.q_rotary(cq).view(B, T, self.n_head, self.rope_dim), cos.unsqueeze(2), sin.unsqueeze(2))  # (B, T, n_head, rope_dim)
        KR = apply_rope(self.k_rotary(x), cos, sin)          # (B, T, rope_dim)

        if use_cache:
            ckv, KR = self._update_cache(ckv, KR)
        S = ckv.size(1)  # keys available

        # expand the latent into per-head content K and full V (the "naive" part: redone every step)
        KC = self.up_k(ckv).view(B, S, self.n_head, self.non_rope_dim)
        V = self._split_heads(self.up_v(ckv))  # (B, n_head, S, head_dim)

        # concat content and rotary parts: (B, n_head, ., head_dim)
        QC = self.up_q(cq).view(B, T, self.n_head, self.non_rope_dim)
        Q = torch.cat([QC, QR], dim=-1).transpose(1, 2)
        K = torch.cat([KC, KR.unsqueeze(2).expand(-1, -1, self.n_head, -1)], dim=-1).transpose(1, 2)

        # scaled dot-product attention: (B, n_head, T, S)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)
        if T > 1:
            scores = scores.masked_fill(self.tril[:T, :S] == 0, float('-inf'))
        attn = self.dropout(F.softmax(scores, dim=-1))

        # weighted sum of values, merge heads: (B, T, C)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, C)

        # output projection mixes information across heads
        return self.dropout(self.proj(out))

    def _split_heads(self, x):
        # (B, T, n_head * head_dim) -> (B, n_head, T, head_dim)
        B, T, _ = x.shape
        return x.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

    def _update_cache(self, ckv, KR):
        if self.kv_cache is not None:
            ckv = torch.cat([self.kv_cache, ckv], dim=1)
            KR = torch.cat([self.kr_cache, KR], dim=1)
        # The model was trained on block_size tokens of context, so keep only the last block_size.
        self.kv_cache = ckv[:, -self.block_size:]
        self.kr_cache = KR[:, -self.block_size:]
        return self.kv_cache, self.kr_cache
