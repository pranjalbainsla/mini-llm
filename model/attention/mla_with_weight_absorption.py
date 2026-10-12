import torch, math
import torch.nn as nn
import torch.nn.functional as F

from .rope import apply_rope


class MLADeepSeekOptimized(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.rope_dim = int(self.head_dim * config.rotary_ratio)  # per-head dims that carry RoPE
        self.non_rope_dim = self.head_dim - self.rope_dim
        self.block_size = config.block_size

        # Q and KV are each compressed into a latent
        self.down_proj_q = nn.Linear(config.n_embd, config.latent_q_dim, bias=config.bias)
        self.down_proj_kv = nn.Linear(config.n_embd, config.latent_kv_dim, bias=config.bias)

        # content (non-RoPE) parts, expanded from the latents
        self.up_q = nn.Linear(config.latent_q_dim, config.n_head * self.non_rope_dim, bias=config.bias)
        self.up_k = nn.Linear(config.latent_kv_dim, config.n_head * self.non_rope_dim, bias=config.bias)
        self.up_v = nn.Linear(config.latent_kv_dim, config.n_head * self.head_dim, bias=config.bias)

        # decoupled RoPE parts: per-head rotary query, single rotary key shared by all heads
        self.q_rotary = nn.Linear(config.latent_q_dim, config.n_head * self.rope_dim, bias=config.bias)
        self.k_rotary = nn.Linear(config.n_embd, self.rope_dim, bias=config.bias)

        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

        self.register_buffer("tril", torch.tril(torch.ones(config.block_size, config.block_size)))
        self.reset_cache()

    def reset_cache(self):
        self.kv_cache = None  # latent KV
        self.kr_cache = None  # rotary key, cached after RoPE (shared across heads, so tiny)

    def forward(self, x, cos, sin, use_cache=False, use_weight_absorption=False, **kwargs):
        B, T, C = x.shape

        cq = self.down_proj_q(x)    # (B, T, latent_q_dim)
        ckv = self.down_proj_kv(x)  # (B, T, latent_kv_dim)

        # rotary parts; cos/sin are (B, T, rope_dim/2)
        QR = apply_rope(self.q_rotary(cq).view(B, T, self.n_head, self.rope_dim), cos.unsqueeze(2), sin.unsqueeze(2))  # (B, T, n_head, rope_dim)
        KR = apply_rope(self.k_rotary(x), cos, sin)  # (B, T, rope_dim)

        if use_cache:
            ckv, KR = self._update_cache(ckv, KR)
        S = ckv.size(1)  # keys available

        # scores: (B, n_head, T, S)
        if use_weight_absorption:
            scores = self._scores_absorbed(cq, ckv, QR, KR)
        else:
            scores = self._scores_expanded(cq, ckv, QR, KR)
        if T > 1:
            scores = scores.masked_fill(self.tril[:T, :S] == 0, float('-inf'))
        attn = self.dropout(F.softmax(scores, dim=-1))

        # weighted sum of values, then output projection: (B, T, C)
        if use_weight_absorption:
            out = self._output_absorbed(attn, ckv)
        else:
            out = self._output_expanded(attn, ckv)

        return self.dropout(out)

    def _scores_expanded(self, cq, ckv, QR, KR):
        # original MLA: reconstruct full per-head K from the latent, then Q @ K^T
        B, T = cq.shape[:2]
        S = ckv.size(1)
        KC = self.up_k(ckv).view(B, S, self.n_head, self.non_rope_dim)
        QC = self.up_q(cq).view(B, T, self.n_head, self.non_rope_dim)

        Q = torch.cat([QC, QR], dim=-1).transpose(1, 2)  # (B, n_head, T, head_dim)
        K = torch.cat([KC, KR.unsqueeze(2).expand(-1, -1, self.n_head, -1)], dim=-1).transpose(1, 2)  # (B, n_head, S, head_dim)
        return Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)

    def _scores_absorbed(self, cq, ckv, QR, KR):
        # q_c . k_c = cq W_q^T W_k ckv^T: fold W_k into the query so scores come straight from the latent cache
        # TODO: absorbed path ignores the up_q / up_k biases, so it only matches the expanded
        # path when bias=False. Either assert config.bias is False for this variant, or fold the biases into the absorbed computation.
        Wq = self.up_q.weight.view(self.n_head, self.non_rope_dim, -1)  # (n_head, non_rope_dim, latent_q_dim)
        Wk = self.up_k.weight.view(self.n_head, self.non_rope_dim, -1)  # (n_head, non_rope_dim, latent_kv_dim)

        # each head produces a latent-space query instead of reconstructing full K
        Q_absorb = torch.einsum("btq,hdq,hdk->bhtk", cq, Wq, Wk)  # (B, n_head, T, latent_kv_dim)
        content = Q_absorb @ ckv.unsqueeze(1).transpose(-2, -1)   # (B, n_head, T, S)
        rotary = QR.transpose(1, 2) @ KR.unsqueeze(1).transpose(-2, -1)  # (B, n_head, T, S), KR shared across heads
        return (content + rotary) / math.sqrt(self.head_dim)

    def _output_expanded(self, attn, ckv):
        # reconstruct full per-head V from the latent, then merge heads and project
        B, H, T, S = attn.shape
        V = self.up_v(ckv).view(B, S, H, self.head_dim).transpose(1, 2)  # (B, n_head, S, head_dim)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, H * self.head_dim)
        return self.out_proj(out)

    def _output_absorbed(self, attn, ckv):
        # fold W_v into W_o, so attention-weighted latents map straight to the output
        C = self.out_proj.weight.size(0)
        Wv = self.up_v.weight.view(self.n_head, self.head_dim, -1)    # (n_head, head_dim, latent_kv_dim)
        Wo = self.out_proj.weight.view(C, self.n_head, self.head_dim)  # (C, n_head, head_dim)
        W_out = torch.einsum("chd,hdl->hcl", Wo, Wv)                   # (n_head, C, latent_kv_dim)

        latent_out = attn @ ckv.unsqueeze(1)  # (B, n_head, T, latent_kv_dim)
        return torch.einsum("bhtl,hcl->btc", latent_out, W_out)

    def _update_cache(self, ckv, KR):
        if self.kv_cache is not None:
            ckv = torch.cat([self.kv_cache, ckv], dim=1)
            KR = torch.cat([self.kr_cache, KR], dim=1)
        # The model was trained on block_size tokens of context, so keep only the last block_size.
        self.kv_cache = ckv[:, -self.block_size:]
        self.kr_cache = KR[:, -self.block_size:]
        return self.kv_cache, self.kr_cache
