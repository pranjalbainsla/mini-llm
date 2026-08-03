import torch, math
import torch.nn as nn
import torch.nn.functional as F
from .rope import apply_rope, precompute_freqs
from config import (
    dropout,
    block_size,
    max_seq_len,
    device
)

class MultiheadLatentAttentionDeepSeek(nn.Module):

    def __init__(self, n_embd, n_head, latent_kv_dim, latent_q_dim):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.dh = n_embd // n_head # 16
        self.dh_non_rotary = 3 * self.dh // 4
        self.dh_rotary = self.dh - self.dh_non_rotary
        cos, sin = precompute_freqs(self.dh_rotary, max_seq_len, device)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

        self.kv_cache = None
        self.kr_cache = None
        self.cache_pos = 0

        self.down_proj_kv = nn.Linear(n_embd, latent_kv_dim)
        self.up_k = nn.Linear(latent_kv_dim, n_head * self.dh_non_rotary) # (48, dc) @ (dc, 1)
        self.k_rotary = nn.Linear(n_embd, self.dh_rotary) # During attention you'll broadcast this same rotary key to every head.

        self.up_v = nn.Linear(latent_kv_dim, n_head * self.dh)
        self.down_proj_q = nn.Linear(n_embd, latent_q_dim)
        self.up_q = nn.Linear(latent_q_dim, n_head * self.dh_non_rotary)
        self.q_rotary = nn.Linear(latent_q_dim, n_head * self.dh_rotary)

        self.register_buffer('tril', torch.tril(torch.ones(max_seq_len, max_seq_len)))
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, use_cache):
        B, T, C = x.shape
        n, dh, dh_nr = self.n_head, self.dh, self.dh_non_rotary

        ckv = self.down_proj_kv(x) 
        cq = self.down_proj_q(x) # separate q compression
        KR = self.k_rotary(x) # shared rotary key (B, T, dh_rotary)

        if use_cache:
            if self.kv_cache is None:
                self.kv_cache = ckv
                self.kr_cache = KR
            else:
                self.kv_cache = torch.cat([self.kv_cache, ckv], dim=1)
                self.kr_cache = torch.cat([self.kr_cache, KR], dim=1)
                
                if self.kv_cache.size(1) > block_size:
                    self.kv_cache = self.kv_cache[:, -block_size:, :]
                    self.kr_cache = self.kr_cache[:, -block_size:, :]

            KC = self.up_k(self.kv_cache) # still reconstruct k
            VC = self.up_v(self.kv_cache) 
        else:
            KC = self.up_k(ckv) 
            VC = self.up_v(ckv) 

        L = KC.size(1) 
        if use_cache:
            KR = self.kr_cache

        # apply RoPE
        cos_k = self.cos[:L].unsqueeze(0) # (1, L, dh_rotary/2)
        sin_k = self.sin[:L].unsqueeze(0) # (1, L, dh_rotary/2)
        KR = apply_rope(KR, cos_k, sin_k)
        KR = KR.unsqueeze(2)   # (B, L, 1, dh_rotary)
        KR = KR.expand(-1, -1, self.n_head, -1) # (B, L, n_head, dh_rotary)
        KC = KC.view(B, L, n, dh_nr) 
        K = torch.cat([KC, KR], dim=-1) # (B, L, n_head, dh)
        K = K.transpose(1, 2) # (B, n_head, L, dh)

        QC = self.up_q(cq)
        QR = self.q_rotary(cq) # B, T, n * (dh - dh_nr)
        QR = QR.view(B, T, n, self.dh_rotary)
        if use_cache:
            start = self.cache_pos 
        else:
            start = 0
        end = start + T
        cos_q = self.cos[start:end].unsqueeze(0).unsqueeze(2)  # (1, T, 1, dh_rotary/2)
        sin_q = self.sin[start:end].unsqueeze(0).unsqueeze(2)  # (1, T, 1, dh_rotary/2)
        QR = apply_rope(QR, cos_q, sin_q)
        # QR = QR.view(B, T, n, dh - dh_nr)
        QC = QC.view(B, T, n, dh_nr)
        Q = torch.cat([QC, QR], dim=-1)
        Q = Q.transpose(1, 2) # (B, n_head, T, dh)
        
        VC = VC.view(B, L, n,dh) # (B, L, n_head, dh)
        VC = VC.transpose(1, 2) # (B, n_head, L, dh)

        if use_cache:
            self.cache_pos += T

        wei = Q @ K.transpose(-2,-1) / math.sqrt(dh) # (B, H, T, D) @ (B, H, D, T) -> (B, H, T, T)
        # During training (or prefill), apply causal mask.
        # During cached decoding (T == 1), no mask is needed.
        if not (use_cache and T == 1):
            wei = wei.masked_fill(
                self.tril[:T, :L] == 0,
                float("-inf")
            )
        wei = F.softmax(wei, dim=-1) # (B, H, T, T)
        wei = self.dropout(wei)
        out = wei @ VC # (B, H, T, T) @ (B, H, T, D) -> (B, H, T, D)

        # (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        out = self.proj(out)
        out = self.dropout(out)

        return out

    def reset_cache(self):
        self.kv_cache = None
        self.kr_cache = None
        self.cache_pos = 0