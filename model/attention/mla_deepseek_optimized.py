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

class MLADeepSeekOptimized(nn.Module):

    def __init__(self, n_embd, n_head, latent_kv_dim, latent_q_dim):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.dh = n_embd // n_head 
        self.dh_non_rotary = 3 * self.dh // 4
        self.dh_rotary = self.dh - self.dh_non_rotary
        cos, sin = precompute_freqs(self.dh_rotary, max_seq_len, device)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

        self.kv_cache = None
        self.kr_cache = None
        self.cache_pos = 0

        self.down_proj_kv = nn.Linear(n_embd, latent_kv_dim)
        self.up_k = nn.Linear(latent_kv_dim, n_head * self.dh_non_rotary) 
        self.k_rotary = nn.Linear(n_embd, self.dh_rotary) # During attention you'll broadcast this same rotary key to every head.

        self.up_v = nn.Linear(latent_kv_dim, n_head * self.dh)
        self.down_proj_q = nn.Linear(n_embd, latent_q_dim)
        self.up_q = nn.Linear(latent_q_dim, n_head * self.dh_non_rotary)
        self.q_rotary = nn.Linear(latent_q_dim, n_head * self.dh_rotary)

        self.register_buffer('tril', torch.tril(torch.ones(max_seq_len, max_seq_len)))
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, use_cache, use_weight_absorption):
        B, T, C = x.shape

        # ------------------- input projection -------------------------
        ckv = self.down_proj_kv(x) 
        cq = self.down_proj_q(x) # separate q compression
        K_rope = self.k_rotary(x) # shared rotary key (B, T, dh_rotary)

        # --------------------- cache update ---------------------------
        if use_cache:
            if self.kv_cache is None:
                self.kv_cache = ckv
                self.kr_cache = K_rope
            else:
                self.kv_cache = torch.cat([self.kv_cache, ckv], dim=1)
                self.kr_cache = torch.cat([self.kr_cache, K_rope], dim=1)
                
                if self.kv_cache.size(1) > block_size:
                    self.kv_cache = self.kv_cache[:, -block_size:, :]
                    self.kr_cache = self.kr_cache[:, -block_size:, :]
            self.cache_pos += T

        # --------------- rotary parts --------------------------------
        if use_cache:
            K_rope = self.kr_cache

        L = K_rope.size(1) 

        # apply RoPE
        cos_k = self.cos[:L].unsqueeze(0) # (1, L, dh_rotary/2)
        sin_k = self.sin[:L].unsqueeze(0) # (1, L, dh_rotary/2)
        K_rope = apply_rope(K_rope, cos_k, sin_k)
        K_rope = K_rope.unsqueeze(2)   # (B, L, 1, dh_rotary)
        K_rope = K_rope.expand(-1, -1, self.n_head, -1) # (B, L, n_head, dh_rotary)

        Q_rope = self.q_rotary(cq) # B, T, n * (dh - dh_nr)
        Q_rope = Q_rope.view(B, T, self.n_head, self.dh_rotary)
        if use_cache:
            start = self.cache_pos 
        else:
            start = 0
        end = start + T
        cos_q = self.cos[start:end].unsqueeze(0).unsqueeze(2)  # (1, T, 1, dh_rotary/2)
        sin_q = self.sin[start:end].unsqueeze(0).unsqueeze(2)  # (1, T, 1, dh_rotary/2)
        Q_rope = apply_rope(Q_rope, cos_q, sin_q)
        Q_rope = Q_rope.transpose(1, 2)

        # ---------------------- attention ----------------------------
        if use_weight_absorption:
            # Attention path using absorbed weights

            # reshaping to add head dimension
            Wq = self.up_q.weight.view(self.n_head, self.dh_non_rotary, -1)
            Wk = self.up_k.weight.view(self.n_head, self.dh_non_rotary, -1) 

            Q_absorb = torch.einsum(
                "btl,hdl,hdk->bhtk",
                cq,
                Wq,
                Wk
            )  # (B, n_head, T, latent_kv_dim)

            K_latent = self.kv_cache.unsqueeze(1)  # (B, 1, L, latent_kv_dim)
            # you simply cannot have use_cache=False if you have turned on use_weight_absorption (it's senseless)
            wei = (Q_absorb @ K_latent.transpose(-2, -1) + Q_rope @ K_rope.transpose(-2, -1)) / math.sqrt(self.dh) 
            # scale it by dh only because dh = dh_nr + dh_r (by choice)

        else:
            # original MLA path
            if use_cache:
                KC = self.up_k(self.kv_cache) 
            else:
                KC = self.up_k(ckv)

            KC = KC.view(B, L, self.n_head, self.dh_non_rotary) 
            K = torch.cat([KC, K_rope], dim=-1) # (B, L, n_head, dh)
            K = K.transpose(1, 2) # (B, n_head, L, dh)

            QC = self.up_q(cq)
            QC = QC.view(B, T, self.n_head, self.dh_non_rotary)

            Q = torch.cat([QC, Q_rope], dim=-1)
            Q = Q.transpose(1, 2) # (B, n_head, T, dh)

            wei = Q @ K.transpose(-2,-1) / math.sqrt(self.dh) # (B, H, T, D) @ (B, H, D, T) -> (B, H, T, T)
        
        # ---------------------------------------------------
        # During training (or prefill), apply causal mask.
        # During cached decoding (T == 1), no mask is needed.
        if not (use_cache and T == 1):
            wei = wei.masked_fill(
                self.tril[:T, :L] == 0,
                float("-inf")
            )
        wei = F.softmax(wei, dim=-1) # (B, H, T, T)
        wei = self.dropout(wei)

        # ------------- output ----------------------
        # TODO: go over this again

        if use_weight_absorption:
            # (H, dh, latent_kv)
            Wv = self.up_v.weight.view(self.n_head, self.dh, -1)

            # (C, H, dh)
            Wo = self.proj.weight.view(C, self.n_head, self.dh)

            # (H, C, latent_kv)
            W_out_absorb = torch.einsum(
                "chd,hdl->hcl",
                Wo,
                Wv
            )
            latent_values = self.kv_cache if use_cache else ckv
            latent_out = wei @ latent_values.unsqueeze(1)   # (B, H, T, latent_kv_dim)
            out = torch.einsum("bhtl,hcl->btc", latent_out, W_out_absorb)
        
        else:
            if use_cache:
                VC = self.up_v(self.kv_cache)
            else:
                VC = self.up_v(ckv)
            
            VC = VC.view(B, L, self.n_head, self.dh)
            VC = VC.transpose(1, 2)   # (B, H, L, dh)
            out = wei @ VC # (B, H, T, T) @ (B, H, T, D) -> (B, H, T, D)

            out = out.transpose(1, 2).contiguous().view(B, T, C)
            out = self.proj(out)

        out = self.dropout(out)

        return out

    def reset_cache(self):
        self.kv_cache = None
        self.kr_cache = None
        self.cache_pos = 0