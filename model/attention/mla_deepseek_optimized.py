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
        self.dh = n_embd // n_head # Dimension of each attention head
        ROTARY_RATIO = 0.25  
        self.dh_rotary = int(self.dh * ROTARY_RATIO)
        self.dh_non_rotary = self.dh - self.dh_rotary 

        cos, sin = precompute_freqs(self.dh_rotary, max_seq_len, device)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

        self.kv_cache = None
        self.kr_cache = None
        self.cache_pos = 0

        self.down_proj_kv = nn.Linear(n_embd, latent_kv_dim, bias=False)
        self.down_proj_q = nn.Linear(n_embd, latent_q_dim, bias=False)

        self.k_rotary = nn.Linear(n_embd, self.dh_rotary, bias=False) # Shared rotary key is broadcast to every head to avoid storing per-head rotary keys.
        self.q_rotary = nn.Linear(latent_q_dim, n_head * self.dh_rotary, bias=False)

        self.up_k = nn.Linear(latent_kv_dim, n_head * self.dh_non_rotary, bias=False) 
        self.up_v = nn.Linear(latent_kv_dim, n_head * self.dh, bias=False)
        self.up_q = nn.Linear(latent_q_dim, n_head * self.dh_non_rotary, bias=False)
        
        self.register_buffer('tril', torch.tril(torch.ones(max_seq_len, max_seq_len)))
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, use_cache, use_weight_absorption):
        B, T, C = x.shape

        # ------------------- compress inputs -------------------------
        ckv = self.down_proj_kv(x)  # (B, T, latent_kv_dim)
        cq = self.down_proj_q(x) # (B, T, latent_q_dim)
        K_rope = self.k_rotary(x) # shared rotary key (B, T, dh_rotary)

        # -------------------- update cache ---------------------------
        if use_cache:
            if self.kv_cache is None:
                self.kv_cache = ckv
                self.kr_cache = K_rope
            else:
                self.kv_cache = torch.cat([self.kv_cache, ckv], dim=1) # (B, L, latent_kv_dim)
                self.kr_cache = torch.cat([self.kr_cache, K_rope], dim=1) # (B, L, dh_rotary)
                
                if self.kv_cache.size(1) > block_size:
                    self.kv_cache = self.kv_cache[:, -block_size:, :] # (B, T, latent_kv_dim)
                    self.kr_cache = self.kr_cache[:, -block_size:, :] # (B, T, dh_rotary)


        # ---------------------- apply RoPE ---------------------------
        if use_cache:
            K_rope = self.kr_cache

        L = K_rope.size(1) # Current KV cache length (sequence length for attention)
        cos_k = self.cos[:L].unsqueeze(0) # (1, L, dh_rotary/2)
        sin_k = self.sin[:L].unsqueeze(0) # (1, L, dh_rotary/2)
        K_rope = apply_rope(K_rope, cos_k, sin_k)
        K_rope = K_rope.unsqueeze(2)   # (B, L, 1, dh_rotary)
        K_rope = K_rope.expand(-1, -1, self.n_head, -1) # (B, L, n_head, dh_rotary)

        Q_rope = self.q_rotary(cq) # (B, T, n_head * dh_rotary)
        Q_rope = Q_rope.view(B, T, self.n_head, self.dh_rotary) # (B, T, n_head, dh_rotary)

        if use_cache:
            start = self.cache_pos 
        else:
            start = 0
        end = start + T
        # K_rope contains the entire cached sequence, so positions are simply 0...L-1
        # Q_rope contains only the newly generated token(s), so RoPE must start at cache_pos
        cos_q = self.cos[start:end].unsqueeze(0).unsqueeze(2)  # (1, T, 1, dh_rotary/2)
        sin_q = self.sin[start:end].unsqueeze(0).unsqueeze(2)  # (1, T, 1, dh_rotary/2)
        Q_rope = apply_rope(Q_rope, cos_q, sin_q) # (B, T, n_head, dh_rotary)

        if use_cache:
            self.cache_pos += T 

        # ------------------------ attention --------------------------
        if use_weight_absorption:
            # Compute attention scores directly in latent space using absorbed weights

            # Reshaping to add head dimension
            Wq = self.up_q.weight.view(self.n_head, self.dh_non_rotary, -1) # (n_head, dh_non_rotary, latent_q_dim)
            Wk = self.up_k.weight.view(self.n_head, self.dh_non_rotary, -1) # (n_head, dh_non_rotary, latent_kv_dim)

            # Each head produces a latent-space query instead of reconstructing full K.
            Q_absorb = torch.einsum(
                "btq,hdq,hdk->bhtk",
                cq,
                Wq,
                Wk
            )  # (B, n_head, T, latent_kv_dim)
            # you simply cannot have use_cache=False if you have turned on use_weight_absorption (it's senseless)
            K_latent = self.kv_cache.unsqueeze(1)  # (B, 1, L, latent_kv_dim)
            wei = (Q_absorb @ K_latent.transpose(-2, -1) + Q_rope @ K_rope.transpose(-2, -1)) / math.sqrt(self.dh) 

        else:
            # original MLA path
            if use_cache:
                KC = self.up_k(self.kv_cache) # (B, L, n_head * dh_non_rotary)
            else:
                KC = self.up_k(ckv)

            KC = KC.view(B, L, self.n_head, self.dh_non_rotary) # (B, L, n_head, dh_non_rotary)
            K = torch.cat([KC, K_rope], dim=-1) # (B, L, n_head, dh)
            K = K.transpose(1, 2) # (B, n_head, L, dh)

            QC = self.up_q(cq) # (B, T, n_head * dh_non_rotary)
            QC = QC.view(B, T, self.n_head, self.dh_non_rotary) # (B, T, n_head, dh_non_rotary)

            Q = torch.cat([QC, Q_rope], dim=-1) # (B, T, n_head, dh)
            Q = Q.transpose(1, 2) # (B, n_head, T, dh)

            wei = Q @ K.transpose(-2,-1) / math.sqrt(self.dh) # (B, n_head, T, dh) @ (B, n_head, dh, L) -> (B, n_head, T, L)
        
        # During training/prefill, each query can only attend to past tokens, so apply a causal mask.
        # During autoregressive decoding (T == 1), the single query is already the newest token, so every cached key is valid and no mask is needed.
        if not (use_cache and T == 1):
            wei = wei.masked_fill(
                self.tril[:T, :L] == 0,
                float("-inf")
            )
        wei = F.softmax(wei, dim=-1) # (B, n_head, T, L)
        wei = self.dropout(wei)

        # ------------- compute output ----------------------
        if use_weight_absorption: 
            Wv = self.up_v.weight.view(self.n_head, self.dh, -1) # (n_head, dh, latent_kv_dim)

            Wo = self.out_proj.weight.view(C, self.n_head, self.dh) # (n_embd, n_head, dh)

            W_out_absorb = torch.einsum(
                "chd,hdl->hcl",
                Wo,
                Wv
            ) # (n_head, n_embd, latent_kv_dim)
            latent_values = self.kv_cache if use_cache else ckv # (B, L, latent_kv_dim) if cached, else (B, T, latent_kv_dim)
            latent_out = wei @ latent_values.unsqueeze(1)   # (B, H, T, latent_kv_dim)
            out = torch.einsum("bhtl,hcl->btc", latent_out, W_out_absorb) # (B, n_head, T, latent_kv_dim) @ (n_head, n_embd, latent_kv_dim) -> (B, T, n_embd)
        
        else:
            if use_cache:
                VC = self.up_v(self.kv_cache) # (B, L, n_head * dh)
            else:
                VC = self.up_v(ckv) # (B, L, n_head * dh)
            
            VC = VC.view(B, L, self.n_head, self.dh) # (B, L, n_head, dh)
            VC = VC.transpose(1, 2)   # (B, n_head, L, dh)
            out = wei @ VC # (B, n_head, T, L) @ (B, n_head, L, dh) -> (B, n_head, T, dh)

            out = out.transpose(1, 2).contiguous().view(B, T, C) # (B, n_head, T, dh) -> (B, T, n_head * dh)    
            out = self.out_proj(out)

        out = self.dropout(out)

        return out

    def reset_cache(self):
        self.kv_cache = None
        self.kr_cache = None
        self.cache_pos = 0