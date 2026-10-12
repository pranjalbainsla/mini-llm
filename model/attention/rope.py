import torch
import torch.nn as nn

# Legacy: per-layer cos/sin table indexed by cache_pos, which overflows past block_size.
# Remove once mha, gqa, mla_naive, mla_noabs and mla_abs consume the (cos, sin) passed in by
# GPT via RotaryEmbedding (mha_optimized already does).
def precompute_freqs(head_dim, block_size):
    theta = 1.0 / (
        10000 ** (torch.arange(0, head_dim, 2).float() / head_dim)
    )
    positions = torch.arange(block_size)
    freqs = torch.outer(positions, theta)
    return freqs.cos(), freqs.sin()

class RotaryEmbedding(nn.Module):
    """Computes (cos, sin) once per forward from explicit position_ids."""

    def __init__(self, dim, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids):  # (B, T) int
        freqs = position_ids[..., None].float() * self.inv_freq  # (B, T, dim/2)
        return freqs.cos(), freqs.sin()

def apply_rope(x, cos, sin):
    """
    x:   (..., head_dim)
    cos: broadcastable to (..., head_dim//2)
    sin: broadcastable to (..., head_dim//2)
    """

    x_even = x[..., 0::2]
    x_odd  = x[..., 1::2]

    out_even = x_even * cos - x_odd * sin
    out_odd  = x_even * sin + x_odd * cos

    return torch.stack((out_even, out_odd), dim=-1).flatten(-2)