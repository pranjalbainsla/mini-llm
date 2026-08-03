import torch

def precompute_freqs(head_dim, max_seq_len, device):
    theta = 1.0 / (
        10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    positions = torch.arange(max_seq_len, device=device)
    freqs = torch.outer(positions, theta)
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