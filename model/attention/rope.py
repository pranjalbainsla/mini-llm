import torch

def precompute_freqs(head_size, max_seq_len, device):
    theta = 1.0 / (
        10000 ** (torch.arange(0, head_size, 2, device=device).float() / head_size)
    )
    positions = torch.arange(max_seq_len, device=device) # max_seq_len == block_size == T
    freqs = torch.outer(positions, theta)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin
  
def apply_rope(x, cos, sin):
    x_even = x[..., ::2]
    x_odd  = x[..., 1::2]

    x_rot_even = x_even * cos - x_odd * sin
    x_rot_odd  = x_even * sin + x_odd * cos

    x = torch.stack((x_rot_even, x_rot_odd), dim=-1)
    return x.flatten(-2)