import torch.nn as nn

from .rmsnorm import RMSNorm

NORM_REGISTRY = {
    "layernorm": lambda config: nn.LayerNorm(config.n_embd),
    "rmsnorm": RMSNorm,
}