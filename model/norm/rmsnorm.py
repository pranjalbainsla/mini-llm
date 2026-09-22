import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.eps = config.rmsnorm_eps
        self.weight = nn.Parameter(torch.ones(config.n_embd))

    def forward(self, x):
        rms = torch.sqrt(
            torch.mean(x * x, dim=-1, keepdim=True) + self.eps
        )

        x = x / rms
        return x * self.weight