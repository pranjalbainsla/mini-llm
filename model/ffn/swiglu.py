import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, config):

        super().__init__()

        hidden_dim = int(8 * config.n_embd / 3)

        self.gate = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.up = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, config.n_embd, bias=False)

    def forward(self, x):
        gate = F.silu(self.gate(x))
        value = self.up(x)

        return self.down(gate * value), None