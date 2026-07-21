import torch 
import torch.nn as nn
from .attention.mha import MultiHeadAttention
from .ffn.moe import MoE
from config import num_experts

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        # self.ffwd = FeedFoward(n_embd)
        self.moe = MoE(n_embd, num_experts)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, cos, sin):
        x = x + self.sa(self.ln1(x), cos, sin)
        x = x + self.moe(self.ln2(x))
        return x