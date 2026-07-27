import torch 
import torch.nn as nn
from .builder import build_ffn
from .builder import build_attention
from .builder import build_norm

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.attn = build_attention()
        # self.ffwd = FeedFoward(n_embd)
        # self.moe = MoE(n_embd, num_experts)
        self.ffn = build_ffn()
        self.ln1 = build_norm()
        self.ln2 = build_norm()

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        ffn_out, aux_loss = self.ffn(self.ln2(x))
        x = x + ffn_out

        return x, aux_loss