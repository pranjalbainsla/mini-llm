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
        self.attn = build_attention()
        # self.ffwd = FeedFoward(n_embd)
        # self.moe = MoE(n_embd, num_experts)
        self.ffn = build_ffn()
        self.ln1 = build_norm()
        self.ln2 = build_norm()

    def forward(self, x, use_cache, use_weight_absorption):
        x = x + self.attn(self.ln1(x), use_cache=use_cache, use_weight_absorption=use_weight_absorption)
        # ffn_out, aux_loss  = self.ffn(self.ln2(x))
        ffn_out, topk_idx = self.ffn(self.ln2(x))
        x = x + ffn_out

        # return x, aux_loss
        return x, topk_idx