from config import *
import torch.nn as nn
from model.ffn.mlp import FeedForward
from model.ffn.moe import MoE
from model.attention.mha import MultiHeadAttention


def build_ffn():
    if ffn == "mlp":
        return FeedForward(n_embd)

    elif ffn == "moe":
        return MoE(n_embd, num_experts)

    else:
        raise ValueError(f"Unknown FFN: {ffn}")

def build_attention():

    if attention == "mha":
        return MultiHeadAttention(n_head, n_embd // n_head) # head_size = n_embd // n_head

    # elif attention == "gqa":
    #     return GroupedQueryAttention(...)

    # elif attention == "mla":
    #     return MLA(...)

    raise ValueError(f"Unknown Attention: {attention}")

def build_norm():

    if norm == "layernorm":
        return nn.LayerNorm(n_embd)

    # elif norm == "rmsnorm":
    #     return RMSNorm(n_embd)

    raise ValueError(f"Unknown Norm: {norm}")