import torch.nn as nn

from .attention.mha import MultiHeadAttention
from .attention.mha_optimized import MultiHeadAttentionOptimized
from .attention.gqa import GroupedQueryAttention
from .attention.mla_deepseek import MultiheadLatentAttentionDeepSeek
from .attention.mla_deepseek_optimized import MLADeepSeekOptimized

from .ffn.mlp import FeedForward
from .ffn.swiglu import SwiGLU
from .ffn.moe import MoE
from .ffn.moe_deepseek import MoEDeepSeek

from .norm.rmsnorm import RMSNorm


def build_ffn(config):
    if config.ffn == "mlp":
        return FeedForward(config)

    elif config.ffn == "moe":
        return MoE(config)
    
    elif config.ffn == "moe_deepseek":
        return MoEDeepSeek(config)
    
    elif config.ffn == "swiglu":
        return SwiGLU(config)

    else:
        raise ValueError(f"Unknown FFN: {config.ffn}")

def build_attention(config):

    if config.attention == "mha":
        return MultiHeadAttention(config)
    
    elif config.attention == "mha_optimized":
        return MultiHeadAttentionOptimized(config)

    elif config.attention == "gqa":
        return GroupedQueryAttention(config)

    elif config.attention == "mla_deepseek":
        return MultiheadLatentAttentionDeepSeek(config)
    
    elif config.attention == "mla_deepseek_optimized":
        return MLADeepSeekOptimized(config)
    
    raise ValueError(f"Unknown Attention: {config.attention}")

def build_norm(config):

    if config.norm == "layernorm":
        return nn.LayerNorm(config.n_embd)

    elif config.norm == "rmsnorm":
        return RMSNorm(config)

    raise ValueError(f"Unknown Norm: {config.norm}")