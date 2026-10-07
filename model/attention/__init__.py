from .mha import MultiHeadAttention
from .mha_optimized import MultiHeadAttentionOptimized
from .gqa import GroupedQueryAttention
from .mla_naive import MultiheadLatentAttention
from .mla_without_weight_absorption import MultiheadLatentAttentionDeepSeek
from .mla_with_weight_absorption import MLADeepSeekOptimized


ATTENTION_REGISTRY = {
    "mha": MultiHeadAttention,
    "mha_optimized": MultiHeadAttentionOptimized,
    "gqa": GroupedQueryAttention,
    "mla_naive": MultiheadLatentAttention,        # latent KV, RoPE on the full up-projected K
    "mla_noabs": MultiheadLatentAttentionDeepSeek,     # latent KV cache + decoupled RoPE, no weight absorption
    "mla_abs": MLADeepSeekOptimized,         # latent KV cache + decoupled RoPE, weight absorption 
}
