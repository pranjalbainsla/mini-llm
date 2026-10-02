from .mha import MultiHeadAttention
from .mha_optimized import MultiHeadAttentionOptimized
from .gqa import GroupedQueryAttention
from .mla_without_weight_absorption import MultiheadLatentAttentionDeepSeek
from .mla_with_weight_absorption import MLADeepSeekOptimized


ATTENTION_REGISTRY = {
    "mha": MultiHeadAttention,
    "mha_optimized": MultiHeadAttentionOptimized,
    "gqa": GroupedQueryAttention,
    "mla_deepseek": MultiheadLatentAttentionDeepSeek,
    "mla_deepseek_optimized": MLADeepSeekOptimized,
}