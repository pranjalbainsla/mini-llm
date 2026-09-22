from .mha import MultiHeadAttention
from .mha_optimized import MultiHeadAttentionOptimized
from .gqa import GroupedQueryAttention
from .mla_deepseek import MultiheadLatentAttentionDeepSeek
from .mla_deepseek_optimized import MLADeepSeekOptimized


ATTENTION_REGISTRY = {
    "mha": MultiHeadAttention,
    "mha_optimized": MultiHeadAttentionOptimized,
    "gqa": GroupedQueryAttention,
    "mla_deepseek": MultiheadLatentAttentionDeepSeek,
    "mla_deepseek_optimized": MLADeepSeekOptimized,
}