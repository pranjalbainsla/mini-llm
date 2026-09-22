from .mlp import FeedForward
from .swiglu import SwiGLU
from .moe import MoE
from .moe_deepseek import MoEDeepSeek


FFN_REGISTRY = {
    "mlp": FeedForward,
    "swiglu": SwiGLU,
    "moe": MoE,
    "moe_deepseek": MoEDeepSeek,
}