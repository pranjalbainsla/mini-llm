from .attention import ATTENTION_REGISTRY
from .ffn import FFN_REGISTRY
from .norm import NORM_REGISTRY


def build_attention(config):
    try:
        attention_cls = ATTENTION_REGISTRY[config.attention]
    except KeyError:
        raise ValueError(f"Unknown Attention: {config.attention}")

    return attention_cls(config)


def build_ffn(config):
    try:
        ffn_cls = FFN_REGISTRY[config.ffn]
    except KeyError:
        raise ValueError(f"Unknown FFN: {config.ffn}")

    return ffn_cls(config)


def build_norm(config):
    try:
        norm_cls = NORM_REGISTRY[config.norm]
    except KeyError:
        raise ValueError(f"Unknown Norm: {config.norm}")

    return norm_cls(config)