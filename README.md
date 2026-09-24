# miniLLM

### Project Structure
```text
.
├── configs/
│   └── base.py                         # Default model, architecture and training configuration
│
├── data/
│   ├── dataset.py                      # Dataset loading, tokenization and batch generation
│   └── shakespeare_char/
│       └── prepare.py                  # Shakespeare dataset preparation (character-level)
│
├── model/
│   ├── __init__.py              
│   ├── builder.py                      # Builds attention, FFN and normalization modules from config
│   ├── block.py                        # Transformer block
│   ├── gpt_moe.py                      # Main configurable GPT model
│   ├── gpt_nanogpt.py                  # nanoGPT-style reference GPT implementation
│   │
│   ├── attention/
│   │   ├── __init__.py                 # Attention registry
│   │   ├── mha.py                      # Basic multi-head self-attention
│   │   ├── mha_optimized.py            # Optimized MHA with KV caching
│   │   ├── gqa.py                      # Grouped-Query Attention with KV caching
│   │   ├── mla.py                      # Basic Multi-head Latent Attention
│   │   ├── mla_deepseek.py             # DeepSeek-style MLA implementation
│   │   ├── mla_deepseek_optimized.py   # Optimized DeepSeek MLA with weight absorption
│   │   └── rope.py                     # Rotary positional embedding utilities
│   │
│   ├── ffn/
│   │   ├── __init__.py                 # FFN registry
│   │   ├── mlp.py                      # Standard Transformer MLP
│   │   ├── swiglu.py                   # SwiGLU feed-forward network
│   │   ├── moe.py                      # Vanilla top-k Mixture-of-Experts
│   │   └── moe_deepseek.py             # DeepSeek-style MoE with shared/routed experts
│   │
│   └── norm/
│       ├── __init__.py                 # Normalization registry
│       ├── layernorm.py                # LayerNorm implemented from scratch
│       └── rmsnorm.py                  # RMSNorm implementation
│
├── train.py                            # Main training loop, evaluation and checkpointing
├── toy_train.py                        # Simplified training script for quick experiments
|
├── profiler.py                         # PyTorch profiling (example script, TODO: read more about profiling)
├── configurator.py                     # CLI configuration overrides
├── generate.py                         # Text generation from a checkpoint
├── export.py                           # ONNX model export
|
├── licenses                
├── README.md
└── LICENSE                    
```