# miniLLM

### Project Structure
```text
.
├── configs/
│   ├── base.py                         # Default baseline config for experiments
│   └── finetune_shakespeare.py         # Config overrides for finetuning on Shakespeare
│
├── data/
│   ├── dataset.py                      # Dataset loading, tokenization and batch generation
│   ├── shakespeare/
│   │   └── prepare.py                  # Shakespeare dataset preparation (BPE, GPT-2 compatible)
│   └── shakespeare_char/
│       └── prepare.py                  # Shakespeare dataset preparation (character-level)
│
├── model/
│   ├── __init__.py
│   ├── README.md                       # Design notes and open questions
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
│   │   ├── mla_naive.py                # Naive Multi-head Latent Attention
│   │   ├── mla_without_weight_absorption.py # DeepSeek-style MLA implementation (without weight absorption)
│   │   ├── mla_with_weight_absorption.py # DeepSeek MLA with weight absorption
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
│       ├── layernorm.py                # LayerNorm implemention
│       └── rmsnorm.py                  # RMSNorm implementation
│
├── notebooks/
│   ├── moe_mla_sizing.ipynb            # MoE / MLA parameter and memory sizing
│   └── moe_scaling_practice.ipynb      # MoE scaling practice notebook
│
├── train.py                            # Main training loop, evaluation and checkpointing
├── toy_train.py                        # Simplified training script for quick experiments
│
├── profiler.py                         # PyTorch profiling (example script, TODO: read more about profiling)
├── configurator.py                     # CLI configuration overrides
├── sample.py                           # Text generation from a checkpoint (main script)
├── generate.py                         # Minimal generation script, kept for understanding
├── export.py                           # ONNX model export
│
├── licenses/
│   └── nanogpt_LICENSE
├── README.md
└── LICENSE
```