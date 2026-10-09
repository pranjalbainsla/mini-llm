# miniLLM

An ML playground for simplifying modern LLM systems and their architecture, getting to the crux of each idea or algorithm (attention variants, FFN/MoE, normalization) in small, readable, swappable modules.

### Quick start: train the configurable GPT

The configurable model is [model/gpt_moe.py](model/gpt_moe.py). It is trained by [toy_train.py](toy_train.py), and every architectural choice comes from a config file plus optional CLI overrides.

```bash
pip install torch numpy matplotlib   # matplotlib only for plotting scripts

# train the baseline (mha + mlp + layernorm + rope) on tiny shakespeare, char-level
# data/input.txt is downloaded automatically on first run
python toy_train.py configs/base.py

# swap components with --key=value overrides
python toy_train.py configs/base.py --ffn=swiglu --norm=rmsnorm
python toy_train.py configs/base.py --ffn=moe_deepseek --num_experts=16 --k=4 --num_shared_experts=2

# save curves, best val loss and params to results/<run_name>.json
python toy_train.py configs/base.py --run_name=my_run
```

| knob | options |
|---|---|
| `attention` | `mha`, `mha_optimized`, `gqa`, `mla_naive`, `mla_noabs`, `mla_abs` |
| `ffn` | `mlp`, `swiglu`, `moe`, `moe_deepseek` |
| `norm` | `layernorm`, `rmsnorm` |

Sizes and training knobs (`n_embd`, `n_layer`, `max_iters`, `learning_rate`, MoE/MLA settings, ...) are all in [configs/base.py](configs/base.py). An unknown key raises an error. To train on a different corpus, put your text at `data/input.txt`.

Before trusting a new module, run `python scripts/sanity.py` (shape, gradient, causality and overfit checks). For multi-seed runs see [experiments/run_seeds.sh](experiments/run_seeds.sh), and for my experimentation log, see [docs/log.md](docs/log.md). All my experiments were run on a Tesla T4.

### Project Structure
```text
.
├── configs/
│   └── base.py                         # Default baseline config for experiments
│
├── data/
│   ├── dataset.py                      # Char-level tokenizer + batcher over data/input.txt (auto-downloaded)
│   └── shakespeare_char/
│       └── prepare.py                  # Tiny Shakespeare preparation (character-level, for train.py)
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
│   │   ├── mla_naive.py                # MLA, naive (RoPE on the full up-projected K)
│   │   ├── mla_without_weight_absorption.py # MLA, decoupled RoPE, no weight absorption
│   │   ├── mla_with_weight_absorption.py # MLA, decoupled RoPE, with weight absorption
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
├── train.py                            # nanoGPT-style training loop (for gpt_nanogpt.py), evaluation and checkpointing
├── toy_train.py                        # Training loop for the configurable GPT (gpt_moe.py), writes results/<run_name>.json
├── sample.py                           # Text generation from a checkpoint
├── configurator.py                     # CLI configuration overrides (exec'd by the entry points above)
│
├── experiments/                        # Ablation tooling; run from the repo root
│   ├── run_seeds.sh                    # One config over several seeds
│   ├── ablate.py                       # Resumable queue runner for Colab, syncs results/ to Drive
│   ├── summarize.py                    # One summary row + verdict per tag, log-entry skeleton, plot
│   └── plot_seeds.py                   # Per-seed strip plot against the baseline noise band
│
├── scripts/
│   ├── sanity.py                       # Shape / gradient / causality / overfit checks
│   ├── plot_noshared_sweep.py          # MoE granularity (E/k) vs dense plot, incl. expert-load panels
│   └── profiler.py                     # PyTorch profiling (example script, TODO: read more about profiling)
│
├── results/                            # Per-run JSON (<tag>_s<seed>.json)
├── plots/                              # Generated figures
├── docs/
│   ├── log.md                          # Experiment log (hypothesis / config / results / takeaway)
│   └── notebooks/
│       ├── moe_mla_sizing.ipynb        # MoE / MLA parameter and memory sizing
│       └── moe_scaling_practice.ipynb  # MoE scaling practice notebook
│
├── licenses/
│   └── nanogpt_LICENSE
├── README.md
└── LICENSE
```