
# mini-llm — codebase index

Educational PyTorch LLM repo: a nanoGPT reference model plus a from-scratch, config-driven
GPT with swappable attention (MHA / GQA / MLA), FFN (MLP / SwiGLU / MoE) and norm modules.
No tests, no package manifest, no requirements file. Depends on `torch`, `numpy`, and optionally
`tiktoken`, `requests`, `transformers`, `wandb`. Runs from the repo root; scripts use
relative paths such as `configurator.py` and `data/input.txt`.

## Two independent model stacks

| | nanoGPT stack | Custom "MoE" stack |
|---|---|---|
| Model | `model/gpt_nanogpt.py` (`GPTConfig`, `GPT`) | `model/gpt_moe.py` (`GPT(vocab_size, config)`) |
| Trainer | `train.py` | `toy_train.py` |
| Data | `data/<dataset>/{train,val}.bin` (+ `meta.pkl`) via np.memmap | `data/dataset.py`: char-level from **`data/input.txt`**, loaded at import time |
| Config | globals in `train.py` + `configs/finetune_shakespeare.py` | `configs/base.py` (every key, no defaults in the model) |
| Checkpoint | `out/ckpt.pt` with `model_args` | `out/base_n{L}_h{H}_d{C}.pt` with `config` dict + `vocab_size` |
| `forward` returns | `(logits, loss)` | `(logits, loss, routing_info)` |

`sample.py` handles both. If a file in `out_dir` starts with `base_`, it loads the MoE stack. Otherwise it loads `ckpt.pt` with the nanoGPT stack.

## Config system
- `configurator.py` is `exec`'d inside scripts. Each positional arg is a config file that gets exec'd into
  globals. `--key=value` overrides an existing global; the key must already exist and the type must match.
- `toy_train.py` captures `config_keys` **after** the configurator runs. So every scalar in the config file
  becomes an attribute of `SimpleNamespace(**config)`, which the model reads (e.g. `config.n_embd`).
  To add a new knob, add it to `configs/base.py` and read `config.<name>` in the module. The trainer needs no changes.
- Keys in `configs/base.py`: `batch_size, block_size, max_iters, learning_rate, eval_interval,
  eval_iters, log_interval, n_embd, n_head, n_layer, n_kv_heads, bias, attention, ffn, norm,
  position, latent_kv_dim, latent_q_dim, rotary_ratio, num_experts, num_shared_experts, k,
  gamma, alpha, rmsnorm_eps, dropout, max_seq_len`.
- Nothing reads `position`. RoPE is always applied inside the attention modules.

## Custom model: how the pieces connect
```
GPT (model/gpt_moe.py)
  token_embedding_table -> [Block x n_layer] -> ln_f (nn.LayerNorm, always) -> lm_head
Block (model/block.py): x + attn(ln1(x)); x + ffn(ln2(x))
  built by model/builder.py: build_attention / build_ffn / build_norm
  -> lookup in ATTENTION_REGISTRY / FFN_REGISTRY / NORM_REGISTRY using config.attention / .ffn / .norm
```
**Interface contracts**
- **Attention.** A module that has `reset_cache()` is treated as "cached". `Block` forwards `**kwargs`
  (`use_cache`, `use_weight_absorption`) only to cached modules. Non-cached modules take `forward(x)`.
- **FFN.** Always returns `(out, topk_idx_or_None)`. `GPT.forward` collects the second values into
  `routing_info`. `GPT.update_expert_bias(routing_info)` calls `ffn.update_expert_bias` on each FFN that has it.
  `toy_train.py` calls this after every optimizer step.
- **Norm.** Constructor takes `(config)`.
- **KV cache.** Each cached attention module stores `k_cache`/`v_cache`, or `kv_cache`/`kr_cache` for MLA, plus `cache_pos`.
  RoPE offsets queries by `cache_pos`. The causal mask is skipped when `use_cache and T == 1`.
- **Aux loss.** `total_aux` in `GPT.forward` is hard-coded to `None`. Load balancing is done without an aux loss,
  through the expert bias.


## Registries
| key | class | file |
|---|---|---|
| `mha` | `MultiHeadAttention` (per-`Head` modules, RoPE, no cache) | `model/attention/mha.py` |
| `mha_optimized` | `MultiHeadAttentionOptimized` (batched heads, KV cache) | `model/attention/mha_optimized.py` |
| `gqa` | `GroupedQueryAttention` (`n_kv_heads`, `repeat_interleave`, KV cache) | `model/attention/gqa.py` |
| `mla_deepseek` | `MultiheadLatentAttentionDeepSeek` (latent KV + decoupled RoPE key, no absorption) | `model/attention/mla_deepseek.py` |
| `mla_deepseek_optimized` | `MLADeepSeekOptimized` (adds `use_weight_absorption` einsum path) | `model/attention/mla_deepseek_optimized.py` |
| *(not registered)* | `MultiheadLatentAttention` (naive MLA) | `model/attention/mla.py` |
| `mlp` | `FeedForward` (4x, ReLU) | `model/ffn/mlp.py` |
| `swiglu` | `SwiGLU` (hidden = 8/3·C) | `model/ffn/swiglu.py` |
| `moe` | `MoE` (softmax top-k, Switch-style aux loss) | `model/ffn/moe.py` |
| `moe_deepseek` | `MoEDeepSeek` (sigmoid scores, shared + routed experts, bias-based balancing) | `model/ffn/moe_deepseek.py` |
| `layernorm` | `nn.LayerNorm(n_embd)` (lambda) | `model/norm/__init__.py` |
| `rmsnorm` | `RMSNorm` (`rmsnorm_eps`) | `model/norm/rmsnorm.py` |

`model/norm/layernorm.py::LayerNorm1d` is for learning only and nothing uses it. It normalizes over dim 1.
`model/attention/rope.py` has `precompute_freqs(head_dim, max_seq_len)` and `apply_rope(x, cos, sin)`, which uses interleaved even/odd pairs.

## Symbol locations
- `model/gpt_moe.py`: `GPT.forward` :20, `reset_cache` :44, `generate` :49, `update_expert_bias` :73
- `model/gpt_nanogpt.py`: `CausalSelfAttention` :36, `MLP` :85, `Block` :101, `GPTConfig` :115, `GPT` :125,
  `from_pretrained` :213, `configure_optimizers` :270, `estimate_mfu` :296, `generate` :313
- `model/ffn/moe_deepseek.py`: routing :46-56, `update_expert_bias` :82
- `model/attention/mla_deepseek_optimized.py`: cache update :50, RoPE :63, absorbed scores :92, absorbed output :140
- `train.py`: default globals :23-67, `get_batch` :93, init_from branches :127-168, `estimate_loss` :193, `get_lr` :208, loop :233
- `toy_train.py`: defaults :35-55, `STRUCTURAL_KEYS` :95, resume :100, loop :167, ckpt save :175-186
- `sample.py`: checkpoint type detection :52-74, meta.pkl encoder override :80-93

## Other files
- `data/shakespeare_char/prepare.py`: char-level `train.bin`/`val.bin`/`meta.pkl` for `train.py`.
- `data/shakespeare/prepare.py`: GPT-2 BPE (tiktoken) bins, used with `configs/finetune_shakespeare.py` (`init_from='gpt2'`).
- `profiler.py`: a standalone `torch.profiler` demo on a toy FFN that writes `profile.json`. It is unrelated to the models.
- `generate.py`, `export.py`: **stale**. They import `model.gpt` and `config`, which no longer exist. Use `sample.py` instead of `generate.py`.
- Gitignored: `checkpoints/`, `onnx/`, `*.onnx`, `notes.txt`, `bench.py`, and several `*.ipynb` files.

## Known issues (found by reading the code, not by running it)
- `mha_optimized.py:70`, `mla_deepseek.py:101` and `mla_deepseek_optimized.py:58` use an undefined `block_size`. They raise
  NameError once the cache grows past one token (the `else` branch). `gqa.py:78` correctly uses `self.config.block_size`.
- `mla_deepseek_optimized.py` never sets `self.n_head`, so it raises AttributeError at :72.
- `mla.py` and `mla_deepseek.py` call `precompute_freqs(..., device="cpu")`, but that function takes no `device` argument, so they raise TypeError on construction.
  `mla_deepseek.py` also hard-codes a rotary fraction of 1/4 and ignores `rotary_ratio` and `bias`.
- `Block` passes no kwargs during training, but `mha_optimized`, `gqa` and `mla_deepseek*` require a positional `use_cache`, so training calls
  fail. `GPT.generate` always passes `use_weight_absorption`, which only `mla_deepseek_optimized` accepts.
- `gpt_moe.GPT.generate` crops the context only once. When `use_cache=False`, later steps feed a single token with no context.
- `moe.py:25` calls `Expert(config.n_embd)` with an int where `Expert` expects the config object. `MoE` also returns `aux_loss` in the slot where `topk_idx` is expected.
- `moe_deepseek.py:32` reads `config.bias_update_speed`, but `configs/base.py` defines `gamma` instead.
- `FeedForward` ignores `config.bias`. The `ln_f` in `gpt_moe.GPT` ignores `config.norm`.
- `sample.py` imports `data.dataset` at the top of the file, so it needs `data/input.txt` even for nanoGPT checkpoints.
- The `toy_train.py` docstring says `config/base.py`, but the directory is `configs/`. The README tree is missing `sample.py`,
  `configs/finetune_shakespeare.py` and `data/shakespeare/`.

## Common commands
```
python toy_train.py configs/base.py --attention='gqa' --ffn='moe_deepseek'
python data/shakespeare_char/prepare.py && python train.py --dataset=shakespeare_char --compile=False
python sample.py --out_dir=out
```