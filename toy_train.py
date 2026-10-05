"""
A minimal training loop for gpt_moe.py

Usage:
    python toy_train.py config/base.py
    python toy_train.py config/base.py --max_iters=2000 --n_layer=6   # CLI overrides

Design intent: GPT takes a single `config` object and pulls whatever
attributes it needs off it (config.n_embd, config.alpha, ...). We build that
object directly from everything in the config file, so adding a new
architectural knob to gpt_moe.py never requires touching this script — just
add the field to your config file and read it from `config` wherever you
need it in the model.
"""
import os
import json
import time
from contextlib import nullcontext
from types import SimpleNamespace

import torch

from data.dataset import get_batch, make_generator, vocab_size, chars
from model.gpt_moe import GPT

# -----------------------------------------------------------------------------

# defaults — anything NOT set here MUST be set in the config file, or Python
# raises a NameError the moment we reach the line that uses it.

# No `dataset` key here on purpose: data/dataset.py always reads a single
# fixed path (data/input.txt). Swapping corpora is a Colab-cell concern
# (re-wget/upload that file) — it doesn't need a config entry, and vocab_size
# is derived straight from whatever's in that file, never hand-set.

eval_only = False                # if True, run a single eval pass and exit (sanity check)
run_name = ''                    # if set, write results/<run_name>.json (curves, best val, params, sec/iter) at the end
results_dir = 'results'

# optimizer knobs (nanoGPT-standard AdamW settings; override in config to sweep)
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0                  # clip grad norm; set to 0.0 to disable

# -----------------------------------------------------------------------------
# system
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'

# -----------------------------------------------------------------------------
# override with cli args / config file
# config_keys is captured AFTER configurator.py runs, so it picks up every
# variable the config file defines (batch_size, n_layer, num_experts, ...),
# not just the defaults declared above. This dict is used both for the
# results JSON's logged config AND to build the model's config object below.
exec(open('configurator.py').read())
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------
tokens_per_iter = batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")
# -----------------------------------------------------------------------------
torch.manual_seed(init_seed)
train_gen = make_generator(train_seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# float16 has a much smaller dynamic range than bfloat16, so gradients can
# quietly underflow to zero. GradScaler rescales the loss to keep small
# gradients representable, then unscales before the optimizer step. It's a
# no-op whenever dtype isn't float16 (e.g. bf16 on an A100), so this line is
# safe to leave in unconditionally.
scaler = torch.amp.GradScaler(device_type, enabled=(dtype == 'float16' and device_type == 'cuda'))

# -----------------------------------------------------------------------------
iter_num = 0
# -----------------------------------------------------------------------------
# model init (always from scratch; this script is for experiments, not checkpointing)
print("Initializing a new model from scratch")
# GPT reads whatever attributes it needs off this object (config.n_embd,
# config.alpha, ...) — passing the full dict through means no per-field
# wiring to maintain here as gpt_moe.py grows. Harmless: GPT ignores fields
# it doesn't look up (e.g. learning_rate, run_name).
model_config = SimpleNamespace(**config)
model = GPT(vocab_size, model_config)

model.to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"params: {n_params:,}")
# -----------------------------------------------------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                               betas=(beta1, beta2), weight_decay=weight_decay)
# -----------------------------------------------------------------------------
history = []      # one entry per eval: {iter, train, val}
eval_time = 0.0   # wall time spent inside estimate_loss, excluded from sec/iter
# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    global eval_time
    t_start = time.time()
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        eval_gen = make_generator(eval_seed) 
        for i in range(eval_iters):
            X, Y = get_batch(split, batch_size, block_size, device, eval_gen)
            with ctx:
                logits, loss, _ = model(X, Y)  # third return is per-layer MoE routing info
            losses[i] = loss.item()
        out[split] = losses.mean()
    model.train()
    eval_time += time.time() - t_start
    return out
# -----------------------------------------------------------------------------
# training loop
X, Y = get_batch('train', batch_size, block_size, device, train_gen)  # fetch the very first batch
t0 = time.time()
t_run_start = t0
local_iter_num = 0
raw_model = model   # separate name so this still works if you later wrap
                    # `model` in DDP or torch.compile without touching the loop

while True:
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        history.append({'iter': iter_num, 'train': losses['train'].item(), 'val': losses['val'].item()})

    if iter_num == 0 and eval_only:
        break

    # Forward pass must run inside `ctx`, or autocast only applies during
    # eval and training silently runs in full fp32 regardless of `dtype`.
    with ctx:
        logits, loss, routing_info = model(X, Y)
    # prefetch next batch on CPU while the forward pass above is still queued
    # asynchronously on the GPU — cheap overlap, standard nanoGPT trick.
    X, Y = get_batch('train', batch_size, block_size, device, train_gen)

    # scaler.scale(loss) is a harmless multiply-by-1 when the scaler is
    # disabled (bf16/fp32 runs), so this path is correct for every dtype
    # without an if/else branch.
    scaler.scale(loss).backward()
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # DeepSeek-V3-style aux-loss-free load balancing: nudges each expert's
    # routing bias based on how over/under-used it was this step (speed set
    # by config.gamma). update_expert_bias() checks hasattr(block.ffwd, ...)
    # internally, so it's a safe no-op on any block without an MoE FFN.
    model.update_expert_bias(routing_info)

    if iter_num % log_interval == 0:
        dt = time.time() - t0
        t0 = time.time()
        print(f"iter {iter_num}: loss {loss.item():.4f}, {dt * 1000 / max(log_interval, 1):.1f}ms/iter")

    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break
# -----------------------------------------------------------------------------
# results logging
if run_name and not eval_only:
    if device_type == 'cuda':
        torch.cuda.synchronize()
    best = min(history, key=lambda h: h['val'])
    result = {
        'run_name': run_name,
        'init_seed': init_seed, 'train_seed': train_seed, 'eval_seed': eval_seed,
        'best_val': best['val'], 'best_iter': best['iter'],
        'train_at_best': best['train'], 'gap_at_best': best['val'] - best['train'],
        'final_val': history[-1]['val'], 'final_train': history[-1]['train'],
        'params': n_params,
        'sec_per_iter': (time.time() - t_run_start - eval_time) / iter_num,
        'history': history,
        'config': config,
    }
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, f"{run_name}.json"), 'w') as f:
        json.dump(result, f, indent=1)
    print(f"wrote {results_dir}/{run_name}.json")
