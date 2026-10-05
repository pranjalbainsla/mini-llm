"""
Scratch sanity checks for the custom GPT stack. NOT part of the trainer.

Run from the repo root, same args as toy_train.py:
    python sanity.py configs/base.py --attention='gqa' --ffn='moe_deepseek'

Checks:
  1. shapes + initial loss ~ ln(vocab_size)
  2. every parameter receives a non-zero gradient
  3. causality (changing a future token must not change earlier logits)
  4. overfit ONE fixed batch of 32 examples -> loss should go to ~0
"""
import math
from types import SimpleNamespace

import torch

from data.dataset import get_batch, make_generator, vocab_size
from model.gpt_moe import GPT

# -----------------------------------------------------------------------------
# scratch-only knobs (hard-coded on purpose, edit freely)
N_EXAMPLES = 32      # size of the single fixed batch
N_STEPS = 300
LR = 1e-3
PRINT_EVERY = 20
PASS_LOSS = 0.1      # final loss below this counts as "memorized"
# -----------------------------------------------------------------------------

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# build config exactly the way toy_train.py does
exec(open('configurator.py').read())
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys}
config['dropout'] = 0.0   # no dropout while sanity checking

torch.manual_seed(config['init_seed'])
model_config = SimpleNamespace(**config)
model = GPT(vocab_size, model_config).to(device)
print(f"device={device}  vocab_size={vocab_size}  params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")

B, T = N_EXAMPLES, config['block_size']
gen = make_generator(config['train_seed'])
X, Y = get_batch('train', B, T, device, gen)   # fetched ONCE, reused everywhere below


def fwd(x, y):
    logits, loss, routing_info = model(x, y)
    return logits, loss, routing_info


# -----------------------------------------------------------------------------
# 1) shapes + initial loss
model.train()
logits, loss, routing_info = fwd(X, Y)
expected = math.log(vocab_size)
print(f"\n[1] logits {tuple(logits.shape)} (want ({B}, {T}, {vocab_size}))")
print(f"    initial loss {loss.item():.4f}  vs ln(vocab)={expected:.4f}  "
      f"-> {'OK' if abs(loss.item() - expected) < 0.5 else 'SUSPICIOUS (init scale / last layer too big?)'}")

# -----------------------------------------------------------------------------
# 2) gradient flow
model.zero_grad(set_to_none=True)
loss.backward()
no_grad, zero_grad = [], []
for name, p in model.named_parameters():
    if p.grad is None:
        no_grad.append(name)
    elif p.grad.abs().max().item() == 0.0:
        zero_grad.append(name)
print(f"\n[2] params with grad=None: {len(no_grad)}   all-zero grad: {len(zero_grad)}")
for n in no_grad:
    print("    NO GRAD  ", n)
for n in zero_grad:
    print("    ZERO GRAD", n, "(can be legit for an expert nobody was routed to)")
model.zero_grad(set_to_none=True)

# -----------------------------------------------------------------------------
# 3) causality: perturb the last token, earlier positions must not change
model.eval()
with torch.no_grad():
    x1 = X[:1].clone()
    x2 = x1.clone()
    x2[0, -1] = (x2[0, -1] + 1) % vocab_size
    l1, _, _ = fwd(x1, Y[:1])
    l2, _, _ = fwd(x2, Y[:1])
    diff_early = (l1[:, :-1] - l2[:, :-1]).abs().max().item()
    diff_last = (l1[:, -1] - l2[:, -1]).abs().max().item()
print(f"\n[3] max diff at earlier positions: {diff_early:.2e} (want ~0)   at last position: {diff_last:.2e} (want > 0)")
print(f"    -> {'OK' if diff_early < 1e-4 else 'LEAK: model sees future tokens (mask bug?)'}")

# -----------------------------------------------------------------------------
# 4) overfit one fixed batch
model.train()
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
print(f"\n[4] overfitting one batch of {B} x {T} tokens for {N_STEPS} steps (lr={LR})")
for i in range(N_STEPS):
    logits, loss, routing_info = fwd(X, Y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    model.update_expert_bias(routing_info)   # same call the trainer makes
    if i % PRINT_EVERY == 0 or i == N_STEPS - 1:
        print(f"    step {i:4d}  loss {loss.item():.4f}")

final = loss.item()
if final < PASS_LOSS:
    print(f"\nPASS: loss {final:.4f} < {PASS_LOSS}")
else:
    print(f"\nFAIL: loss {final:.4f} did not reach ~0. Suspect labels/shift, missing grads (see [2]), "
          f"causal mask, or a frozen/detached path. Not a tuning problem.")