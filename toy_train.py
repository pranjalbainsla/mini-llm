import torch
from types import SimpleNamespace

# Load defaults + command-line overrides
exec(open("configurator.py").read())

config = SimpleNamespace(**{
    k: v for k, v in globals().items()
    if not k.startswith("_")
})

device = "cuda" if torch.cuda.is_available() else "cpu"

from model.gpt_moe import GPT
from data.dataset import vocab_size, get_batch
# TODO: handle data loading as per the new structure of per-dataset preparation scripts

model = GPT(vocab_size=vocab_size, config=config).to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(split)
            _, loss, _ = model(X, Y)   # fixed: model returns 3 values, not 2
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
best_val_loss = float("inf")
start_step = 0

try:
    ckpt = torch.load("latest_checkpoint.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_step = ckpt["step"] + 1
    best_val_loss = ckpt["best_val_loss"]
    print(f"Resuming from step {start_step}")
except FileNotFoundError:
    print("No checkpoint found. Starting from scratch.")

step = start_step

try:
    for step in range(start_step, config.max_iters):

        if step % config.eval_interval == 0 or step == config.max_iters - 1:
            losses = estimate_loss()
            print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "best_val_loss": best_val_loss
                }, "best_checkpoint.pt")
                print(f"Saved new best model (val={best_val_loss:.4f})")

        xb, yb = get_batch('train')

        logits, loss, routing_info = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        model.update_expert_bias(routing_info)
finally:
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best_val_loss": best_val_loss
    }, "latest_checkpoint.pt")