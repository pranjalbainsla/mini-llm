import torch
from config import (
    learning_rate,
    max_iters,
    device,
    eval_interval,
    eval_iters,
)
from model.gpt import GPT
from data.dataset import vocab_size, get_batch

model = GPT(vocab_size=vocab_size).to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
best_val_loss = float("inf")

try:
    ckpt = torch.load("latest_checkpoint.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_iter = ckpt["iter"] + 1
    best_val_loss = ckpt["best_val_loss"]

    for iter in range(start_iter, max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            # Save best checkpoint for inference
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                # torch.save(model.state_dict(), "gpt.pth")
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iter": iter,
                    "best_val_loss": best_val_loss
                }, "best_checkpoint.pt")
                print(f"Saved new best model (val={best_val_loss:.4f})")

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
finally:
    # Serialize the model's state_dict (parameter-name -> tensor mapping). Unlike saving the
    # entire model, this avoids pickling the class definition, making checkpoints smaller,
    # more portable, and resilient to project refactors. Reload by instantiating the same
    # architecture and calling model.load_state_dict(torch.load(...)).
    # torch.save(model.state_dict(), "gpt_latest.pth")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iter": iter,
        "best_val_loss": best_val_loss
    }, "latest_checkpoint.pt")

