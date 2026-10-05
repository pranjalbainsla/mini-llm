"""
(Purely for understanding)

Refer to sample.py to get the generation script
"""
import torch

from model.gpt import GPT
from data.dataset import decode, vocab_size

model = GPT(vocab_size=vocab_size).to(device)
ckpt = torch.load("ckpt.pt", map_location=device)
model.load_state_dict(ckpt["model"])
model.eval()

if __name__ == "__main__":
    with torch.no_grad():
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        output = model.generate(context, max_new_tokens=2000, use_cache=True, use_weight_absorption=True)
        print(decode(output[0].tolist()))