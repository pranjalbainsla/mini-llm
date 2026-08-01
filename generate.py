import torch
from config import device
from model.gpt import GPT
from data.dataset import decode, vocab_size

model = GPT(vocab_size=vocab_size).to(device)
model.load_state_dict(torch.load("best_checkpoint.pt", map_location=device))
model.eval()

if __name__ == "__main__":
    with torch.no_grad():
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        output = model.generate(context, max_new_tokens=2000)
        print(decode(output[0].tolist()))