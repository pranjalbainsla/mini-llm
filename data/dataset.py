"""
data/dataset.py

Char-level tokenizer + batcher over a single fixed file: data/input.txt.

To experiment with a different corpus, just replace the contents of
data/input.txt (re-run the wget below with a different URL, or upload your
own file with that exact name) — no config changes anywhere else. vocab_size
and the char<->int mapping are derived from whatever text is currently in
that file. That's deliberate: for architecture experiments (MoE/MLA
correctness), the data pipeline should be a non-variable, so nothing about
it lives in base.py.
"""
import os
import torch

DATA_PATH = 'data/input.txt'

# Download a dataset into DATA_PATH (or upload your own with this exact
# path) before running toy_train.py, e.g. in a Colab cell:
#   !mkdir -p data
#   !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O data/input.txt

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"{DATA_PATH} not found. Download or upload a text file there first, e.g.:\n"
        f"  !mkdir -p data && !wget <url> -O {DATA_PATH}"
    )

with open(DATA_PATH, 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]      # string -> list[int]
decode = lambda l: ''.join([itos[i] for i in l])  # list[int] -> string

# Train and val splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% train, rest val
train_data = data[:n]
val_data = data[n:]


def get_batch(split, batch_size, block_size, device):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y