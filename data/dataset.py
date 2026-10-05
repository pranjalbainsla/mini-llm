"""
data/dataset.py

Char-level tokenizer + batcher over a single fixed file: data/input.txt.

To experiment with a different corpus, just replace the contents of
data/input.txt (change DATA_URL below, or drop in your own file with that
exact name) — no config changes anywhere else. vocab_size
and the char<->int mapping are derived from whatever text is currently in
that file. That's deliberate: for architecture experiments (MoE/MLA
correctness), the data pipeline should be a non-variable, so nothing about
it lives in base.py.
"""
import os
import urllib.request

import torch

DATA_PATH = 'data/input.txt'

DATA_URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'

# Auto-download on first import if the file isn't there. To use your own corpus,
# put it at DATA_PATH beforehand and nothing is downloaded.
if not os.path.exists(DATA_PATH):
    print(f"{DATA_PATH} not found, downloading {DATA_URL}")
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)

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


def make_generator(seed):
    g = torch.Generator()  # CPU generator; indices are sampled on CPU then data moves to device
    g.manual_seed(seed)
    return g


def get_batch(split, batch_size, block_size, device, generator=None):
    # Pass a seeded `generator` for reproducible sampling. With generator=None this
    # falls back to the global RNG (old behaviour).
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
