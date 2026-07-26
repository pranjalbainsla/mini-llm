import torch

# Dataset
dataset_path = "data/input.txt"

# training
batch_size = 16
block_size = 32
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
eval_iters = 200

# model
n_embd = 64
n_head = 4
n_layer = 4
num_experts = 4
dropout = 0.0

# Architecture
attention = "mha"      # mha, gqa, mla
ffn = "swiglu"            # mlp, moe, swiglu
norm = "rmsnorm"     # layernorm, rmsnorm
position = "rope"      # rope, learned, alibi

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(1337)