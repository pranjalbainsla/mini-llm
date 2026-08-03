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
num_shared_experts = 2 # DeepSeek V3 uses 1
k = 2
dropout = 0.0
alpha = 0.001
n_kv_heads = 2
latent_kv_dim = 32 # dc -> KV compression dimension
latent_q_dim = 32 # dc' -> Query compression dimension


# inference
max_seq_len = 4096

# Architecture
attention = "mla_deepseek"      # mha, gqa, mla
ffn = "moe_deepseek"            # mlp, moe, swiglu
norm = "rmsnorm"                # layernorm, rmsnorm
position = "rope"               # rope, learned, alibi

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(1337)