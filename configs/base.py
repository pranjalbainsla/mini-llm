# Dataset
dataset_path = "data/input.txt"
 
# Training
batch_size = 16
max_iters = 5000
learning_rate = 1e-3
eval_interval = 100
eval_iters = 200

# Model
n_embd = 64
n_head = 4
n_layer = 4
n_kv_heads = 2

block_size = 32

# Architecture
attention = "mla_deepseek_optimized"    # mha, mha_optimized, gqa, mla_deepseek, mla_deepseek_optimized
ffn = "moe_deepseek"                    # mlp, swiglu, moe, moe_deepseek
norm = "rmsnorm"                        # layernorm, rmsnorm
position = "rope"                       # rope, learned, alibi

# Attention / position encoding
latent_kv_dim = 32                      # MLA: compressed KV dimension
latent_q_dim = 32                       # MLA: compressed query dimension
rotary_ratio = 0.25                     # MLA: fraction of head dim using RoPE

# MoE
num_experts = 4
num_shared_experts = 2                  # DeepSeek-V3 uses 1
k = 2                                   # experts selected per token
gamma = 0.001                           # expert bias update speed
alpha = 0.001                           # auxiliary loss coefficient

# Regularization / numerical stability
dropout = 0.0
rmsnorm_eps = 1e-5

# Inference
max_seq_len = 4096