# Dataset: no config entry — data/dataset.py always reads data/input.txt.
# Swap corpora by re-running `!wget ... -O data/input.txt` (or uploading a
# file to that path) in the Colab cell before training, not here.                               

# Training
batch_size = 16
block_size = 128                     
max_iters = 2000
learning_rate = 1e-3
eval_interval = 200                  
eval_iters = 200                      
log_interval = 20

init_seed = 1
train_seed = 1
eval_seed = 4242                

# Model
n_embd = 128
n_head = 4
n_layer = 4
n_kv_heads = 2                        

bias = False                        

# Architecture — flip these to test a different combination
attention = "mha"
ffn = "mlp"
norm = "layernorm"
position = "rope"

# MLA
latent_kv_dim = 32                    # MLA: compressed KV dimension
latent_q_dim = 32                     # MLA: compressed query dimension
rotary_ratio = 0.25                   # MLA: fraction of head dim using RoPE

# MoE 
num_experts = 4
num_shared_experts = 2                # DeepSeek-V3 uses 1; with k=2 this means 4 experts always active/token
k = 2                                 # routed experts selected per token
gamma = 0.001                         # aux-loss-free expert-bias update speed
alpha = 0.001                         # auxiliary load-balancing loss coefficient

# Regularization / numerical stability
rmsnorm_eps = 1e-5
dropout = 0.0
