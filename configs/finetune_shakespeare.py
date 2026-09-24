import time

out_dir = 'out-shakespeare'
eval_interval = 5
eval_iters = 40
log_interval = 1

# only save when val loss improves, since we're finetuning briefly
always_save_checkpoint = False

wandb_log = False
wandb_project = 'shakespeare-finetune'
wandb_run_name = 'ft-' + str(time.time())

dataset = 'shakespeare'          # the BPE-level dataset from data/shakespeare/prepare.py
init_from = 'gpt2'               # 'gpt2' | 'gpt2-medium' | 'gpt2-large' | 'gpt2-xl'

# small batch, use grad accumulation to reach an effective batch size
# (the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters)
batch_size = 1
gradient_accumulation_steps = 32
block_size = 1024

# short run, small LR — this is the core of what makes it "finetuning"
max_iters = 40
learning_rate = 3e-5
decay_lr = False
warmup_iters = 0

weight_decay = 1e-1