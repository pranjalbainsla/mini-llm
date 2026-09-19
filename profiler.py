import torch
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

activities = [ProfilerActivity.CPU]
if device.type == "cuda":
    activities.append(ProfilerActivity.CUDA)

batch = 4
seq_len = 128
hidden = 2048
intermediate = hidden * 4
dtype = torch.float16

# --------------------- Example workload ---------------------

x = torch.randn(batch, seq_len, hidden, dtype=dtype, device=device)
W_up = torch.randn(hidden, intermediate, dtype=dtype, device=device)
W_down = torch.randn( intermediate, hidden, dtype=dtype, device=device)


def ffn(x):
    with record_function("FFN.up_projection"):
        h = torch.matmul(x, W_up)

    with record_function("FFN.activation"):
        h = F.silu(h)

    with record_function("FFN.down_projection"):
        h = torch.matmul(h, W_down)

    with record_function("FFN.layer_norm"):
        h = F.layer_norm(h, [h.shape[-1]])

    return h

# Warmup

for _ in range(5):
    ffn(x)

if device.type == "cuda":
    torch.cuda.synchronize()

# ------------------------- Profile --------------------------------

with profile(
    activities=activities,

    # What tensors/shapes were involved?
    record_shapes=True,

    # Track memory allocations
    profile_memory=True,

    # Record Python call stacks
    with_stack=True,

    # Estimate FLOPs for supported operations
    with_flops=True,

) as prof:

    with record_function("FFN.forward"):
        for _ in range(10):
            y = ffn(x)

    if device.type == "cuda":
        torch.cuda.synchronize()


# ------------------------- Results --------------------------------

print("\n=== TOP OPERATIONS ===")

sort_key = (
    "self_cuda_time_total"
    if device.type == "cuda"
    else "self_cpu_time_total"
)

print(
    prof.key_averages().table(
        sort_by=sort_key,
        row_limit=20,
    )
)


print("\n=== MEMORY ===")

print(
    prof.key_averages().table(
        sort_by="self_cuda_memory_usage"
        if device.type == "cuda"
        else "self_cpu_memory_usage",
        row_limit=20,
    )
)

# -------------------- Export Chrome trace ------------------------

prof.export_chrome_trace("profile.json")

print("\nTrace written to profile.json")