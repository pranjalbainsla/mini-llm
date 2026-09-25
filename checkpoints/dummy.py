import torch

ckpt = torch.load("out/ckpt.pt", map_location="cpu", weights_only=False)
print(ckpt.keys())

print(ckpt["config"])
print(ckpt["iter_num"])