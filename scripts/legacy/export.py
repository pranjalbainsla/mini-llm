import torch
from config import (
    block_size,
    device,
)
from data.dataset import vocab_size
from model.gpt import GPT

model = GPT(vocab_size=vocab_size).to(device)

dummy = torch.randint(0, vocab_size, (1, block_size))

torch.onnx.export(
    model,
    dummy,
    "gpt_moe.onnx",
    input_names=["tokens"],
    output_names=["logits"],
    opset_version=17,
)