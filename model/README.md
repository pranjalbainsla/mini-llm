# Notes

### Model

1) **Q: Why are the token embedding and `lm_head` untied in `gpt_moe`?**
A: They are kept as separate weights for now, which keeps the implementation simple and lets the output projection specialize independently of the input lookup. Untying adds a vocab × C matrix to the parameter count. The embedding is a lookup rather than a matmul, so it is normally excluded from "active" parameters anyway.
Next: try weight tying and compare loss and parameter count at a matched budget, since tying is the usual choice at small scale.

### Attention

1) **Q: Why does the current MLA implementation split each head's q/k dim `dh` into a non-rotary part and a rotary part, instead of adding RoPE dims on top like DeepSeek-V2?**
A: The split is simpler and keeps `dh` identical to standard MHA, which is fine for a toy model. The cost is that the rotary dims take capacity away from the content dims, so the model trades content for position.
In DeepSeek-V2's decoupled design, each head keeps its full content dim and gets extra `d_rope` dims on top. The RoPE key is a single key shared across heads, so the KV cache only grows by `d_rope` per token, and the content path stays compressible through the latent. This design is what makes MLA's cache savings work, because RoPE cannot be absorbed into the up-projection matrices.
Next: move to the decoupled formulation (content dim `dh` plus a shared `d_rope` key) and measure the KV cache size and quality against the current split version.

### FFN

1) **Q: Is SwiGLU better than a plain ReLU/GELU FFN?**
A: SwiGLU adds a learned gate. One projection goes through SiLU and is multiplied elementwise with a second "value" projection, so the FFN can modulate each hidden feature per token instead of applying a fixed nonlinearity. Empirically (Shazeer 2020, later adopted by PaLM, LLaMA and DeepSeek) it gives lower loss than ReLU/GELU MLPs at a matched parameter count.
To keep the budget equal, the hidden size shrinks to about 8/3·C (as in `swiglu.py`), so 3 matrices × 8/3·C² ≈ 8C², the same as the 4C ReLU MLP. The gain is quality per parameter, not fewer parameters, and the explanation is mostly empirical rather than theoretically settled.
Next: swiglu-style experts in moe_deepseek to align with the actual paper

2) **Q: Why are the MoE routers bias-free?**
A: In DeepSeek-style balancing, the `expert_bias` buffer is the only bias that should influence expert selection. It is updated by the gamma rule rather than by gradients, so a learned router bias would duplicate it and confound the balancing. LLaMA-style and OLMoE-style routers are bias-free as well.
Next: check that `bias` from the config is never applied to the router in any MoE variant, and monitor expert load balance during training to confirm the `expert_bias` update is doing its job.
