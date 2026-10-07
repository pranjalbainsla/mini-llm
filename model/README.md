# Notes

### Model

1) **Q: Why are the token embedding and `lm_head` untied in `gpt_moe`?**
A: They are kept as separate weights for now, which keeps the implementation simple and lets the output projection specialize independently of the input lookup. Untying adds a vocab × C matrix to the parameter count. The embedding is a lookup rather than a matmul, so it is normally excluded from "active" parameters anyway.
Next: try weight tying and compare loss and parameter count at a matched budget, since tying is the usual choice at small scale.

2) As of now, position = "rope" isn't wired to anything. Nothing in model/ or toy_train.py reads it. RoPE is hard-coded in the attention classes.
TODO: Wire it up and compare with learned pos embeddings etc.

3) Weight decay applies to everything. toy_train.py passes all parameters to AdamW with weight_decay=0.1, including norm weights, biases and embeddings. nanoGPT only decays the 2D weights.
TODO: read about it.

### Attention

1) **Q: Why does the current MLA implementation split each head's q/k dim `dh` into a non-rotary part and a rotary part, instead of adding RoPE dims on top like DeepSeek-V2?**
A: The split is simpler and keeps `dh` identical to standard MHA, which is fine for a toy model. The cost is that the rotary dims take capacity away from the content dims, so the model trades content for position.
In DeepSeek-V2's decoupled design, each head keeps its full content dim and gets extra `d_rope` dims on top. The RoPE key is a single key shared across heads, so the KV cache only grows by `d_rope` per token, and the content path stays compressible through the latent. This design is what makes MLA's cache savings work, because RoPE cannot be absorbed into the up-projection matrices.
Next: move to the decoupled formulation (content dim `dh` plus a shared `d_rope` key) and measure the KV cache size and quality against the current split version.

### FFN

<a id="swiglu-vs-mlp"></a>
1) **Q: Why is SwiGLU better than a plain ReLU/GELU FFN?**
A: Instead of just expanding and applying one nonlinearity, SwiGLU computes two projections of the input: one gets passed through a smooth activation called Swish, and the other acts as a "gate" that gets multiplied elementwise with the first, so the network learns to let some information through more than others, kind of like a volume knob on each feature, and that tends to work a bit better in practice than a plain ReLU MLP.
To keep the budget equal, the hidden size shrinks to about 8/3·C (as in `swiglu.py`), so 3 matrices × 8/3·C² ≈ 8C², the same as the 4C ReLU MLP. The gain is quality per parameter, not fewer parameters, and the explanation is mostly empirical rather than theoretically settled.

2) **Q: Why are the MoE routers bias-free?**
A: In DeepSeek-style balancing, the `expert_bias` buffer is the only bias that should influence expert selection. It is updated by the gamma rule rather than by gradients, so a learned router bias would duplicate it and confound the balancing. LLaMA-style and OLMoE-style routers are bias-free as well.
Next: check that `bias` from the config is never applied to the router in any MoE variant, and monitor expert load balance during training to confirm the `expert_bias` update is doing its job.
