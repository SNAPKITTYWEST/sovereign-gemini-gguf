# Graph IR

```
Token IDs → Embedding Lookup (token_embd.weight) → 36× TransformerBlock → Final RMSNorm → Output Projection (W_embd^T) → Logits
```

Per block:
- `attn_norm` → GQA (QKV RoPE, MHA/MQA/GQA) → `attn_output` + residual
- `ffn_norm` → SwiGLU (gate/up → SiLU → down) + residual

Weight tying: `T000 token_embd.weight ↔ T290 output.weight` if offsets equal.

See `src/architecture/graph.py:1`.
