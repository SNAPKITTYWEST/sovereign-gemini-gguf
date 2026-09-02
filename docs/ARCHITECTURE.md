# Architecture

Gemini GGUF: 36 layers, 4096 hidden, 32 Q heads / 8 KV heads (GQA, factor 4), head_dim 128, 14336 intermediate (SwiGLU), vocab 256000, context 512, RoPE theta 10000, RMSNorm eps 1e-5, 291 tensors, Q4_K/Q6_K quant.

Detector: `general.architecture` → `gemini.*` keys, fallback to tensor shapes (`token_embd.weight`).

See `src/architecture/detector.py:1`, `src/architecture/config.py:1`.
