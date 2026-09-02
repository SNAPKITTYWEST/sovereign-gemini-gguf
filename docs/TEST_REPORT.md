# Test Report

```
python -m src.tests.test_parser      # 3/3 PASS (header, tensor slice, invalid magic)
python -m src.tests.test_validation  # 2/2 PASS (valid shape, invalid hidden_dim)
python -m src.tests.test_security    # 3/3 PASS (no overlap, overlap detect, bounds)
gemini-gguf test                     # invokes TestGGUFParser on mock_gemini.gguf
```

Mock fixture: GGUF v3, 2 tensors (`token_embd.weight` 64×128 F32, `blk.0.attn_q.weight` 64×64 F32), 4 KV (`general.architecture=gemini`, `embedding_length=64`, `block_count=1`, `head_count=4`), 32-byte aligned, zero-copy `memoryview` verified.

Coverage: header, metadata, tensor descriptors, overlap, bounds, shape divisibility, mmap slice.
