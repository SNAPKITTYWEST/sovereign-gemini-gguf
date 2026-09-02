# Security

Zero-trust: GGUF is untrusted binary.

- `MAX_STRING_LENGTH=1M`, `MAX_TENSOR_COUNT=500k`, `MAX_ARRAY_ELEMENTS=10M`, `MAX_DIMENSION_COUNT=8`
- All `read()` check `offset+len ≤ file_size`, else `GGUFSecurityError`
- Tensor overlap: sorted by `file_offset`, check `curr.offset+size ≤ next.offset`
- Bounds: `abs_offset+byte_size ≤ file_size`, `relative_offset ≥0`
- No `eval`/`exec`/`unpickle`, strings decoded as `utf-8` only
- `read_slice` returns `memoryview`, zero-copy, bounds-checked

See `src/gguf/mmap.py:1`, `src/validation/offsets.py:1`.
