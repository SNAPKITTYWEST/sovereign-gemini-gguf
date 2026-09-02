# GGUF Format

GGUF (GGML Universal Format) is an aligned binary format for memory-mapped model loading.

## Header (24 bytes)
- Magic `GGUF` (4B, 0x47475546)
- Version `uint32` (2 or 3)
- Tensor Count `uint64`
- Metadata KV Count `uint64`

## Metadata KV Table
- Key: `uint64 len + UTF-8 bytes`
- Value Type `uint32` (0-12)
- Value: typed bytes

## Tensor Block
- Name `string`
- n_dims `uint32` (1-8)
- dims `uint64[n_dims]`
- dtype `uint32` (GGML type)
- relative_offset `uint64`
- Alignment padding to `general.alignment` (default 32)
- Binary tensor data

See `src/gguf/header.py:1`, `src/gguf/metadata.py:1`, `src/gguf/tensor.py:1`.
