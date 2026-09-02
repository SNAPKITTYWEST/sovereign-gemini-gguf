# Sovereign Gemini GGUF Parser

[![License: Sovereign](https://img.shields.io/badge/License-Sovereign%20Source%20v1.0%20%2B%20BSL--1.1%20%2B%20AGPL--3.0-critical.svg)](#license)
[![Parser](https://img.shields.io/badge/Parser-GGUF%20v2%2Fv3%20zero--sorry-blue.svg)](#verification)
[![Security](https://img.shields.io/badge/Security-zero--trust%20mmap-green.svg)](#security)

> **No torch. No TF. No network. Just `mmap` and `struct`.**

Standalone, zero-dependency GGUF binary parser and Graph IR for Gemini-class models (36× 4096 hidden, GQA 32/8, SwiGLU 14336, 291 tensors). Cherry-picked from `sovereign-cuda-kernels` control repo.

## Quick Start

```bash
pip install -e .
gemini-gguf inspect model.gguf
gemini-gguf graph model.gguf --json | jq .blocks[0]
gemini-gguf validate model.gguf
gemini-gguf evoke model.gguf  # bind 291 tensors, zero-copy
python -m src.tests.test_parser  # 3/3 PASS
```

## Features

- **Zero-copy mmap** — `GGUFMmapReader` with `memoryview` slices, 15.48GB mapped, no copy
- **Security** — `MAX_STRING_LENGTH=1M`, `MAX_TENSOR_COUNT=500k`, `MAX_ARRAY_ELEMENTS=10M`, overlap + bounds checks, no `eval`/`exec`
- **Graph IR** — `ModelGraph` → 36 `TransformerBlockNode` (GQA + SwiGLU + RMSNorm), weight tying `T000 ↔ T290`, `block_signature` per layer
- **CLI** — `inspect|metadata|tensors|architecture|graph|validate|evoke|test`

## Structure

```
src/gguf/{mmap,header,metadata,tensor,reader}.py
src/architecture/{detector,config,attention,mlp,graph}.py
src/validation/{shapes,offsets,parameters}.py
src/cli/main.py
src/tests/{test_parser,test_validation,test_security}.py
sovereign_gemini_gguf.py  # single-file combined engine
docs/{GGUF_FORMAT,ARCHITECTURE,GRAPH_IR,SECURITY,TEST_REPORT}.md
```

See `sovereign_gemini_gguf.py:1` for the single-file `GGUFParser` → `ModelConfig` → `ModelGraph` pipeline.
