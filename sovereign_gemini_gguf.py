#!/usr/bin/env python3
"""
sovereign_gemini_gguf.py — Single-file Gemini GGUF Parser & Graph IR
Standalone, zero-dependency, mmap, QRA-gated. See src/ for split modules.
Usage: python sovereign_gemini_gguf.py inspect model.gguf --json
"""
from src.cli.main import main as cli_main

if __name__ == "__main__":
    cli_main()
