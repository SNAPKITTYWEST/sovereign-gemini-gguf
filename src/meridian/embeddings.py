"""Temporal embeddings: RoPE over patch index (no learned lookup)."""
from __future__ import annotations

import numpy as np

def rope_angles(seq: int, head_dim: int, theta: float = 10000.0) -> tuple[np.ndarray, np.ndarray]:
    half = head_dim // 2
    frac = 2 * np.arange(half) / head_dim
    timescale = theta ** frac
    pos = np.arange(seq)[:, None]
    ang = pos / timescale[None, :]
    return np.cos(ang), np.sin(ang)
