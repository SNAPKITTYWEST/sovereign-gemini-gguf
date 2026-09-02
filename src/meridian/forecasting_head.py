"""Patch-level quantile head (hybrid C+D)."""
from __future__ import annotations

import numpy as np

def reshape_head(logits: np.ndarray, output_patch_len: int, num_quantiles: int) -> np.ndarray:
    *lead, last = logits.shape
    if last != output_patch_len * num_quantiles:
        raise ValueError(f"last dim {last} != {output_patch_len}*{num_quantiles}")
    return logits.reshape(*lead, output_patch_len, num_quantiles)
