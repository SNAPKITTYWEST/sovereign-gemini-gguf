from __future__ import annotations

import numpy as np
from .config import QUANTILES

def sort_quantiles(q: np.ndarray) -> np.ndarray:
    return np.sort(q, axis=-1)

def quantile_loss(y: np.ndarray, q_pred: np.ndarray, quantiles=QUANTILES) -> np.ndarray:
    y = y[..., None]
    qs = np.asarray(quantiles)
    err = y - q_pred
    return np.maximum(qs * err, (qs - 1) * err)
