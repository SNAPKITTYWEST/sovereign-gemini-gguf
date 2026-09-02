from __future__ import annotations

import numpy as np

from .config import SIGMA_EPS

def linear_interpolate(row: np.ndarray) -> np.ndarray:
    row = np.asarray(row, dtype=np.float64).copy()
    n = row.shape[0]
    nan = ~np.isfinite(row)
    if not nan.any():
        return row
    if nan.all():
        return np.zeros_like(row)
    idx = np.arange(n)
    row[nan] = np.interp(idx[nan], idx[~nan], row[~nan])
    return row

def revin(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray, reverse: bool = False) -> np.ndarray:
    s = np.where(sigma < SIGMA_EPS, 1.0, sigma)
    return x * s + mu if reverse else (x - mu) / s

def update_running_stats(n, mu, sigma, patch, mask):
    legit = ~mask
    inc_n = legit.sum()
    inc_mu = patch[legit].mean() if inc_n else 0.0
    inc_sigma = patch[legit].std() if inc_n else 0.0
    new_n = n + inc_n
    new_mu = 0.0 if new_n == 0 else (n * mu + inc_mu * inc_n) / new_n
    new_sigma = 0.0
    if new_n:
        new_sigma = np.sqrt(
            (
                n * sigma * sigma
                + inc_n * inc_sigma * inc_sigma
                + n * (mu - new_mu) ** 2
                + inc_n * (inc_mu - new_mu) ** 2
            )
            / new_n
        )
    return new_n, new_mu, new_sigma
