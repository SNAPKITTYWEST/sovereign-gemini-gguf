from __future__ import annotations

import numpy as np

def make_sine(n: int = 96, period: float = 10.0, noise: float = 0.05, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return np.sin(2 * np.pi * t / period) + noise * rng.normal(size=n)

def make_multivariate(n: int = 128, v: int = 3, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    rows = []
    for i in range(v):
        rows.append(10 + i * 2 + 3 * np.sin(2 * np.pi * t / (12 + i) + i) + 0.4 * rng.normal(size=n))
    return np.stack(rows, axis=0)
