"""Training interface (intentionally empty of Google recipes)."""
from __future__ import annotations

from collections.abc import Iterator
import numpy as np
from .data import make_sine

def synthetic_batches(batch: int = 8, context: int = 128, horizon: int = 16) -> Iterator[dict]:
    step = 0
    while True:
        xs = np.stack([make_sine(context + horizon, seed=step * batch + i) for i in range(batch)])
        yield {"context": xs[:, :context], "target": xs[:, context:], "step": step}
        step += 1
