from __future__ import annotations

import numpy as np
from .validation import ShapeError

def split_and_check(
    target: np.ndarray,
    horizon: int,
    past_only: np.ndarray | None,
    past_future: np.ndarray | None,
) -> None:
    if target.ndim != 2:
        raise ShapeError("target must be [variates, context]")
    _, ctx = target.shape
    if past_only is not None:
        if past_only.ndim != 2 or past_only.shape[1] != ctx:
            raise ShapeError("past_only must be [C, context]")
    if past_future is not None:
        if past_future.ndim != 2 or past_future.shape[1] != ctx + horizon:
            raise ShapeError("past_future must be [C, context+horizon]")
