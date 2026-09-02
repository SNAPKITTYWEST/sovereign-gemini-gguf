"""Numpy reference forecast for the untrained Nano-class contract."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from .config import MEDIAN_INDEX, MeridianConfig, NanoConfig, QUANTILES
from .normalization import linear_interpolate
from .validation import validate_forecast_request

@dataclass
class ForecastOutput:
    forecast: np.ndarray
    quantiles: np.ndarray | None
    context: np.ndarray

def _to_2d(target) -> tuple[np.ndarray, bool]:
    arr = np.asarray(target, dtype=np.float64)
    if arr.ndim == 1:
        return arr[None, :], True
    if arr.ndim == 2:
        return arr, False
    raise ValueError(f"target rank {arr.ndim} invalid")

def _norm_inv(p: float) -> float:
    a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.38357751867269e2, -3.066479806614716e1, 2.506628277459239]
    b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1]
    c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783]
    d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416]
    plow, phigh = 0.02425, 1 - 0.02425
    if p <= 0:
        return -8.0
    if p >= 1:
        return 8.0
    if p < plow:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1))
    if p > phigh:
        q = np.sqrt(-2 * np.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1))
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)

def forecast(
    target,
    horizon: int,
    *,
    past_only_covariates=None,
    past_future_covariates=None,
    return_quantiles: bool = True,
    config: MeridianConfig = NanoConfig,
) -> ForecastOutput:
    if config.size == "g6":
        raise RuntimeError("Meridian-G6 weights are not shipped.")
    y, was_1d = _to_2d(target)
    y = np.vstack([linear_interpolate(row) for row in y])
    validate_forecast_request(y, horizon, past_only_covariates, past_future_covariates, config)
    v, c = y.shape
    mu = y.mean(axis=1, keepdims=True)
    sigma = y.std(axis=1, keepdims=True)
    sigma = np.where(sigma < 1e-6, 1.0, sigma)
    z = np.array([0.5 * _norm_inv(q) for q in config.quantiles])
    q = mu[:, None, :] + sigma[:, None, :] * z[None, None, :]
    q = np.repeat(q, horizon, axis=1)
    if c >= 2:
        slope = (y[:, -1] - y[:, -2])[:, None]
        t = np.arange(1, horizon + 1)[None, :]
        q = q + slope[:, :, None] * 0.15 * t[:, :, None]
    point = q[:, :, MEDIAN_INDEX]
    if was_1d:
        point = point[0]
        q_out = q[0] if return_quantiles else None
        ctx = y[0]
    else:
        q_out = q if return_quantiles else None
        ctx = y
    return ForecastOutput(forecast=point, quantiles=q_out, context=ctx)
