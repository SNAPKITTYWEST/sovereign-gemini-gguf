"""Continuous time series → Gemma hidden dimension."""
from __future__ import annotations

from .config import MeridianConfig

def adapter_input_dim(cfg: MeridianConfig) -> int:
    return cfg.input_feature_dim
