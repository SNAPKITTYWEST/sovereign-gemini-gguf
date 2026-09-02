from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

SIGMA_EPS = 1e-6
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MEDIAN_INDEX = 4

@dataclass
class GemmaConfig:
    model_dims: int = 4096
    n_heads: int = 32
    n_kv_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 11008
    ff_activation: str = "GeGLU"
    qk_norm: bool = True
    slidingWindow: int = 4096
    globalEvery: int = 6

@dataclass
class ResidualConfig:
    hidden_dims: int = 64
    output_dims: int = 4096

@dataclass
class MeridianConfig:
    name: str = "meridian-nano"
    size: str = "nano"
    input_feature_dim: int = 192
    output_patch_len: int = 64
    input_patch_len: int = 32
    quantiles: List[float] = field(default_factory=lambda: QUANTILES.copy())
    max_variates: int = 128
    max_context: int = 15360
    max_horizon: int = 512
    n_layers: int = 4
    use_variate_attention: bool = True
    gemma: GemmaConfig = field(default_factory=lambda: GemmaConfig(model_dims=64, n_heads=4, n_kv_heads=4, head_dim=16, intermediate_size=128))
    residual: ResidualConfig = field(default_factory=lambda: ResidualConfig(hidden_dims=64, output_dims=64))

NanoConfig = MeridianConfig(
    name="meridian-nano",
    size="nano",
    n_layers=4,
    gemma=GemmaConfig(model_dims=64, n_heads=4, n_kv_heads=4, head_dim=16, intermediate_size=128),
    residual=ResidualConfig(hidden_dims=64, output_dims=64),
)

G6Config = MeridianConfig(
    name="meridian-g6",
    size="g6",
    n_layers=28,
    gemma=GemmaConfig(model_dims=4096, n_heads=32, n_kv_heads=8, head_dim=128, intermediate_size=11008),
    residual=ResidualConfig(hidden_dims=2048, output_dims=4096),
)

@dataclass
class G6ConfigAlias:
    gemma: GemmaConfig = field(default_factory=lambda: GemmaConfig())
    n_layers: int = 28
    input_feature_dim: int = 192
    residual: ResidualConfig = field(default_factory=lambda: ResidualConfig(hidden_dims=2048, output_dims=4096))
    output_patch_len: int = 64
    quantiles: List[float] = field(default_factory=lambda: QUANTILES.copy())
    name: str = "meridian-g6"
    size: str = "g6"
