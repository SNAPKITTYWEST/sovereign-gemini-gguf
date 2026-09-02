from __future__ import annotations

from dataclasses import dataclass

from .config import G6Config, MeridianConfig, NanoConfig

@dataclass
class ParamLine:
    name: str
    count: int
    note: str

@dataclass
class ParamReport:
    model: str
    lines: list[ParamLine]
    total: int

def _attn(cfg: MeridianConfig) -> int:
    d = cfg.gemma.model_dims
    q = d * (cfg.gemma.n_heads * cfg.gemma.head_dim)
    k = d * (cfg.gemma.n_kv_heads * cfg.gemma.head_dim)
    v = d * (cfg.gemma.n_kv_heads * cfg.gemma.head_dim)
    o = cfg.gemma.n_heads * cfg.gemma.head_dim * d
    qk = 2 * cfg.gemma.head_dim if cfg.gemma.qk_norm else 0
    return q + k + v + o + qk

def parameter_report(cfg: MeridianConfig) -> ParamReport:
    seq = _attn(cfg)
    var = _attn(cfg) if cfg.use_variate_attention else 0
    mlp = 3 * cfg.gemma.model_dims * cfg.gemma.intermediate_size
    norms = (6 if cfg.use_variate_attention else 4) * cfg.gemma.model_dims
    per = seq + var + mlp + norms
    stack = per * cfg.n_layers
    in_f = cfg.input_feature_dim
    adapter = in_f * cfg.residual.hidden_dims + cfg.residual.hidden_dims * cfg.residual.output_dims + in_f * cfg.residual.output_dims
    out = cfg.output_patch_len * len(cfg.quantiles)
    head = cfg.gemma.model_dims * out + out
    lines = [
        ParamLine("Sequence GQA (per layer)", seq, "Gemma grouped-query"),
        ParamLine("Variate GQA (per layer)", var, "TimesFM-3 mixing"),
        ParamLine("GeGLU MLP (per layer)", mlp, cfg.gemma.ff_activation),
        ParamLine("RMSNorm (per layer)", norms, "pre/post"),
        ParamLine(f"Stacked layers × {cfg.n_layers}", stack, ""),
        ParamLine("Time-series residual adapter", adapter, f"in={in_f}"),
        ParamLine("Quantile forecast head", head, f"{out} outputs"),
    ]
    return ParamReport(cfg.name, lines, stack + adapter + head)

G6_PARAMS = parameter_report(G6Config)
NANO_PARAMS = parameter_report(NanoConfig)
