from __future__ import annotations
from typing import List, Dict, Any

class ParameterCounter:
    @staticmethod
    def count(tensors: List[Dict[str, Any]]) -> Dict[str, Any]:
        total_params = 0
        breakdown = {
            "embedding": 0,
            "attention": 0,
            "mlp": 0,
            "normalization": 0,
            "output": 0,
            "auxiliary": 0
        }
        for t in tensors:
            count = t["element_count"]
            total_params += count
            name = t["name"]
            if "embd" in name or "embed" in name:
                breakdown["embedding"] += count
            elif "attn" in name or "self_attn" in name:
                breakdown["attention"] += count
            elif "ffn" in name or "mlp" in name:
                breakdown["mlp"] += count
            elif "norm" in name:
                breakdown["normalization"] += count
            elif "output" in name or "lm_head" in name:
                breakdown["output"] += count
            else:
                breakdown["auxiliary"] += count
        breakdown["total_parameters"] = total_params
        breakdown["total_parameters_billions"] = round(total_params / 1e9, 3)
        return breakdown
