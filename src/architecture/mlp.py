from __future__ import annotations
from typing import Dict, Any, Optional

class MLPSpec:
    def __init__(self, layer_idx: int = 0, hidden_dim: int = 4096, intermediate_dim: int = 11008,
                 activation: str = "SwiGLU", gate_tensor: str = "", up_tensor: str = "",
                 down_tensor: str = "", router_tensor: Optional[str] = None):
        self.layer_idx = layer_idx
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.activation = activation
        self.gate_tensor = gate_tensor
        self.up_tensor = up_tensor
        self.down_tensor = down_tensor
        self.router_tensor = router_tensor
        self.is_moe = router_tensor is not None

    def representation(self) -> str:
        return f"x -> gate(x) -> {self.activation} -> up(x) -> elementwise_mul -> down -> output"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "activation": self.activation,
            "is_moe": self.is_moe,
            "router_tensor": self.router_tensor,
            "tensors": {"gate": self.gate_tensor, "up": self.up_tensor, "down": self.down_tensor}
        }
