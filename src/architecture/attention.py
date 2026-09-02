from __future__ import annotations
from typing import Dict, Any

class AttentionSpec:
    def __init__(self, layer_idx: int, q_heads: int, kv_heads: int, head_dim: int, hidden_dim: int,
                 q_tensor: str = "", k_tensor: str = "", v_tensor: str = "", o_tensor: str = ""):
        self.layer_idx = layer_idx
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.hidden_dim = hidden_dim
        self.q_dim = q_heads * head_dim
        self.kv_dim = kv_heads * head_dim
        self.k_dim = self.kv_dim
        self.v_dim = self.kv_dim
        self.output_dim = hidden_dim
        self.q_tensor = q_tensor
        self.k_tensor = k_tensor
        self.v_tensor = v_tensor
        self.o_tensor = o_tensor
        if q_heads == kv_heads:
            self.attention_type = "MHA"
        elif kv_heads == 1:
            self.attention_type = "MQA"
        else:
            self.attention_type = "GQA"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attention_type": self.attention_type,
            "q_heads": self.q_heads,
            "kv_heads": self.kv_heads,
            "head_dim": self.head_dim,
            "q_dim": self.q_dim,
            "kv_dim": self.kv_dim,
            "tensors": {"q": self.q_tensor, "k": self.k_tensor, "v": self.v_tensor, "output": self.o_tensor}
        }
