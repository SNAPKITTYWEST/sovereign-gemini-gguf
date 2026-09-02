from __future__ import annotations
from typing import Dict, Any, List

class ModelConfig:
    def __init__(self, metadata: Dict[str, Any], tensors: List[Dict[str, Any]]):
        self.metadata = metadata
        self.tensors = tensors
        self.architecture = str(metadata.get("general.architecture", "unknown")).lower()
        self.model_family = str(metadata.get(f"{self.architecture}.family", metadata.get("general.name", self.architecture)))
        self.hidden_dim = int(metadata.get(f"{self.architecture}.embedding_length", 0))
        self.hidden_size = self.hidden_dim
        self.intermediate_dim = int(metadata.get(f"{self.architecture}.feed_forward_length", 0))
        self.num_layers = int(metadata.get(f"{self.architecture}.block_count", 0))
        self.num_attention_heads = int(metadata.get(f"{self.architecture}.attention.head_count", 0))
        self.num_kv_heads = int(metadata.get(f"{self.architecture}.attention.head_count_kv", self.num_attention_heads))
        self.vocab_size = int(metadata.get(f"{self.architecture}.vocab_size", 0))
        self.context_length = int(metadata.get(f"{self.architecture}.context_length", 0))
        self.rope_theta = float(metadata.get(f"{self.architecture}.rope.freq_base", 10000.0))
        self.rms_norm_eps = float(metadata.get(f"{self.architecture}.attention.layer_norm_rms_epsilon", 1e-5))
        self.num_experts = int(metadata.get(f"{self.architecture}.expert_count", 0))
        self.num_experts_used = int(metadata.get(f"{self.architecture}.expert_used_count", 0))
        self._infer_missing_from_tensors()
        if self.num_attention_heads > 0 and self.hidden_dim > 0:
            self.head_dim = self.hidden_dim // self.num_attention_heads
        else:
            self.head_dim = 0
        if self.head_dim == 0:
            self.head_dim = 128

    def _infer_missing_from_tensors(self):
        tensor_map = {t["name"]: t for t in self.tensors}
        for embed_key in ["token_embd.weight", "model.embed_tokens.weight", "embed_tokens.weight"]:
            if embed_key in tensor_map:
                shape = tensor_map[embed_key]["shape"]
                if self.vocab_size == 0:
                    self.vocab_size = shape[1] if len(shape) > 1 else shape[0]
                if self.hidden_dim == 0:
                    self.hidden_dim = shape[0] if len(shape) > 1 else shape[1]
                self.hidden_size = self.hidden_dim
        if self.num_layers == 0:
            max_layer = -1
            for name in tensor_map:
                if "blk." in name or "layers." in name:
                    parts = name.split(".")
                    for p in parts:
                        if p.isdigit():
                            max_layer = max(max_layer, int(p))
            if max_layer >= 0:
                self.num_layers = max_layer + 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "architecture": self.architecture,
            "hidden_size": self.hidden_size,
            "hidden_dim": self.hidden_dim,
            "intermediate_size": self.intermediate_dim,
            "num_layers": self.num_layers,
            "attention_heads": self.num_attention_heads,
            "kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "rope_theta": self.rope_theta,
        }
