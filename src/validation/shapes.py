from __future__ import annotations
from typing import List, Dict, Any
from ..gguf.mmap import GGUFValidationError

class ShapeValidator:
    @staticmethod
    def validate(config, tensors: List[Dict[str, Any]]):
        if config.num_attention_heads > 0 and config.hidden_dim > 0:
            if config.hidden_dim % config.num_attention_heads != 0:
                raise GGUFValidationError(f"Hidden dimension ({config.hidden_dim}) is not divisible by attention heads ({config.num_attention_heads})")
        if config.num_kv_heads > 0 and config.num_attention_heads > 0:
            if config.num_attention_heads % config.num_kv_heads != 0:
                raise GGUFValidationError(f"Query heads ({config.num_attention_heads}) is not divisible by Key/Value heads ({config.num_kv_heads})")
        for t in tensors:
            if not t["shape"] or any(d <= 0 for d in t["shape"]):
                raise GGUFValidationError(f"Tensor '{t['name']}' has invalid or non-positive dimensions: {t['shape']}")
        tensor_map = {t["name"]: t for t in tensors}
        for emb_key in ["token_embd.weight", "model.embed_tokens.weight"]:
            if emb_key in tensor_map:
                shape = tensor_map[emb_key]["shape"]
                if config.hidden_dim > 0 and config.hidden_dim not in shape:
                    raise GGUFValidationError(f"Embedding tensor '{emb_key}' shape {shape} mismatch with hidden dim {config.hidden_dim}")
