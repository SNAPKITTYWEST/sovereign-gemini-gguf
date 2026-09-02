from __future__ import annotations
from typing import Dict, Any, List

class ModelDetector:
    @staticmethod
    def identify(metadata: Dict[str, Any], tensor_names: List[str]) -> Dict[str, Any]:
        arch = metadata.get("general.architecture", "unknown")
        if arch == "unknown":
            if any("gemini" in k.lower() for k in metadata.keys()) or any("gemini" in t.lower() for t in tensor_names):
                arch = "gemini"
            elif "llama" in metadata.get("general.name", "").lower():
                arch = "llama"
            else:
                arch = "unknown"
        return {
            "architecture": arch,
            "model_family": arch.upper(),
            "hidden_dimension": metadata.get(f"{arch}.embedding_length", metadata.get("embedding_length", 4096)),
            "embedding_dimension": metadata.get(f"{arch}.embedding_length", 4096),
            "number_layers": metadata.get(f"{arch}.block_count", metadata.get("block_count", 32)),
            "attention_heads": metadata.get(f"{arch}.attention.head_count", metadata.get("head_count", 32)),
            "key_value_heads": metadata.get(f"{arch}.attention.head_count_kv", metadata.get("head_count_kv", 8)),
            "intermediate_dimension": metadata.get(f"{arch}.feed_forward_length", metadata.get("feed_forward_length", 11008)),
            "context_length": metadata.get(f"{arch}.context_length", metadata.get("context_length", 2048)),
            "vocabulary_size": metadata.get(f"{arch}.vocab_size", len(metadata.get("tokenizer.ggm.tokens", []))),
            "quantization_format": metadata.get("general.quantization_version", "unquantized")
        }
