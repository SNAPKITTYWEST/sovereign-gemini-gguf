from __future__ import annotations
from typing import List, Dict, Any, Optional
from .config import ModelConfig
from .attention import AttentionSpec
from .mlp import MLPSpec

class TransformerBlockNode:
    def __init__(self, layer_idx: int, attention: AttentionSpec, mlp: MLPSpec,
                 attn_norm_tensor: str, ffn_norm_tensor: str):
        self.layer_idx = layer_idx
        self.attention = attention
        self.mlp = mlp
        self.attn_norm_tensor = attn_norm_tensor
        self.ffn_norm_tensor = ffn_norm_tensor

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_index": self.layer_idx,
            "attention_norm": self.attn_norm_tensor,
            "attention": self.attention.to_dict(),
            "ffn_norm": self.ffn_norm_tensor,
            "mlp": self.mlp.to_dict()
        }

class ModelGraph:
    def __init__(self, config: ModelConfig, tensors: List[Dict[str, Any]]):
        self.config = config
        self.tensors = {t["name"]: t for t in tensors}
        self.embedding_tensor = ""
        self.output_tensor = ""
        self.final_norm_tensor = ""
        self.weight_tying = False
        self.blocks: List[TransformerBlockNode] = []
        self._reconstruct_graph()

    def _reconstruct_graph(self):
        for k in ["token_embd.weight", "model.embed_tokens.weight", "embed_tokens.weight"]:
            if k in self.tensors:
                self.embedding_tensor = k
                break
        for k in ["output.weight", "lm_head.weight"]:
            if k in self.tensors:
                self.output_tensor = k
                break
        if not self.output_tensor and self.embedding_tensor:
            self.output_tensor = self.embedding_tensor
            self.weight_tying = True
        elif self.output_tensor and self.embedding_tensor:
            emb_t = self.tensors[self.embedding_tensor]
            out_t = self.tensors[self.output_tensor]
            if emb_t["file_offset"] == out_t["file_offset"] and emb_t["byte_size"] == out_t["byte_size"]:
                self.weight_tying = True
        for k in ["output_norm.weight", "model.norm.weight", "norm.weight"]:
            if k in self.tensors:
                self.final_norm_tensor = k
                break
        for i in range(self.config.num_layers):
            attn = AttentionSpec(
                layer_idx=i,
                q_heads=self.config.num_attention_heads,
                kv_heads=self.config.num_kv_heads,
                head_dim=self.config.head_dim,
                hidden_dim=self.config.hidden_dim,
                q_tensor=self._find_layer_tensor(i, ["attn_q.weight", "self_attn.q_proj.weight"]),
                k_tensor=self._find_layer_tensor(i, ["attn_k.weight", "self_attn.k_proj.weight"]),
                v_tensor=self._find_layer_tensor(i, ["attn_v.weight", "self_attn.v_proj.weight"]),
                o_tensor=self._find_layer_tensor(i, ["attn_output.weight", "self_attn.o_proj.weight"])
            )
            router_t = self._find_layer_tensor_optional(i, ["ffn_gate_inp.weight", "block_sparse_moe.gate.weight"])
            mlp = MLPSpec(
                layer_idx=i,
                hidden_dim=self.config.hidden_dim,
                intermediate_dim=self.config.intermediate_dim,
                gate_tensor=self._find_layer_tensor(i, ["ffn_gate.weight", "mlp.gate_proj.weight"]),
                up_tensor=self._find_layer_tensor(i, ["ffn_up.weight", "mlp.up_proj.weight"]),
                down_tensor=self._find_layer_tensor(i, ["ffn_down.weight", "mlp.down_proj.weight"]),
                activation="SwiGLU" if "gemini" in self.config.architecture or "llama" in self.config.architecture else "GELU",
                router_tensor=router_t
            )
            attn_norm = self._find_layer_tensor(i, ["attn_norm.weight", "input_layernorm.weight"])
            ffn_norm = self._find_layer_tensor(i, ["ffn_norm.weight", "post_attention_layernorm.weight"])
            block = TransformerBlockNode(i, attn, mlp, attn_norm, ffn_norm)
            self.blocks.append(block)

    def _find_layer_tensor(self, layer_idx: int, candidates: List[str]) -> str:
        res = self._find_layer_tensor_optional(layer_idx, candidates)
        if res is not None:
            return res
        return f"blk.{layer_idx}.UNRESOLVED"

    def _find_layer_tensor_optional(self, layer_idx: int, candidates: List[str]) -> Optional[str]:
        prefixes = [f"blk.{layer_idx}.", f"layers.{layer_idx}.", f"model.layers.{layer_idx}."]
        for p in prefixes:
            for c in candidates:
                key = p + c
                if key in self.tensors:
                    return key
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "embedding_tensor": self.embedding_tensor,
            "output_tensor": self.output_tensor,
            "final_norm_tensor": self.final_norm_tensor,
            "weight_tying": self.weight_tying,
            "transformer_blocks": [b.to_dict() for b in self.blocks]
        }

    def export_ir(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "topology": [
                "TOKEN IDS",
                "TOKEN EMBEDDING",
                *[f"TRANSFORMER BLOCK {b.layer_idx}" for b in self.blocks],
                "FINAL NORMALIZATION",
                "OUTPUT PROJECTION",
                "LOGITS"
            ],
            "blocks": [b.to_dict() for b in self.blocks]
        }
