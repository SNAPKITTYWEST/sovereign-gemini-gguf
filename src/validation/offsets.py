from __future__ import annotations
from typing import List, Dict, Any
from ..gguf.mmap import GGUFSecurityError

class OffsetValidator:
    @staticmethod
    def validate_no_overlap(tensors: List[Dict[str, Any]]):
        sorted_tensors = sorted(tensors, key=lambda x: x["file_offset"])
        for i in range(len(sorted_tensors) - 1):
            curr = sorted_tensors[i]
            nxt = sorted_tensors[i + 1]
            if curr["file_offset"] + curr["byte_size"] > nxt["file_offset"]:
                raise GGUFSecurityError(f"Overlapping tensor regions: '{curr['name']}' and '{nxt['name']}'")

    @staticmethod
    def validate_bounds(tensors: List[Dict[str, Any]], file_size: int):
        for t in tensors:
            if t["file_offset"] + t["byte_size"] > file_size:
                raise GGUFSecurityError(f"Tensor '{t['name']}' exceeds file bounds.")
            if t["relative_offset"] < 0:
                raise GGUFSecurityError(f"Tensor '{t['name']}' has negative offset.")
