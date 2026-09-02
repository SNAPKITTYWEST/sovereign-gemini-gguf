from __future__ import annotations

import struct
from enum import IntEnum
from typing import Dict, List, Tuple

from .mmap import GGUFValidationError

class GGUFTensorType(IntEnum):
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23
    BF16 = 30

    @classmethod
    def to_str(cls, dtype: int) -> str:
        mapping = {
            0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
            8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
            14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
            19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
            30: "BF16"
        }
        return mapping.get(dtype, f"UNKNOWN_DTYPE_{dtype}")

    @classmethod
    def block_size_and_bytes(cls, dtype: int) -> Tuple[int, int]:
        specs = {
            cls.F32: (1, 4),
            cls.F16: (1, 2),
            cls.BF16: (1, 2),
            cls.Q4_0: (32, 18),
            cls.Q4_1: (32, 20),
            cls.Q5_0: (32, 22),
            cls.Q5_1: (32, 24),
            cls.Q8_0: (32, 34),
            cls.Q8_1: (32, 36),
            cls.Q2_K: (256, 84),
            cls.Q3_K: (256, 110),
            cls.Q4_K: (256, 144),
            cls.Q5_K: (256, 176),
            cls.Q6_K: (256, 210),
            cls.Q8_K: (256, 292),
            cls.IQ2_XXS: (256, 66),
            cls.IQ2_XS: (256, 74),
            cls.IQ3_XXS: (256, 98),
            cls.IQ1_S: (256, 50),
            cls.IQ4_NL: (32, 18),
            cls.IQ3_S: (256, 110),
            cls.IQ2_S: (256, 82),
            cls.IQ4_XS: (256, 136),
        }
        if dtype not in specs:
            raise GGUFValidationError(f"Unsupported GGUF tensor quantization type ID: {dtype}")
        return specs[dtype]

class GGMLType(IntEnum):
     F32 = 0
     F16 = 1
     Q4_0 = 2
     Q4_1 = 3
     Q5_0 = 6
     Q5_1 = 7
     Q8_0 = 8
     Q8_1 = 9
     Q2_K = 10
     Q3_K = 11
     Q4_K = 12
     Q5_K = 13
     Q6_K = 14
     Q8_K = 15
     I8 = 24
     I16 = 25
     I32 = 26
     F64 = 29

def get_type_info(dtype: int) -> Tuple[str, int, int]:
    try:
        t = GGMLType(dtype)
    except ValueError:
        return GGUFTensorType.to_str(dtype), 1, 4
    mapping = {
        GGMLType.F32: ("F32", 1, 4), GGMLType.F16: ("F16", 1, 2), GGMLType.F64: ("F64", 1, 8),
        GGMLType.I32: ("I32", 1, 4), GGMLType.I16: ("I16", 1, 2), GGMLType.I8: ("I8", 1, 1),
        GGMLType.Q4_0: ("Q4_0", 32, 18), GGMLType.Q4_1: ("Q4_1", 32, 20),
        GGMLType.Q5_0: ("Q5_0", 32, 22), GGMLType.Q5_1: ("Q5_1", 32, 24),
        GGMLType.Q8_0: ("Q8_0", 32, 34), GGMLType.Q8_1: ("Q8_1", 32, 36),
        GGMLType.Q2_K: ("Q2_K", 256, 84), GGMLType.Q3_K: ("Q3_K", 256, 110),
        GGMLType.Q4_K: ("Q4_K", 256, 144), GGMLType.Q5_K: ("Q5_K", 256, 176),
        GGMLType.Q6_K: ("Q6_K", 256, 210), GGMLType.Q8_K: ("Q8_K", 256, 292),
    }
    return mapping.get(t, (GGUFTensorType.to_str(dtype), 1, 4))

def calculate_tensor_size(shape: List[int], dtype: int) -> int:
    elements = 1
    for dim in shape:
        if dim < 0:
            raise ValueError(f"Negative dimension in shape: {shape}")
        elements *= dim
    name, block_size, type_size = get_type_info(dtype)
    if block_size == 1:
        return elements * type_size
    else:
        if elements % block_size != 0:
            raise ValueError(f"Element count {elements} not divisible by block size {block_size} for quantized dtype {name}")
        return (elements // block_size) * type_size

def parse_tensor_descriptors(mm, offset: int, tensor_count: int) -> Tuple[List[Dict], int]:
    tensors = []
    curr = offset
    for _ in range(tensor_count):
        length = struct.unpack_from("<Q", mm, curr)[0]
        start = curr + 8
        end = start + length
        name = mm[start:end].decode("utf-8", errors="replace")
        curr = end
        if len(mm) < curr + 4:
            raise ValueError("Truncated tensor dimension count.")
        n_dims = struct.unpack_from("<I", mm, curr)[0]
        curr += 4
        if n_dims > 8 or n_dims == 0:
            raise ValueError(f"Invalid tensor dimension count: {n_dims}. Must be between 1 and 8.")
        if len(mm) < curr + (n_dims * 8):
            raise ValueError("Truncated tensor dimensions array.")
        dimensions = []
        for _ in range(n_dims):
            dim = struct.unpack_from("<Q", mm, curr)[0]
            dimensions.append(dim)
            curr += 8
        if len(mm) < curr + 12:
            raise ValueError("Truncated tensor type/offset structure.")
        dtype, tensor_offset = struct.unpack_from("<IQ", mm, curr)
        curr += 12
        tensors.append({"name": name, "shape": dimensions, "dtype": dtype, "offset": tensor_offset})
    return tensors, curr - offset

def parse_string(mm, offset: int) -> Tuple[str, int]:
    length = struct.unpack_from("<Q", mm, offset)[0]
    start = offset + 8
    end = start + length
    return mm[start:end].decode("utf-8", errors="replace"), 8 + length
