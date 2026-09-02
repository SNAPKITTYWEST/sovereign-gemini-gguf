from __future__ import annotations

import struct
from enum import IntEnum
from typing import Any, Dict, Tuple

from .mmap import GGUFSecurityError, GGUFError

MAX_STRING_LENGTH = 1_048_576
MAX_ARRAY_ELEMENTS = 10_000_000

class GGUFValueType(IntEnum):
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12

    @classmethod
    def to_str(cls, val_type: int) -> str:
        names = {
            0: "UINT8", 1: "INT8", 2: "UINT16", 3: "INT16", 4: "UINT32",
            5: "INT32", 6: "FLOAT32", 7: "BOOL", 8: "STRING", 9: "ARRAY",
            10: "UINT64", 11: "INT64", 12: "FLOAT64"
        }
        return names.get(val_type, f"UNKNOWN_{val_type}")

class GGUFMetadataError(ValueError):
    pass

def parse_string(mm, offset: int) -> Tuple[str, int]:
    if len(mm) < offset + 8:
        raise GGUFMetadataError("Truncated string length prefix.")
    length = struct.unpack_from("<Q", mm, offset)[0]
    if length > MAX_STRING_LENGTH:
        raise GGUFMetadataError(f"String length {length} exceeds maximum safety threshold.")
    start = offset + 8
    end = start + length
    if len(mm) < end:
        raise GGUFMetadataError("Truncated string data.")
    val = mm[start:end].decode("utf-8", errors="replace")
    return val, 8 + length

def parse_metadata_value(mm, offset: int) -> Tuple[Any, int]:
    if len(mm) < offset + 4:
        raise GGUFMetadataError("Truncated value type identifier.")
    vtype = struct.unpack_from("<I", mm, offset)[0]
    curr = offset + 4

    try:
        enum_type = GGUFValueType(vtype)
    except ValueError:
        raise GGUFMetadataError(f"Unknown GGUF metadata value type: {vtype}")

    if enum_type == GGUFValueType.UINT8:
        return struct.unpack_from("<B", mm, curr)[0], curr + 1 - offset
    elif enum_type == GGUFValueType.INT8:
        return struct.unpack_from("<b", mm, curr)[0], curr + 1 - offset
    elif enum_type == GGUFValueType.UINT16:
        return struct.unpack_from("<H", mm, curr)[0], curr + 2 - offset
    elif enum_type == GGUFValueType.INT16:
        return struct.unpack_from("<h", mm, curr)[0], curr + 2 - offset
    elif enum_type == GGUFValueType.UINT32:
        return struct.unpack_from("<I", mm, curr)[0], curr + 4 - offset
    elif enum_type == GGUFValueType.INT32:
        return struct.unpack_from("<i", mm, curr)[0], curr + 4 - offset
    elif enum_type == GGUFValueType.FLOAT32:
        return struct.unpack_from("<f", mm, curr)[0], curr + 4 - offset
    elif enum_type == GGUFValueType.BOOL:
        return bool(struct.unpack_from("<B", mm, curr)[0]), curr + 1 - offset
    elif enum_type == GGUFValueType.STRING:
        s, length = parse_string(mm, curr)
        return s, curr + length - offset
    elif enum_type == GGUFValueType.UINT64:
        return struct.unpack_from("<Q", mm, curr)[0], curr + 8 - offset
    elif enum_type == GGUFValueType.INT64:
        return struct.unpack_from("<q", mm, curr)[0], curr + 8 - offset
    elif enum_type == GGUFValueType.FLOAT64:
        return struct.unpack_from("<d", mm, curr)[0], curr + 8 - offset
    elif enum_type == GGUFValueType.ARRAY:
        if len(mm) < curr + 12:
            raise GGUFMetadataError("Truncated array header.")
        elem_type, arr_len = struct.unpack_from("<IQ", mm, curr)
        curr += 12
        if arr_len > MAX_ARRAY_ELEMENTS:
            raise GGUFMetadataError(f"Array length {arr_len} exceeds safety limit.")
        arr = []
        item_parser_type = GGUFValueType(elem_type)
        for _ in range(arr_len):
            val, consumed = parse_typed_array_item(mm, curr, item_parser_type)
            arr.append(val)
            curr += consumed
        return arr, curr - offset
    else:
        raise GGUFMetadataError(f"Unsupported value type: {enum_type}")

def parse_typed_array_item(mm, offset: int, item_type: GGUFValueType) -> Tuple[Any, int]:
    if item_type == GGUFValueType.UINT8:
        return struct.unpack_from("<B", mm, offset)[0], 1
    elif item_type == GGUFValueType.INT8:
        return struct.unpack_from("<b", mm, offset)[0], 1
    elif item_type == GGUFValueType.UINT16:
        return struct.unpack_from("<H", mm, offset)[0], 2
    elif item_type == GGUFValueType.INT16:
        return struct.unpack_from("<h", mm, offset)[0], 2
    elif item_type == GGUFValueType.UINT32:
        return struct.unpack_from("<I", mm, offset)[0], 4
    elif item_type == GGUFValueType.INT32:
        return struct.unpack_from("<i", mm, offset)[0], 4
    elif item_type == GGUFValueType.FLOAT32:
        return struct.unpack_from("<f", mm, offset)[0], 4
    elif item_type == GGUFValueType.BOOL:
        return bool(struct.unpack_from("<B", mm, offset)[0]), 1
    elif item_type == GGUFValueType.STRING:
        return parse_string(mm, offset)
    elif item_type == GGUFValueType.UINT64:
        return struct.unpack_from("<Q", mm, offset)[0], 8
    elif item_type == GGUFValueType.INT64:
        return struct.unpack_from("<q", mm, offset)[0], 8
    elif item_type == GGUFValueType.FLOAT64:
        return struct.unpack_from("<d", mm, offset)[0], 8
    else:
        raise GGUFMetadataError(f"Unsupported array element type: {item_type}")

def parse_metadata_kv(mm, offset: int, kv_count: int) -> Tuple[Dict[str, Any], int]:
    metadata = {}
    curr = offset
    for _ in range(kv_count):
        key, consumed_key = parse_string(mm, curr)
        curr += consumed_key
        val, consumed_val = parse_metadata_value(mm, curr)
        curr += consumed_val
        metadata[key] = val
    return metadata, curr - offset
