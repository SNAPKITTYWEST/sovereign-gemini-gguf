from __future__ import annotations

import os
import mmap
import math
from typing import Any, Dict, List, Optional

from .header import parse_header
from .metadata import parse_metadata_kv
from .tensor import parse_tensor_descriptors, calculate_tensor_size
from .mmap import GGUFMmapReader, GGUFSecurityError

MAX_TENSOR_COUNT = 500_000
MAX_METADATA_COUNT = 1_000_000
MAX_DIMENSION_COUNT = 8

class GGUFParser:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.reader = GGUFMmapReader(filepath)
        self.magic = b""
        self.version = 0
        self.tensor_count = 0
        self.metadata_kv_count = 0
        self.metadata: Dict[str, Any] = {}
        self.tensors: List[Dict[str, Any]] = []
        self.tensor_index: Dict[str, Dict[str, Any]] = {}
        self.alignment = 32
        self.data_offset = 0

    def parse(self):
        self._parse_header()
        self._parse_metadata()
        self._calculate_tensor_offsets()
        self._validate_tensor_regions()

    def _parse_header(self):
        self.magic = self.reader.read(4)
        if self.magic != b"GGUF":
            raise GGUFSecurityError(f"Invalid GGUF magic header: {self.magic.hex()}. Expected 'GGUF'.")
        self.version = self.reader.read_value("<I")
        if self.version not in {2, 3}:
            raise GGUFSecurityError(f"Unsupported GGUF spec version: {self.version}.")
        self.tensor_count = self.reader.read_value("<Q")
        if self.tensor_count > MAX_TENSOR_COUNT:
            raise GGUFSecurityError(f"Tensor count {self.tensor_count} exceeds safety limit.")
        self.metadata_kv_count = self.reader.read_value("<Q")
        if self.metadata_kv_count > MAX_METADATA_COUNT:
            raise GGUFSecurityError(f"Metadata count {self.metadata_kv_count} exceeds safety limit.")

    def _parse_metadata(self):
        for _ in range(self.metadata_kv_count):
            key = self.reader.read_string()
            val_type = self.reader.read_value("<I")
            value = self._parse_metadata_value(val_type)
            self.metadata[key] = value
            if key == "general.alignment":
                self.alignment = int(value)

    def _parse_metadata_value(self, val_type: int) -> Any:
        from .metadata import GGUFValueType
        if val_type == GGUFValueType.UINT8: return self.reader.read_value("<B")
        elif val_type == GGUFValueType.INT8: return self.reader.read_value("<b")
        elif val_type == GGUFValueType.UINT16: return self.reader.read_value("<H")
        elif val_type == GGUFValueType.INT16: return self.reader.read_value("<h")
        elif val_type == GGUFValueType.UINT32: return self.reader.read_value("<I")
        elif val_type == GGUFValueType.INT32: return self.reader.read_value("<i")
        elif val_type == GGUFValueType.FLOAT32: return self.reader.read_value("<f")
        elif val_type == GGUFValueType.BOOL: return bool(self.reader.read_value("<B"))
        elif val_type == GGUFValueType.STRING: return self.reader.read_string()
        elif val_type == GGUFValueType.UINT64: return self.reader.read_value("<Q")
        elif val_type == GGUFValueType.INT64: return self.reader.read_value("<q")
        elif val_type == GGUFValueType.FLOAT64: return self.reader.read_value("<d")
        elif val_type == GGUFValueType.ARRAY:
            elem_type = self.reader.read_value("<I")
            elem_count = self.reader.read_value("<Q")
            if elem_count > 10_000_000:
                raise GGUFSecurityError(f"Array count {elem_count} exceeds safety limit.")
            return [self._parse_metadata_value(elem_type) for _ in range(elem_count)]
        else:
            raise GGUFSecurityError(f"Unknown GGUF metadata value type: {val_type}")

    def _calculate_tensor_offsets(self):
        for _ in range(self.tensor_count):
            name = self.reader.read_string()
            n_dims = self.reader.read_value("<I")
            if n_dims > MAX_DIMENSION_COUNT or n_dims == 0:
                raise GGUFSecurityError(f"Invalid tensor dimension count {n_dims} for {name}")
            dims = [self.reader.read_value("<Q") for _ in range(n_dims)]
            tensor_type = self.reader.read_value("<I")
            relative_offset = self.reader.read_value("<Q")
            from .tensor import GGUFTensorType
            element_count = 1
            for d in dims:
                element_count *= d
            block_size, bytes_per_block = GGUFTensorType.block_size_and_bytes(tensor_type)
            if element_count % block_size != 0:
                raise GGUFSecurityError(f"Tensor {name} elements {element_count} not aligned with block size {block_size}")
            byte_size = (element_count // block_size) * bytes_per_block
            info = {
                "name": name, "shape": dims, "dtype": tensor_type,
                "dtype_str": GGUFTensorType.to_str(tensor_type),
                "element_count": element_count, "byte_size": byte_size,
                "block_size": block_size, "bytes_per_block": bytes_per_block,
                "relative_offset": relative_offset
            }
            self.tensors.append(info)
        current_offset = self.reader.tell()
        padding = (self.alignment - (current_offset % self.alignment)) % self.alignment
        self.data_offset = current_offset + padding
        for t in self.tensors:
            abs_offset = self.data_offset + t["relative_offset"]
            t["file_offset"] = abs_offset
            if abs_offset + t["byte_size"] > self.reader.file_size:
                raise GGUFSecurityError(f"Tensor {t['name']} exceeds file bounds.")
            self.tensor_index[t["name"]] = t

    def _validate_tensor_regions(self):
        sorted_tensors = sorted(self.tensors, key=lambda x: x["file_offset"])
        for i in range(len(sorted_tensors) - 1):
            curr = sorted_tensors[i]
            nxt = sorted_tensors[i + 1]
            if curr["file_offset"] + curr["byte_size"] > nxt["file_offset"]:
                raise GGUFSecurityError(f"Overlapping tensors: {curr['name']} and {nxt['name']}")

    def read_tensor_slice(self, name: str, offset_elements: int = 0, count_elements: Optional[int] = None) -> memoryview:
        if name not in self.tensor_index:
            raise KeyError(f"Tensor '{name}' not found.")
        t = self.tensor_index[name]
        if count_elements is None:
            count_elements = t["element_count"] - offset_elements
        if offset_elements + count_elements > t["element_count"]:
            raise GGUFSecurityError(f"Slice exceeds tensor {name} bounds.")
        block_size = t["block_size"]
        bytes_per_block = t["bytes_per_block"]
        start_block = offset_elements // block_size
        num_blocks = math.ceil(count_elements / block_size)
        slice_byte_offset = t["file_offset"] + (start_block * bytes_per_block)
        slice_byte_len = num_blocks * bytes_per_block
        return self.reader.read_slice(slice_byte_offset, slice_byte_len)

    def close(self):
        self.reader.close()

class GGUFReader:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file_size = os.path.getsize(filepath)
        self.fd = open(filepath, "rb")
        self.mm = mmap.mmap(self.fd.fileno(), 0, access=mmap.ACCESS_READ)
        from .header import parse_header as ph
        from .metadata import parse_metadata_kv as pmkv
        from .tensor import parse_tensor_descriptors as ptd, calculate_tensor_size as cts
        self.version, self.tensor_count, self.kv_count, consumed = ph(self.mm, 0)
        curr = consumed
        self.metadata, consumed_meta = pmkv(self.mm, curr, self.kv_count)
        curr += consumed_meta
        alignment = self.metadata.get("general.alignment", 32)
        self.tensor_descriptors, consumed_tensors = ptd(self.mm, curr, self.tensor_count)
        curr += consumed_tensors
        self.data_offset = (curr + alignment - 1) & ~(alignment - 1)
        self.tensors_map = {}
        for desc in self.tensor_descriptors:
            abs_offset = self.data_offset + desc["offset"]
            size_bytes = cts(desc["shape"], desc["dtype"])
            if abs_offset + size_bytes > self.file_size:
                raise ValueError(f"Tensor {desc['name']} exceeds file bounds.")
            desc["absolute_offset"] = abs_offset
            desc["byte_size"] = size_bytes
            self.tensors_map[desc["name"]] = desc

    def close(self):
        if self.mm:
            self.mm.close()
        if self.fd:
            self.fd.close()

    def get_metadata(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(key, default)

    def list_tensors(self) -> List[str]:
        return list(self.tensors_map.keys())

    def tensor_info(self, name: str) -> Dict[str, Any]:
        if name not in self.tensors_map:
            raise KeyError(f"Tensor {name} not found.")
        return self.tensors_map[name]

    def read_tensor_slice(self, name: str, offset_bytes: int, length_bytes: int) -> bytes:
        info = self.tensor_info(name)
        if offset_bytes + length_bytes > info["byte_size"]:
            raise ValueError("Slice exceeds tensor bounds.")
        start = info["absolute_offset"] + offset_bytes
        end = start + length_bytes
        return self.mm[start:end]
