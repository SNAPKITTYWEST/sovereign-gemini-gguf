from __future__ import annotations

import struct
from typing import Tuple

GGUF_MAGIC = b"GGUF"
SUPPORTED_VERSIONS = {2, 3}

class GGUFHeaderError(ValueError):
    pass

def parse_header(mm, offset: int = 0) -> Tuple[int, int, int, int]:
    """
    Parses and validates the GGUF binary header.
    Returns (version, tensor_count, metadata_kv_count, bytes_read).
    """
    if len(mm) < offset + 24:
        raise GGUFHeaderError("File too small to contain a valid GGUF header.")

    magic = mm[offset:offset+4]
    if magic != GGUF_MAGIC:
        raise GGUFHeaderError(f"Invalid GGUF magic bytes: {magic!r}. Expected {GGUF_MAGIC!r}.")

    version, tensor_count, kv_count = struct.unpack_from("<IQQ", mm, offset + 4)
    if version not in SUPPORTED_VERSIONS:
        raise GGUFHeaderError(f"Unsupported GGUF version: {version}. Supported versions: {SUPPORTED_VERSIONS}")

    return version, tensor_count, kv_count, 24
