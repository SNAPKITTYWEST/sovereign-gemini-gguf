from __future__ import annotations

import os
import mmap
import struct
from typing import Any

MAX_STRING_LENGTH = 1_048_576

class GGUFError(Exception):
    pass

class GGUFSecurityError(GGUFError):
    pass

class GGUFValidationError(GGUFError):
    pass

class GGUFMmapReader:
    """
    Zero-copy memory-mapped binary reader for GGUF files.
    Enforces strict boundary bounds, prevent allocation attacks, and allows
    lazy tensor slice reads without copying underlying byte buffers.
    """
    def __init__(self, filepath: str):
        if not os.path.exists(filepath):
            raise GGUFError(f"File not found: {filepath}")
        self.filepath = os.path.abspath(filepath)
        self.file_size = os.path.getsize(filepath)
        if self.file_size < 12:
            raise GGUFSecurityError("File size too small to contain valid GGUF header.")

        self.fd = open(filepath, "rb")
        try:
            self.mm = mmap.mmap(self.fd.fileno(), 0, access=mmap.ACCESS_READ)
        except Exception as e:
            self.fd.close()
            raise GGUFSecurityError(f"Failed to memory-map file: {e}")
        self.offset = 0

    def close(self):
        if hasattr(self, 'mm') and self.mm:
            self.mm.close()
            self.mm = None
        if hasattr(self, 'fd') and self.fd:
            self.fd.close()
            self.fd = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def seek(self, offset: int):
        if offset < 0 or offset > self.file_size:
            raise GGUFSecurityError(f"Seek offset {offset} out of bounds [0, {self.file_size}]")
        self.offset = offset

    def tell(self) -> int:
        return self.offset

    def read(self, length: int) -> bytes:
        if length < 0:
            raise GGUFSecurityError(f"Negative read length requested: {length}")
        end_offset = self.offset + length
        if end_offset > self.file_size:
            raise GGUFSecurityError(f"Read buffer overrun: requested {length} bytes at offset {self.offset}, file size is {self.file_size}")
        data = self.mm[self.offset:end_offset]
        self.offset = end_offset
        return data

    def read_value(self, fmt: str) -> Any:
        size = struct.calcsize(fmt)
        data = self.read(size)
        return struct.unpack(fmt, data)[0]

    def read_string(self) -> str:
        length = self.read_value("<Q")
        if length > MAX_STRING_LENGTH:
            raise GGUFSecurityError(f"String length {length} exceeds maximum safety threshold ({MAX_STRING_LENGTH})")
        data = self.read(length)
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as e:
            raise GGUFValidationError(f"Invalid UTF-8 string encoding in GGUF metadata: {e}")

    def read_slice(self, abs_offset: int, length: int) -> memoryview:
        if abs_offset < 0 or length < 0 or (abs_offset + length) > self.file_size:
            raise GGUFSecurityError(f"Slice range [{abs_offset}, {abs_offset + length}] exceeds file bounds ({self.file_size})")
        return memoryview(self.mm)[abs_offset:abs_offset + length]
