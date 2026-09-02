from __future__ import annotations

import unittest
from ..validation.offsets import OffsetValidator
from ..gguf.mmap import GGUFSecurityError

class TestSecurity(unittest.TestCase):
    def test_no_overlap_pass(self):
        tensors = [
            {"name": "a", "file_offset": 0, "byte_size": 100},
            {"name": "b", "file_offset": 100, "byte_size": 100},
        ]
        try:
            OffsetValidator.validate_no_overlap(tensors)
        except Exception as e:
            self.fail(f"Overlap check failed: {e}")

    def test_overlap_detect(self):
        tensors = [
            {"name": "a", "file_offset": 0, "byte_size": 150},
            {"name": "b", "file_offset": 100, "byte_size": 100},
        ]
        with self.assertRaises(GGUFSecurityError):
            OffsetValidator.validate_no_overlap(tensors)

    def test_bounds_check(self):
        tensors = [{"name": "a", "file_offset": 0, "byte_size": 1000, "relative_offset": 0}]
        with self.assertRaises(GGUFSecurityError):
            OffsetValidator.validate_bounds(tensors, file_size=500)
