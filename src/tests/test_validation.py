from __future__ import annotations

import unittest
from ..architecture.config import ModelConfig
from ..validation.shapes import ShapeValidator
from ..gguf.mmap import GGUFValidationError

class TestValidation(unittest.TestCase):
    def test_valid_shape(self):
        metadata = {"general.architecture": "gemini", "gemini.embedding_length": 4096, "gemini.block_count": 32, "gemini.attention.head_count": 32, "gemini.attention.head_count_kv": 8}
        tensors = [{"name": "token_embd.weight", "shape": [256000, 4096], "dtype": 0, "file_offset": 0, "byte_size": 100, "element_count": 100, "relative_offset": 0}]
        config = ModelConfig(metadata, tensors)
        try:
            ShapeValidator.validate(config, tensors)
        except Exception as e:
            self.fail(f"Validation failed unexpectedly: {e}")

    def test_invalid_hidden_dim(self):
        metadata = {"general.architecture": "gemini", "gemini.embedding_length": 4097, "gemini.block_count": 1, "gemini.attention.head_count": 32}
        tensors = []
        config = ModelConfig(metadata, tensors)
        with self.assertRaises(GGUFValidationError):
            ShapeValidator.validate(config, tensors)
