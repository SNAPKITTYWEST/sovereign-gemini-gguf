from __future__ import annotations

import os
import struct
import tempfile
import unittest

from ..gguf.reader import GGUFParser
from ..gguf.mmap import GGUFSecurityError
from ..gguf.metadata import GGUFValueType
from ..gguf.tensor import GGUFTensorType

def create_mock_gguf_file(filepath: str):
    with open(filepath, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))
        f.write(struct.pack("<Q", 2))
        f.write(struct.pack("<Q", 4))
        f.write(struct.pack("<Q", len("general.architecture")))
        f.write("general.architecture".encode("utf-8"))
        f.write(struct.pack("<I", GGUFValueType.STRING))
        f.write(struct.pack("<Q", len("gemini")))
        f.write("gemini".encode("utf-8"))
        f.write(struct.pack("<Q", len("gemini.embedding_length")))
        f.write("gemini.embedding_length".encode("utf-8"))
        f.write(struct.pack("<I", GGUFValueType.UINT32))
        f.write(struct.pack("<I", 64))
        f.write(struct.pack("<Q", len("gemini.block_count")))
        f.write("gemini.block_count".encode("utf-8"))
        f.write(struct.pack("<I", GGUFValueType.UINT32))
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<Q", len("gemini.attention.head_count")))
        f.write("gemini.attention.head_count".encode("utf-8"))
        f.write(struct.pack("<I", GGUFValueType.UINT32))
        f.write(struct.pack("<I", 4))
        name1 = "token_embd.weight"
        f.write(struct.pack("<Q", len(name1)))
        f.write(name1.encode("utf-8"))
        f.write(struct.pack("<I", 2))
        f.write(struct.pack("<Q", 64))
        f.write(struct.pack("<Q", 128))
        f.write(struct.pack("<I", GGUFTensorType.F32))
        f.write(struct.pack("<Q", 0))
        name2 = "blk.0.attn_q.weight"
        f.write(struct.pack("<Q", len(name2)))
        f.write(name2.encode("utf-8"))
        f.write(struct.pack("<I", 2))
        f.write(struct.pack("<Q", 64))
        f.write(struct.pack("<Q", 64))
        f.write(struct.pack("<I", GGUFTensorType.F32))
        f.write(struct.pack("<Q", 32768))
        curr = f.tell()
        padding = (32 - (curr % 32)) % 32
        f.write(b'\x00' * padding)
        f.write(b'\x00' * (64 * 128 * 4))
        f.write(b'\x00' * (64 * 64 * 4))

class TestGGUFParser(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.mock_gguf = os.path.join(self.tmp_dir, "mock_gemini.gguf")
        create_mock_gguf_file(self.mock_gguf)

    def tearDown(self):
        if os.path.exists(self.mock_gguf):
            os.remove(self.mock_gguf)
        os.rmdir(self.tmp_dir)

    def test_header_and_metadata_parsing(self):
        parser = GGUFParser(self.mock_gguf)
        parser.parse()
        self.assertEqual(parser.version, 3)
        self.assertEqual(parser.tensor_count, 2)
        self.assertEqual(parser.metadata["general.architecture"], "gemini")
        self.assertEqual(parser.metadata["gemini.embedding_length"], 64)
        parser.close()

    def test_tensor_indexing_and_slicing(self):
        parser = GGUFParser(self.mock_gguf)
        parser.parse()
        self.assertIn("token_embd.weight", parser.tensor_index)
        t1 = parser.tensor_index["token_embd.weight"]
        self.assertEqual(t1["shape"], [64, 128])
        self.assertEqual(t1["byte_size"], 64 * 128 * 4)
        data_slice = parser.read_tensor_slice("token_embd.weight", offset_elements=0, count_elements=10)
        self.assertEqual(len(data_slice), 10 * 4)
        parser.close()

    def test_invalid_magic(self):
        bad_gguf = os.path.join(self.tmp_dir, "bad_magic.gguf")
        with open(bad_gguf, "wb") as f:
            f.write(b"BADM\x03\x00\x00\x00")
        with self.assertRaises(GGUFSecurityError):
            parser = GGUFParser(bad_gguf)
            parser.parse()
        os.remove(bad_gguf)

def run_unit_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGGUFParser)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        import sys
        sys.exit(1)
