"""Execute the actual Triton padding branch without importing Torch or CUDA.

Requires NumPy and a patched flash_mla_sm120.py passed with --source.
"""

import argparse
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


class Pointer:
    def __init__(self, values):
        self.values = values

    def to(self, dtype):
        return self

    def __add__(self, offsets):
        return self.values, offsets


class PaddingSentinelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse(SOURCE.read_text())
        kernel = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_page_split_kernel"
        )
        kernel.decorator_list = []
        cls.tree = ast.Module(body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            kernel,
        ], type_ignores=[])
        ast.fix_missing_locations(cls.tree)

    def test_reserved_page_is_cleared_before_mask_skip(self):
        for masked in (False, True):
            for block_size in (256, 1024):
                with self.subTest(masked=masked, block_size=block_size):
                    words = 37440 // 8
                    output = np.full(words * 8, np.iinfo(np.uint64).max, dtype=np.uint64)

                    def store(pointer, value, mask):
                        values, offsets = pointer
                        values[offsets[mask]] = value[mask]

                    def unexpected_load(*args, **kwargs):
                        self.fail("Sentinel initialization must precede mask/source reads")

                    tl = SimpleNamespace(
                        program_id=lambda axis: 0, range=range, arange=np.arange,
                        uint64=np.uint64, pointer_type=lambda dtype: dtype,
                        full=np.full, store=store, load=unexpected_load,
                    )
                    namespace = {"tl": tl}
                    exec(compile(self.tree, "padding-sentinel-kernel", "exec"), namespace)
                    source = Pointer(np.empty(0, dtype=np.uint64))
                    touched = Pointer(np.empty(0, dtype=np.int8))
                    namespace["_page_split_kernel"](
                        source, Pointer(output), 2, 149760, 37440, 36864, 512,
                        147456, 36864, 4, block_size, touched, masked,
                    )
                    np.testing.assert_array_equal(output[:words], 0)
                    np.testing.assert_array_equal(output[words:], np.iinfo(np.uint64).max)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Patched flash_mla_sm120.py")
    args, remaining = parser.parse_known_args()
    SOURCE = args.source
    unittest.main(argv=[parser.prog, *remaining])
