"""Check the streamed expert clamp without importing Torch or CUDA modules.

By default this checks the streamer bundled in the current tree patch.
--source can instead check an installed/patched kt_stream_prefill.py.
"""

import argparse
import ast
import unittest
from pathlib import Path
from types import SimpleNamespace


def bundled_source():
    patch = (
        Path(__file__).resolve().parents[1]
        / "patches"
        / "sglang-dsv41-tree-20261007.patch"
    ).read_text()
    name = "python/sglang/srt/layers/moe/kt_stream_prefill.py"
    added = patch.split(f"diff --git a/{name} b/{name}\n", 1)[1].split(
        "\ndiff --git ", 1
    )[0]
    lines = added.split("\n@@ ", 1)[1].split("\n", 1)[1].splitlines()
    if not all(line.startswith("+") for line in lines):
        raise AssertionError("Expected the streamer to be added as one complete hunk")
    return "\n".join(line[1:] for line in lines) + "\n"


class StreamClampTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse(SOURCE)
        cls.streamer = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "KTStreamPrefill"
        )

    def resolve(self, gpu_method):
        initializer = next(
            node
            for node in self.streamer.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        first = next(
            i
            for i, node in enumerate(initializer.body)
            if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "gm"
        )
        last = next(
            i
            for i, node in enumerate(initializer.body)
            if isinstance(node, ast.Assign)
            and ast.unparse(node.targets[0]) == "self.swiglu_limit"
        )
        policy = ast.Module(body=initializer.body[first : last + 1], type_ignores=[])
        target = SimpleNamespace(G=4, device="cpu")
        fake_torch = SimpleNamespace(
            float32="float32", full=lambda shape, value, **kw: value
        )
        exec(
            compile(policy, "stream-clamp-policy", "exec"),
            {
                "method": SimpleNamespace(gpu_method=gpu_method),
                "self": target,
                "torch": fake_torch,
            },
        )
        return target.swiglu_limit

    def test_current_marlin_configuration(self):
        method = SimpleNamespace(moe_runner_config=SimpleNamespace(swiglu_limit=10.0))
        self.assertEqual(self.resolve(method), 10.0)

    def test_configuration_takes_precedence(self):
        method = SimpleNamespace(
            moe_runner_config=SimpleNamespace(swiglu_limit=10.0),
            _swiglu_limit_value=7.0,
        )
        self.assertEqual(self.resolve(method), 10.0)

    def test_legacy_scalar(self):
        self.assertEqual(self.resolve(SimpleNamespace(_swiglu_limit_value=7.0)), 7.0)

    def test_legacy_tensor(self):
        class Tensor:
            def numel(self):
                return 1

            def __getitem__(self, index):
                return 5.0

        self.assertEqual(
            self.resolve(SimpleNamespace(_swiglu_limit_tensor=Tensor())), 5.0
        )

    def test_no_clamp(self):
        self.assertIsNone(self.resolve(SimpleNamespace()))

    def test_every_cutlass_group_call_receives_clamp(self):
        calls = [
            node
            for node in ast.walk(self.streamer)
            if isinstance(node, ast.Call)
            and ast.unparse(node.func) == "self._cutlass_fused_moe"
        ]
        self.assertEqual(len(calls), 3)
        for call in calls:
            keyword = next(
                (kw for kw in call.keywords if kw.arg == "swiglu_limit"), None
            )
            self.assertIsNotNone(keyword)
            self.assertIn("self.swiglu_limit", ast.unparse(keyword.value))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, help="Patched kt_stream_prefill.py to check"
    )
    args, remaining = parser.parse_known_args()
    SOURCE = args.source.read_text() if args.source else bundled_source()
    unittest.main(argv=[parser.prog, *remaining])
