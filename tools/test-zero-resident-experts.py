"""Exercise resident-weight postprocessing without loading CUDA or model weights."""

import argparse
import ast
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


class ZeroResidentExpertsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse(SOURCE.read_text())
        wrapper = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "KTEPWrapperMethod"
        )
        method = next(
            node
            for node in wrapper.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "process_weights_after_loading"
        )
        # Execute the complete method, including CPU loading and early streamer init.
        cls.tree = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                method,
            ],
            type_ignores=[],
        )
        ast.fix_missing_locations(cls.tree)

    def run_postprocess(self, experts, rank=0, last_layer=False, gpu_postprocess=True):
        gpu = SimpleNamespace()
        if gpu_postprocess:
            gpu.process_weights_after_loading = Mock()
        wrapper = SimpleNamespace(load_weights=Mock())
        target = SimpleNamespace(
            num_gpu_experts=experts,
            gpu_method=gpu,
            tp_rank=rank,
            wrapper=wrapper,
            kt_config=SimpleNamespace(layer_idx=2 if last_layer else 0, num_layers=3),
        )
        layer = object()
        mapping = Mock()
        metadata = SimpleNamespace(
            physical_to_logical_map_cpu={
                target.kt_config.layer_idx: SimpleNamespace(contiguous=lambda: mapping),
            }
        )
        early = Mock()
        synchronize = Mock()
        namespace = {
            "torch": SimpleNamespace(cuda=SimpleNamespace(synchronize=synchronize)),
            "_drop_checkpoint_cache": Mock(),
        }
        exec(compile(self.tree, "resident-weight-postprocess", "exec"), namespace)
        modules = {
            "sglang.srt.eplb.expert_location_dispatch": SimpleNamespace(
                get_global_expert_location_metadata=lambda: metadata
            ),
            "sglang.srt.layers.moe.kt_stream_prefill": SimpleNamespace(
                stream_prefill_init_early=early
            ),
        }
        with patch.dict(sys.modules, modules):
            namespace["process_weights_after_loading"](target, layer)
        return target, layer, mapping, early, synchronize

    def test_zero_skips_gpu_repacking_but_loads_cpu_weights(self):
        target, _, mapping, early, synchronize = self.run_postprocess(0)
        target.gpu_method.process_weights_after_loading.assert_not_called()
        target.wrapper.load_weights.assert_called_once_with(mapping)
        synchronize.assert_called_once_with()
        early.assert_not_called()

    def test_positive_counts_keep_gpu_and_cpu_postprocessing(self):
        for experts in (1, 5):
            with self.subTest(experts=experts):
                target, layer, mapping, _, _ = self.run_postprocess(experts)
                target.gpu_method.process_weights_after_loading.assert_called_once_with(
                    layer
                )
                target.wrapper.load_weights.assert_called_once_with(mapping)

    def test_zero_last_layer_initializes_streamer_on_both_ranks(self):
        for rank in (0, 1):
            with self.subTest(rank=rank):
                target, layer, _, early, synchronize = self.run_postprocess(
                    0, rank, True
                )
                target.gpu_method.process_weights_after_loading.assert_not_called()
                early.assert_called_once_with(target, layer)
                self.assertEqual(target.wrapper.load_weights.call_count, int(rank == 0))
                self.assertEqual(synchronize.call_count, int(rank == 0))

    def test_optional_gpu_postprocessor_stays_optional(self):
        target, _, mapping, _, _ = self.run_postprocess(5, gpu_postprocess=False)
        target.wrapper.load_weights.assert_called_once_with(mapping)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True, help="Patched kt_ep_wrapper.py to check"
    )
    args, remaining = parser.parse_known_args()
    SOURCE = args.source
    unittest.main(argv=[parser.prog, *remaining])
