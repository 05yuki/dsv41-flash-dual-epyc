"""Skip the CPU experts' checkpoint tensors before the model's weight loop
looks at them (KT_LOAD_SKIP_CPU_EXPERTS=0 turns it off).

Before the first expert layer, a V4.1 launch spends 74 s walking the 48
shards (10-07: 1.6 s a shard, steady, both TP ranks). The shards are 96%
routed-expert tensors, about 100K of them, and each one goes through the
name remap, the 768-entry expert mapping scan and a weight_loader call that
returns once it sees the expert is not on the GPU. Nothing is read from disk
for them; the time is the Python per tensor.

With this patch load_weights asks each KT MoE layer which logical experts
have no GPU slot (the same logical -> physical -> local mapping and
num_gpu_experts check the weight loader applies) and drops those tensors
right after the name remap, before the mapping scan.

Usage: python sglang-dsv4-load-skip-cpu-experts.py <sglang tree>
"""
import sys
from pathlib import Path

if len(sys.argv) < 2:
    raise SystemExit(f"usage: python {Path(__file__).name} <sglang tree>")
F = Path(sys.argv[1]) / "python/sglang/srt/models/deepseek_v4.py"
s = F.read_text()
if "KT_LOAD_SKIP_CPU_EXPERTS" in s:
    print(f"already patched {F}")
    raise SystemExit(0)


def once(old, new, text):
    if text.count(old) != 1:
        raise SystemExit("anchor found %d times:\n%s" % (text.count(old), old[:200]))
    return text.replace(old, new)


HELPER = '''

_KT_EXPERT_NAME = re.compile(r"^model\\.layers\\.(\\d+)\\.mlp\\.experts\\.(\\d+)\\.")


def _kt_cpu_only_experts(model) -> dict:
    """{layer_id: frozenset(logical expert ids with no GPU slot)} for every
    MoE layer whose quant method is the KT wrapper; {} when there is none or
    KT_LOAD_SKIP_CPU_EXPERTS=0."""
    import os

    if os.environ.get("KT_LOAD_SKIP_CPU_EXPERTS", "1") != "1":
        return {}
    from sglang.srt.eplb.expert_location import get_global_expert_location_metadata
    from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
    from sglang.srt.layers.moe.kt_ep_wrapper import KTEPWrapperMethod

    meta = get_global_expert_location_metadata()
    out = {}
    for module in model.modules():
        qm = getattr(module, "quant_method", None)
        if not isinstance(module, FusedMoE) or not isinstance(qm, KTEPWrapperMethod):
            continue
        if qm.num_gpu_experts == -1 or getattr(module, "layer_id", None) is None:
            continue
        cpu_only = set()
        for logical in range(meta.num_logical_experts):
            on_gpu = False
            for phys in meta.logical_to_all_physical(module.layer_id, logical):
                local = module._map_global_expert_id_to_local_expert_id(phys)
                if 0 <= local < module.num_local_experts and local < qm.num_gpu_experts:
                    on_gpu = True
                    break
            if not on_gpu:
                cpu_only.add(logical)
        out[module.layer_id] = frozenset(cpu_only)
    return out
'''

s = once(
    "from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation\n",
    "from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation\n" + HELPER,
    s,
)
if "\nimport re\n" not in s:
    s = once("\nimport torch\n", "\nimport re\nimport torch\n", s)

s = once(
    """        cache_compressor_weight = {}
        COMPRESSOR_PART = ".compressor.w"
""",
    """        cache_compressor_weight = {}
        COMPRESSOR_PART = ".compressor.w"
        kt_cpu_only = _kt_cpu_only_experts(self) if not is_nextn else {}
        kt_skipped = 0
""",
    s,
)
s = once(
    """                    name = self.remap_weight_name_to_dpsk_hf_format(
                        name,
                        is_nextn=is_nextn,
                        num_hidden_layers=self.config.num_hidden_layers,
                    )

""",
    """                    name = self.remap_weight_name_to_dpsk_hf_format(
                        name,
                        is_nextn=is_nextn,
                        num_hidden_layers=self.config.num_hidden_layers,
                    )
                    if kt_cpu_only:
                        m = _KT_EXPERT_NAME.match(name)
                        if m and int(m.group(2)) in kt_cpu_only.get(int(m.group(1)), ()):
                            kt_skipped += 1
                            continue

""",
    s,
)
F.write_text(s)
print(f"patched {F}")
