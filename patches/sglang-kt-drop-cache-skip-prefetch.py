"""With KT_LOAD_PREFETCH on, the per-layer drop of the whole checkpoint's
page cache (KT_GPU_STREAM_DROP_CACHE) would evict the next layer's
background read. Keep layer 0's drop (it clears the dense shards sglang
just read) and the drops of the last KT_LOAD_PREFETCH_TAIL layers (the
loader reads those after their arena exists); skip the rest, the loader evicts each finished layer itself.

Usage: python sglang-kt-drop-cache-skip-prefetch.py <sglang tree>
"""
import sys
from pathlib import Path

if len(sys.argv) < 2:
    raise SystemExit(f"usage: python {Path(__file__).name} <sglang tree>")
F = Path(sys.argv[1]) / "python/sglang/srt/layers/moe/kt_ep_wrapper.py"
s = F.read_text()
if "KT_LOAD_PREFETCH" in s:
    print(f"already patched {F}")
    raise SystemExit(0)
old = '''    root_dir = os.environ.get("KT_GPU_STREAM_DROP_CACHE", "")
    if not root_dir:
        return
'''
new = '''    root_dir = os.environ.get("KT_GPU_STREAM_DROP_CACHE", "")
    if not root_dir:
        return
    # the loader's prefetch evicts each finished layer itself; a whole-tree
    # drop here would throw away the next layer's background read
    # (the loader reads the last KT_LOAD_PREFETCH_TAIL layers after their
    # arena exists; this drop runs again for them so the arena finds 2 MB
    # blocks, as it did before the prefetch)
    if layer_idx > 0 and int(os.environ.get("KT_LOAD_PREFETCH", "0") or 0) > 0:
        tail = int(os.environ.get("KT_LOAD_PREFETCH_TAIL", "4"))
        if not num_layers or layer_idx < num_layers - tail - 1:
            return
'''
if s.count(old) != 1:
    raise SystemExit("anchor found %d times" % s.count(old))
s = s.replace(old, new)
old2 = "def _drop_checkpoint_cache(layer_idx: int) -> None:\n"
new2 = "def _drop_checkpoint_cache(layer_idx: int, num_layers: int = 0) -> None:\n"
if s.count(old2) != 1:
    raise SystemExit("signature anchor found %d times" % s.count(old2))
s = s.replace(old2, new2)
old3 = "            _drop_checkpoint_cache(self.kt_config.layer_idx)\n"
new3 = "            _drop_checkpoint_cache(self.kt_config.layer_idx, getattr(self.kt_config, \"num_layers\", 0) or 0)\n"
if s.count(old3) != 1:
    raise SystemExit("call anchor found %d times" % s.count(old3))
s = s.replace(old3, new3)
F.write_text(s)
print(f"patched {F}")
