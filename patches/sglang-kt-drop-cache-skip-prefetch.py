"""With KT_LOAD_PREFETCH on, the per-layer drop of the whole checkpoint's
page cache (KT_GPU_STREAM_DROP_CACHE) would evict the next layer's
background read. Keep layer 0's drop (it clears the dense shards sglang
just read); skip the rest, the loader evicts each finished layer itself.

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
    if layer_idx > 0 and int(os.environ.get("KT_LOAD_PREFETCH", "0") or 0) > 0:
        return
'''
if s.count(old) != 1:
    raise SystemExit("anchor found %d times" % s.count(old))
F.write_text(s.replace(old, new))
print(f"patched {F}")
