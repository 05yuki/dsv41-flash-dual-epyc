"""Drop the checkpoint's page cache before each KT layer's CPU-expert load
(KT_GPU_STREAM_DROP_CACHE=<model dir>).

kt-kernel creates a layer's memfd arenas inside that layer's load_weights and
asks for 2 MB pages (MADV_HUGEPAGE; the host needs shmem_enabled=advise). By
then sglang has read the shards and every earlier layer has read its experts,
so the page cache fills memory and from about layer 14 on the kernel finds no
free 2 MB block: 09-24, 200-213 of 335 GB huge, the rest 4K, and the 4K
layers are the ones that pin in 10-900 s (0.2 s on 2 MB pages).
MADV_COLLAPSE afterwards did not help (ENOMEM with the cache full, EINVAL
with it dropped).

Dropping the files' clean pages first (posix_fadvise, no root; 26 s for the
whole V4.1 checkpoint) gives the next layer's arenas free memory in long runs.
Pages still mapped are left alone by the kernel, so a reader that still holds
a shard is not affected; anything dropped and needed again is re-read.

Apply once to source/sglang-dsv41.
"""
from pathlib import Path

F = (Path.home() / "KTransformers/source/sglang-dsv41/python/sglang/srt/layers/moe/kt_ep_wrapper.py")
s = F.read_text()


def once(old, new, text):
    if text.count(old) != 1:
        raise SystemExit("anchor found %d times:\n%s" % (text.count(old), old[:200]))
    return text.replace(old, new)


s = once('''            physical_to_logical_map_cpu = (
                get_global_expert_location_metadata()
                .physical_to_logical_map_cpu[self.kt_config.layer_idx]
                .contiguous()
            )
            self.wrapper.load_weights(physical_to_logical_map_cpu)
''', '''            physical_to_logical_map_cpu = (
                get_global_expert_location_metadata()
                .physical_to_logical_map_cpu[self.kt_config.layer_idx]
                .contiguous()
            )
            _drop_checkpoint_cache(self.kt_config.layer_idx)
            self.wrapper.load_weights(physical_to_logical_map_cpu)
''', s)

# module-level helper, placed above KTConfig's decorator (inserting between
# @dataclass and the class decorated the helper instead, 09-24 11:50)
anchor = "\n@dataclass\nclass KTConfig:"
if s.count(anchor) != 1:
    raise SystemExit("anchor found %d times" % s.count(anchor))
i = s.index(anchor)
helper = '''

def _drop_checkpoint_cache(layer_idx: int) -> None:
    """KT_GPU_STREAM_DROP_CACHE=<dir>: posix_fadvise(DONTNEED) every file
    under it, so the arenas kt-kernel is about to create find free 2 MB
    blocks (patches/sglang-kt-drop-cache-per-layer.py)."""
    import os
    import time

    root_dir = os.environ.get("KT_GPU_STREAM_DROP_CACHE", "")
    if not root_dir:
        return
    t0, n = time.time(), 0
    for root, _, files in os.walk(root_dir, followlinks=True):
        for name in files:
            try:
                fd = os.open(os.path.join(root, name), os.O_RDONLY)
            except OSError:
                continue
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                n += 1
            finally:
                os.close(fd)
    if layer_idx == 0 or os.environ.get("KT_GPU_STREAM_TIMING") == "1":
        print(f"[kt-stream] layer {layer_idx}: dropped the page cache of {n} files in "
              f"{time.time() - t0:.1f} s", flush=True)
'''
s = s[:i] + helper + s[i:]

F.write_text(s)
print("patched", F.name)
