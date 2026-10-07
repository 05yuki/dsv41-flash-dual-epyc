"""Read each shard's dense tensors into the page cache with parallel preads
as the shard loader opens it (KT_LOAD_PREFETCH=<threads>, shared with the
kt-kernel loader's knob; 0 = off).

Before the expert layers, sglang's shard iterator hands the model zero-copy
mmap views and the model's weight loaders fault the dense tensors in while
copying them to the GPU, one readahead window at a time: 10-07, V4.1, each
TP rank read 64 GB at 0.85 GB/s (/proc/<pid>/io), 75 of the 95 s before the
first expert layer, with the drive able to do 6 GiB/s. The CPU experts are
never touched here (their weight loader returns before reading), so only
tensors outside `.experts.` (routed experts, `layers.N.ffn.experts.E.` in this checkpoint) and outside the engram tables (`.engram.embed.`, 2 x 94 GiB, read at inference from the NVMe) are prefetched; the hot experts stay on the
fault path (about 5 GB).

Usage: python sglang-dense-prefetch.py <sglang tree>
"""
import sys
from pathlib import Path

if len(sys.argv) < 2:
    raise SystemExit(f"usage: python {Path(__file__).name} <sglang tree>")
F = Path(sys.argv[1]) / "python/sglang/srt/model_loader/weight_utils.py"
s = F.read_text()
if "KT_LOAD_PREFETCH" in s:
    print(f"already patched {F}")
    raise SystemExit(0)


def once(old, new, text):
    if text.count(old) != 1:
        raise SystemExit("anchor found %d times:\n%s" % (text.count(old), old[:200]))
    return text.replace(old, new)


HELPER = '''

_DENSE_PREFETCH_THREADS = int(os.environ.get("KT_LOAD_PREFETCH", "0") or 0)
_DENSE_PREFETCH_PIECE = int(os.environ.get("KT_LOAD_PREFETCH_PIECE_MB", "16")) << 20
_DENSE_PREFETCH_SKIP = re.compile(os.environ.get("KT_LOAD_PREFETCH_SKIP", r"\.experts\.|\.engram\.embed\."))


def _prefetch_dense_tensors(st_file: str) -> None:
    """KT_LOAD_PREFETCH=<threads>: pread this shard's tensors whose names do
    not match KT_LOAD_PREFETCH_SKIP (default: routed experts and the engram tables; `shared_experts` does not match) in 16 MiB
    pieces on a thread pool, so the weight loaders' mmap faults find the
    pages in the cache instead of reading one readahead window at a time."""
    if _DENSE_PREFETCH_THREADS <= 0:
        return
    import time

    t0 = time.time()
    with open(st_file, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        meta = json.loads(fh.read(n))
    ranges = []
    for k, v in meta.items():
        if k == "__metadata__" or _DENSE_PREFETCH_SKIP.search(k):
            continue
        b, e = v["data_offsets"]
        ranges.append((8 + n + b, e - b))
    if not ranges:
        return
    ranges.sort()
    merged = []
    cur_off, cur_end = ranges[0][0], ranges[0][0] + ranges[0][1]
    for off, sz in ranges[1:]:
        if off <= cur_end + (1 << 20):
            cur_end = max(cur_end, off + sz)
        else:
            merged.append((cur_off, cur_end))
            cur_off, cur_end = off, off + sz
    merged.append((cur_off, cur_end))
    pieces = []
    for a, b in merged:
        p = a
        while p < b:
            pieces.append((p, min(_DENSE_PREFETCH_PIECE, b - p)))
            p += _DENSE_PREFETCH_PIECE
    local = threading.local()
    fd = os.open(st_file, os.O_RDONLY)

    def read(piece):
        off, sz = piece
        buf = getattr(local, "buf", None)
        if buf is None or len(buf) < sz:
            buf = local.buf = bytearray(max(sz, _DENSE_PREFETCH_PIECE))
        mv = memoryview(buf)[:sz]
        got = 0
        while got < sz:
            r = os.preadv(fd, [mv[got:]], off + got)
            if r <= 0:
                break
            got += r
        return got

    try:
        with concurrent.futures.ThreadPoolExecutor(_DENSE_PREFETCH_THREADS) as ex:
            done = sum(ex.map(read, pieces))
    finally:
        os.close(fd)
    logger.info(
        "[dense prefetch] %s: %.2f GiB in %.2f s (%d tensors)",
        os.path.basename(st_file), done / 2**30, time.time() - t0, len(ranges),
    )
'''

s = once(
    "def buffered_multi_thread_safetensors_weights_iterator(\n",
    HELPER.lstrip("\n") + "\n\ndef buffered_multi_thread_safetensors_weights_iterator(\n",
    s,
)
s = once(
    '''        else:
            with safetensors.safe_open(st_file, framework="pt", device="cpu") as f:
                result = {k: f.get_tensor(k) for k in f.keys()}
        return result

    # Sliding window: max_workers loading + 1 prefetched.
''',
    '''        else:
            _prefetch_dense_tensors(st_file)
            with safetensors.safe_open(st_file, framework="pt", device="cpu") as f:
                result = {k: f.get_tensor(k) for k in f.keys()}
        return result

    # Sliding window: max_workers loading + 1 prefetched.
''',
    s,
)
F.write_text(s)
print(f"patched {F}")
