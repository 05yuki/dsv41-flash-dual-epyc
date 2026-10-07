"""The indexer's prefill slices shrink with the prefix
(SGLANG_DSV41_INDEXER_SLICE_MB, default 256; on top of
sglang-dsv4-indexer-query-step.py).

A slice of SGLANG_DSV41_INDEXER_QUERY_STEP rows (512) scores every compressed
position so far in fp32: 512 x prefix x 4 bytes, 708 MiB at a ~356K-token
prefix and 2 GiB at 1M. That one allocation is what a long V4.1 prefill ran
out of memory on once the candidate masks were made small (10-06). The rows
of a slice are now capped so that its logits stay within the budget (at
least 16 rows); short prefixes keep the 512. Same numbers, more slices.

Usage: python sglang-dsv4-indexer-slice-budget.py [sglang tree]   (default source/sglang-dsv41)
"""
import sys
from pathlib import Path

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "KTransformers/source/sglang-dsv41"
F = root / "python/sglang/srt/layers/attention/deepseek_v4_backend.py"
s = F.read_text()
MARK = "SGLANG_DSV41_INDEXER_SLICE_MB"
if MARK in s:
    print(f"already patched {F.name}")
    raise SystemExit(0)


def once(old, new, text):
    if text.count(old) != 1:
        raise SystemExit("anchor found %d times:\n%s" % (text.count(old), old[:300]))
    return text.replace(old, new)


s = once('''_INDEXER_QUERY_STEP = int(__import__("os").environ.get("SGLANG_DSV41_INDEXER_QUERY_STEP", "512"))
''', '''_INDEXER_QUERY_STEP = int(__import__("os").environ.get("SGLANG_DSV41_INDEXER_QUERY_STEP", "512"))
# a slice's fp32 logits [rows, prefix] stay within this many MiB (long prefixes)
_INDEXER_SLICE_BYTES = int(__import__("os").environ.get("SGLANG_DSV41_INDEXER_SLICE_MB", "256")) << 20
''', s)
s = once('''                for s0 in range(0, t_len, step):
                    rows = slice(r0 + s0, r0 + min(s0 + step, t_len))
''', '''                # fewer rows a slice where the prefix is long (SGLANG_DSV41_INDEXER_SLICE_MB)
                step_b = max(16, min(step, _INDEXER_SLICE_BYTES // (ceil_align(lc, 4) * 4)))
                for s0 in range(0, t_len, step_b):
                    rows = slice(r0 + s0, r0 + min(s0 + step_b, t_len))
''', s)
F.write_text(s)
print("patched", F.name)
