"""MXFP4 group-32 decode without vinserti128.

The fast path's per-group decode is bound by Zen2's shuffle ports, not by
memory bandwidth: adding one broadcast to it (E8M0 scales, 09-23) cost 18% on
gate+up even though it cut the bytes read by 15%. Nine shuffle-class uops per
group, one of them only to join the low and high nibble halves with
_mm256_set_m128i. Here the 16 raw bytes are broadcast to both lanes by the load
itself (vbroadcasti128 from memory) and a per-lane variable shift (0 low, 4 high)
does the split, so the join disappears. w0..w3 are bit-identical for every
input (kt-mxfp4-decode-noinsert-test.cpp, 2^20 random groups and every byte in
every position), so the output is too.

Apply once to source/ktransformers-gemma4/kt-kernel, then
native-ubuntu/build-kt-dsv41.sh.
Usage: python kt-mxfp4-decode-noinsert.py [ktransformers tree]   (default source/ktransformers-gemma4)
"""
import sys
from pathlib import Path

F = (Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "KTransformers/source/ktransformers-gemma4") / "kt-kernel/operators/avx2/mxfp4-moe.hpp"
s = F.read_text()
if "lane_shift" in s:
    print(f"already patched {F.name}")
    raise SystemExit(0)
start = s.index("if (group_size == 32 && (k % 32) == 0")
end = s.index("#undef KT_MXFP4_DECODE_GROUP", start)
fast = s[start:end]

old_consts = "    const __m128i nib_mask = _mm_set1_epi8(0x0F);\n"
if fast.count(old_consts) != 1:
    raise SystemExit("nib_mask anchor")
fast = fast.replace(old_consts, old_consts +
                    "    const __m256i nib_mask256 = _mm256_set1_epi8(0x0F);\n"
                    "    const __m256i lane_shift = _mm256_setr_epi32(0, 0, 0, 0, 4, 4, 4, 4);\n")

lines = fast.split("\n")
out, replaced = [], 0
i = 0
while i < len(lines):
    ln = lines[i]
    if ln.lstrip().startswith("const __m128i raw = _mm_loadu_si128((const __m128i*)((b_row) + (size_t)(g)*16));"):
        # the four lines raw / lo / hi / v become two
        nxt = [l.lstrip() for l in lines[i + 1:i + 4]]
        if not (nxt[0].startswith("const __m128i lo =") and nxt[1].startswith("const __m128i hi =")
                and nxt[2].startswith("const __m256i v = _mm256_set_m128i(hi, lo);")):
            raise SystemExit("unexpected decode macro shape")
        out.append("  const __m256i r2 = _mm256_broadcastsi128_si256(                                 \\")
        out.append("      _mm_loadu_si128((const __m128i*)((b_row) + (size_t)(g)*16)));                 \\")
        out.append("  const __m256i v = _mm256_and_si256(_mm256_srlv_epi32(r2, lane_shift), nib_mask256); \\")
        i += 4
        replaced += 1
        continue
    out.append(ln)
    i += 1
if replaced != 1:
    raise SystemExit("replaced %d decode macros" % replaced)
fast = "\n".join(out)
if "nib_mask)" in fast.split("#define KT_MXFP4_DECODE_GROUP")[1]:
    raise SystemExit("macro still references the 128-bit mask")
F.write_text(s[:start] + fast + s[end:])
print("patched", F)
