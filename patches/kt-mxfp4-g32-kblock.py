"""Group-32 prefill GEMM: k-blocked so a 4-token block's activations stay in L1.

On top of kt-mxfp4-g32-loop-order.py (and row-decode, default off). With rows
outer, a 4-token block still reads 4 x k FP32 activations (64 KiB at k=4096)
per output column; that does not fit a Zen 2 core's 32 KiB L1, so every
column streams it from L2 (16 x 32 B loads per 32-value group per block, ~16
cycles against ~10 for the FMAs). Here k is cut into blocks of
KT_MXFP4_G32_KBLK groups (default 32 = 1024 values: 16 KiB of activations for
the block), and each column's per-token 8-lane running sums are parked in a
small buffer between k blocks. Parking an FP32 vector and loading it back is
exact and the groups are still summed in order, so the result is
bit-identical. KT_MXFP4_G32_KBLK=0 turns it off.
"""
import sys
from pathlib import Path

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "KTransformers/source/ktransformers-gemma4"
f = root / "kt-kernel/operators/avx2/mxfp4-moe.hpp"
s = f.read_text(encoding="utf-8")

MARK = "// k-blocked (group-32)."
if MARK in s:
    print("already applied")
    sys.exit(0)
anchor = "    int mi = 0;\n    if (kTile >= 8 && m >= 8) {\n"
if s.count(anchor) != 1:
    raise SystemExit("apply kt-mxfp4-g32-row-decode.py first (anchor found %d times)" % s.count(anchor))

block = r'''    // k-blocked (group-32). A 4-token block's activations for kBlk groups
    // (16 KiB at 32 groups) stay in L1 while this thread's columns go by; each
    // column's 8-lane running sums are parked between k blocks. Exact parking,
    // groups summed in order: bit-identical.
    static const int kBlk = [] {
      const char* v = std::getenv("KT_MXFP4_G32_KBLK");
      return v ? std::max(0, std::atoi(v)) : 32;
    }();
    int mi = 0;
    if (kBlk > 0 && kBlk < group_count && n_end > n_start) {
      const int ncols = n_end - n_start;
      static thread_local std::vector<float> park_storage;
      if (park_storage.size() < (size_t)ncols * 4 * 8) park_storage.resize((size_t)ncols * 4 * 8);
      float* park = park_storage.data();
      for (; mi + 4 <= m; mi += 4) {
        const float* p0 = a_perm + (size_t)(mi + 0) * k;
        const float* p1 = a_perm + (size_t)(mi + 1) * k;
        const float* p2 = a_perm + (size_t)(mi + 2) * k;
        const float* p3 = a_perm + (size_t)(mi + 3) * k;
        std::fill(park, park + (size_t)ncols * 4 * 8, 0.f);
        for (int gb = 0; gb < group_count; gb += kBlk) {
          const int ge = std::min(group_count, gb + kBlk);
          for (int ni = n_start; ni < n_end; ni++) {
            const uint8_t* b_row = b.b + (size_t)ni * row_bytes;
            const float* b_scales = b.d + (size_t)ni * group_count;
            float* pk = park + (size_t)(ni - n_start) * 32;
            __m256 tot0 = _mm256_loadu_ps(pk), tot1 = _mm256_loadu_ps(pk + 8);
            __m256 tot2 = _mm256_loadu_ps(pk + 16), tot3 = _mm256_loadu_ps(pk + 24);
            for (int g = gb; g < ge; g++) {
              const int base = g * 32;
              KT_MXFP4_DECODE_GROUP(b_row, g);
              __m256 g0 = _mm256_mul_ps(_mm256_loadu_ps(p0 + base), w0);
              __m256 g1 = _mm256_mul_ps(_mm256_loadu_ps(p1 + base), w0);
              __m256 g2 = _mm256_mul_ps(_mm256_loadu_ps(p2 + base), w0);
              __m256 g3 = _mm256_mul_ps(_mm256_loadu_ps(p3 + base), w0);
              g0 = _mm256_fmadd_ps(_mm256_loadu_ps(p0 + base + 8), w1, g0);
              g1 = _mm256_fmadd_ps(_mm256_loadu_ps(p1 + base + 8), w1, g1);
              g2 = _mm256_fmadd_ps(_mm256_loadu_ps(p2 + base + 8), w1, g2);
              g3 = _mm256_fmadd_ps(_mm256_loadu_ps(p3 + base + 8), w1, g3);
              g0 = _mm256_fmadd_ps(_mm256_loadu_ps(p0 + base + 16), w2, g0);
              g1 = _mm256_fmadd_ps(_mm256_loadu_ps(p1 + base + 16), w2, g1);
              g2 = _mm256_fmadd_ps(_mm256_loadu_ps(p2 + base + 16), w2, g2);
              g3 = _mm256_fmadd_ps(_mm256_loadu_ps(p3 + base + 16), w2, g3);
              g0 = _mm256_fmadd_ps(_mm256_loadu_ps(p0 + base + 24), w3, g0);
              g1 = _mm256_fmadd_ps(_mm256_loadu_ps(p1 + base + 24), w3, g1);
              g2 = _mm256_fmadd_ps(_mm256_loadu_ps(p2 + base + 24), w3, g2);
              g3 = _mm256_fmadd_ps(_mm256_loadu_ps(p3 + base + 24), w3, g3);
              const __m256 sv = _mm256_broadcast_ss(&b_scales[g]);
              tot0 = _mm256_fmadd_ps(g0, sv, tot0);
              tot1 = _mm256_fmadd_ps(g1, sv, tot1);
              tot2 = _mm256_fmadd_ps(g2, sv, tot2);
              tot3 = _mm256_fmadd_ps(g3, sv, tot3);
            }
            _mm256_storeu_ps(pk, tot0);
            _mm256_storeu_ps(pk + 8, tot1);
            _mm256_storeu_ps(pk + 16, tot2);
            _mm256_storeu_ps(pk + 24, tot3);
          }
        }
        for (int ni = n_start; ni < n_end; ni++) {
          const float* pk = park + (size_t)(ni - n_start) * 32;
          c.data[(size_t)(mi + 0) * n + ni] = hsum_avx2(_mm256_loadu_ps(pk));
          c.data[(size_t)(mi + 1) * n + ni] = hsum_avx2(_mm256_loadu_ps(pk + 8));
          c.data[(size_t)(mi + 2) * n + ni] = hsum_avx2(_mm256_loadu_ps(pk + 16));
          c.data[(size_t)(mi + 3) * n + ni] = hsum_avx2(_mm256_loadu_ps(pk + 24));
        }
      }
      // 0-3 remaining rows fall through to the single-row loop below, which
      // is unchanged (its per-parity accumulators already match).
    }
    if (kTile >= 8 && m >= 8 && mi == 0) {
'''
# replace "    int mi = 0;\n    if (kTile >= 8 && m >= 8) {\n" with the block (which declares mi)
s = s.replace(anchor, block, 1)
f.write_text(s, encoding="utf-8")
print("patched", f)
