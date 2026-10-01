"""Group-32 (MXFP4, DeepSeek V4) prefill GEMM: rows outer, output columns inner.

The same fix kt-mxfp4-g16-loop-order.patch made for GLM's NVFP4 path
(2.19x prefill, bit-identical, 09-21), which left group-32 with the old order.
gemm_mxfp4's fast path walked output columns in the outer loop and 4-token
blocks inside, so every output column re-read the block's four FP32 activation
rows (4 x k x 4 B = 64 KiB at k = 4096) from L2. With the rows outside they
stay in L1 across the thread's columns and the weights stream instead. Each
output element keeps its own k loop, accumulators and write, so the result is
bit-identical.

Apply to kt-kernel/operators/avx2/mxfp4-moe.hpp of source/ktransformers-gemma4
(venv-dsv41: V4-Flash-Vision, V4.1).
"""
import sys
from pathlib import Path

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "KTransformers/source/ktransformers-gemma4"
f = root / "kt-kernel/operators/avx2/mxfp4-moe.hpp"
s = f.read_text(encoding="utf-8")

MARK = "// Rows outer, output columns inner (group-32)."
if MARK in s:
    print("already applied")
    sys.exit(0)

old = '''    for (int ni = n_start; ni < n_end; ni++) {
      const uint8_t* b_row = b.b + (size_t)ni * row_bytes;
      const float* b_scales = b.d + (size_t)ni * group_count;
      if (ni + 1 < n_end) {
        const char* nr = (const char*)(b.b + (size_t)(ni + 1) * row_bytes);
        _mm_prefetch(nr, _MM_HINT_T0);
        _mm_prefetch(nr + 64, _MM_HINT_T0);
        _mm_prefetch(nr + 128, _MM_HINT_T0);
        _mm_prefetch(nr + 192, _MM_HINT_T0);
      }

      // 4-token blocked path: each decoded group feeds 4 accumulators.
      int mi = 0;
      for (; mi + 4 <= m; mi += 4) {
        const float* p0 = a_perm + (size_t)(mi + 0) * k;
        const float* p1 = a_perm + (size_t)(mi + 1) * k;
        const float* p2 = a_perm + (size_t)(mi + 2) * k;
        const float* p3 = a_perm + (size_t)(mi + 3) * k;
        __m256 tot0 = _mm256_setzero_ps(), tot1 = _mm256_setzero_ps();
        __m256 tot2 = _mm256_setzero_ps(), tot3 = _mm256_setzero_ps();

        for (int g = 0; g < group_count; g++) {'''

new = '''    // Rows outer, output columns inner (group-32). The four activation rows
    // (64 KiB at k = 4096) stay in L1 across every output column of this
    // thread instead of being re-streamed from L2 once per column; the
    // weights stream instead. Each output keeps its own untouched k loop and
    // is written exactly once, so the result is bit-identical.
#define KT_MXFP4_PREFETCH_NEXT_ROW(ni)                                         \\
  if ((ni) + 1 < n_end) {                                                      \\
    const char* nr = (const char*)(b.b + (size_t)((ni) + 1) * row_bytes);      \\
    _mm_prefetch(nr, _MM_HINT_T0);                                             \\
    _mm_prefetch(nr + 64, _MM_HINT_T0);                                        \\
    _mm_prefetch(nr + 128, _MM_HINT_T0);                                       \\
    _mm_prefetch(nr + 192, _MM_HINT_T0);                                       \\
  }
    int mi = 0;
    for (; mi + 4 <= m; mi += 4) {
      const float* p0 = a_perm + (size_t)(mi + 0) * k;
      const float* p1 = a_perm + (size_t)(mi + 1) * k;
      const float* p2 = a_perm + (size_t)(mi + 2) * k;
      const float* p3 = a_perm + (size_t)(mi + 3) * k;
      for (int ni = n_start; ni < n_end; ni++) {
        const uint8_t* b_row = b.b + (size_t)ni * row_bytes;
        const float* b_scales = b.d + (size_t)ni * group_count;
        KT_MXFP4_PREFETCH_NEXT_ROW(ni)
        __m256 tot0 = _mm256_setzero_ps(), tot1 = _mm256_setzero_ps();
        __m256 tot2 = _mm256_setzero_ps(), tot3 = _mm256_setzero_ps();

        for (int g = 0; g < group_count; g++) {'''

if s.count(old) != 1:
    raise SystemExit("head anchor found %d times" % s.count(old))
s = s.replace(old, new)

old_tail = '''        c.data[(size_t)(mi + 0) * n + ni] = hsum_avx2(tot0);
        c.data[(size_t)(mi + 1) * n + ni] = hsum_avx2(tot1);
        c.data[(size_t)(mi + 2) * n + ni] = hsum_avx2(tot2);
        c.data[(size_t)(mi + 3) * n + ni] = hsum_avx2(tot3);
      }

      // Single-row remainder (also the whole decode path when m == 1).
      for (; mi < m; mi++) {
        const float* ap_row = a_perm + (size_t)mi * k;
        __m256 total0 = _mm256_setzero_ps();
        __m256 total1 = _mm256_setzero_ps();
        for (int g = 0; g < group_count; g++) {
          const float* ap = ap_row + g * 32;
          KT_MXFP4_DECODE_GROUP(b_row, g);

          __m256 gacc = _mm256_mul_ps(_mm256_loadu_ps(ap), w0);
          gacc = _mm256_fmadd_ps(_mm256_loadu_ps(ap + 8), w1, gacc);
          gacc = _mm256_fmadd_ps(_mm256_loadu_ps(ap + 16), w2, gacc);
          gacc = _mm256_fmadd_ps(_mm256_loadu_ps(ap + 24), w3, gacc);

          const __m256 sv = _mm256_broadcast_ss(&b_scales[g]);
          if (g & 1)
            total1 = _mm256_fmadd_ps(gacc, sv, total1);
          else
            total0 = _mm256_fmadd_ps(gacc, sv, total0);
        }
        c.data[(size_t)mi * n + ni] = hsum_avx2(_mm256_add_ps(total0, total1));
      }
    }
#undef KT_MXFP4_DECODE_GROUP
    return;
  }
'''
new_tail = '''        c.data[(size_t)(mi + 0) * n + ni] = hsum_avx2(tot0);
        c.data[(size_t)(mi + 1) * n + ni] = hsum_avx2(tot1);
        c.data[(size_t)(mi + 2) * n + ni] = hsum_avx2(tot2);
        c.data[(size_t)(mi + 3) * n + ni] = hsum_avx2(tot3);
      }
    }

    // Single-row remainder (also the whole decode path when m == 1).
    for (; mi < m; mi++) {
      const float* ap_row = a_perm + (size_t)mi * k;
      for (int ni = n_start; ni < n_end; ni++) {
        const uint8_t* b_row = b.b + (size_t)ni * row_bytes;
        const float* b_scales = b.d + (size_t)ni * group_count;
        KT_MXFP4_PREFETCH_NEXT_ROW(ni)
        __m256 total0 = _mm256_setzero_ps();
        __m256 total1 = _mm256_setzero_ps();
        for (int g = 0; g < group_count; g++) {
          const float* ap = ap_row + g * 32;
          KT_MXFP4_DECODE_GROUP(b_row, g);

          __m256 gacc = _mm256_mul_ps(_mm256_loadu_ps(ap), w0);
          gacc = _mm256_fmadd_ps(_mm256_loadu_ps(ap + 8), w1, gacc);
          gacc = _mm256_fmadd_ps(_mm256_loadu_ps(ap + 16), w2, gacc);
          gacc = _mm256_fmadd_ps(_mm256_loadu_ps(ap + 24), w3, gacc);

          const __m256 sv = _mm256_broadcast_ss(&b_scales[g]);
          if (g & 1)
            total1 = _mm256_fmadd_ps(gacc, sv, total1);
          else
            total0 = _mm256_fmadd_ps(gacc, sv, total0);
        }
        c.data[(size_t)mi * n + ni] = hsum_avx2(_mm256_add_ps(total0, total1));
      }
    }
#undef KT_MXFP4_PREFETCH_NEXT_ROW
#undef KT_MXFP4_DECODE_GROUP
    return;
  }
'''
if s.count(old_tail) != 1:
    raise SystemExit("tail anchor found %d times" % s.count(old_tail))
s = s.replace(old_tail, new_tail)
f.write_text(s, encoding="utf-8")
print("patched", f)
