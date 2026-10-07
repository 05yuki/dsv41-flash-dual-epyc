# DeepSeek V4.1-Flash on a dual EPYC 7452 + 2 × RTX 5070 Ti, 1M context

Serving DeepSeek's 552 B-backbone MoE (plus 196 B of Engram conditional memory;
8 B parameters active per token in prefill, 16 B in decode, per the model card)
from a used dual-socket EPYC Rome board and two 16 GB gaming cards: 384 routed
experts in MXFP4 on 512 GB of host DRAM through
[KTransformers'](https://github.com/kvcache-ai/ktransformers) kt-kernel,
attention and the five hottest experts per layer on the GPUs (Marlin) through
[SGLang's `dsv4.1` branch](https://github.com/sgl-project/sglang), the 200 GB
of engram tables read straight from the checkpoint shards on NVMe. Every routed
expert computed (no deferral, since 09-23), one request at a time, 1M-token
context, the writer's sampling (t=1.0, top_p 0.95) unless noted.

This repository is the patches, launchers, probes and write-ups from getting
it from *works* to *usable*. Nothing here is a fork; every change is a small
patch against the two upstream trees, and the measurements say what each one
bought.

## Where it stands (10-07)

| | 09-11 | 09-15 | 10-01 | 10-07, prefill lend |
|---|---|---|---|---|
| decode, single stream, Japanese prose | 11-12 tok/s | 34 tok/s ¹ | 28-30 tok/s (27.9-28.4 on 09-26, 30.6 on 10-01) | **27.6-28.0 tok/s** (25.4-25.6 on 10-06 without the lend) ² |
| decode, English synthetic, greedy, depth 0 → 128k | — | — | 23.5 → 22.9 tok/s | — |
| prefill, ~6K tokens | — | 6,945 tokens in 36.5 s | 6,050 tokens in 23.4 s (258 tok/s) | 6,050 tokens in **10.0 s** (607 tok/s; the first prompt after a launch takes 17.6 s while the engram cache is cold) |
| prefill, ~38-48K tokens | ~11 min on the CPU | — | 38,072 tokens in 148.5 s (256 tok/s); 48,345 in 198 s | 38,071 tokens in **66.1 s** (576 tok/s) |
| prefill, ~114K tokens | — | — | 114,166 tokens in 441 s (259 tok/s, 10-05) | 114,162 tokens in **191.6 s** (596 tok/s) |
| prefill at depth 0 / 29k / 64k / 128k | — | — | 242 / 249 / 185 / 211 tok/s | — |
| prefill chunk | 2048 | 2048 | 2048 | 6144, the widest that keeps the KV pool; calibrated with the lend probe up to 1M |
| startup (476 GB loaded and pinned) | — | 60+ min | 11 min | 8-11 min; a setup's first launch adds one calibration launch (the probe itself takes 2 min) |
| context | 1M | 1M | 1M (the ladder ran at a 262K pool with radix on) | 1M, KV pool 1,048,576 tokens; a 989,154-token prompt prefills in **2,433.9 s** (406 tok/s, the first prefill after a launch) |

¹ With four of the six routed experts per token deferred (the launcher's
default until 09-23), so it is not comparable with the every-expert numbers
after that. Deferral changes the arithmetic, and on V4-Flash-Vision it broke
UTF-8 in Japanese output, which is why it is off now.

² 512 tokens at t=1.0 right after a 114K prefill; the 10-01 figure is held-out
prose. Compare within a column, not across.

The depth ladder follows [llama-split-bench](https://github.com/jimoto-no-llm/bench-of-us)'s
`measure_ladder.py` / `measure_pp0.py`, ported to SGLang's `/generate`
(`tools/sglang-ladder.py`); the full report is
[bench-of-us #17](https://github.com/jimoto-no-llm/bench-of-us/pull/17). The
ladder's decode is lower than the prose figure because the five GPU experts
per layer were chosen from Japanese writing traffic and catch less of
synthetic English (not verified).

Numerics: every kernel change below was checked byte-identical or by paired
per-token NLL. A caution learned on 10-01: V4.1's mean NLL moves by about 0.03
between launches of the same configuration even though no setting changes, so
compare configurations over four or more launches each. FlashInfer's autotune,
which re-picks kernel tactics per launch and was the cause on 09-11, is off in
the launcher (`--disable-flashinfer-autotune`); with it off, two launches then
reproduced greedy output byte for byte. The newer spread comes from something
added after 09-11 and has not been traced yet.

## Hardware

- 2 × AMD EPYC 7452 (Rome, 32 cores, **4 CCDs each** — the memory ceiling is
  the CCD count, not the DIMMs: ~113 GB/s per socket, 226 GB/s total, see
  `docs/CCD-BANDWIDTH-20260910.md`)
- 16 × 32 GB DDR4-2666 (512 GB), HUANANZHI H12D-16D
- 2 × RTX 5070 Ti 16 GB (SM120), PCIe 4.0 x16, no NVLink, no P2P
- one NVMe for the checkpoint (the MXFP4 experts are loaded to DRAM; the
  engram tables stay on disk)
- a fan directly over socket 1 (see finding 1)

## What was wrong, in the order it was found

1. **Socket 1's memory path throttles when the case runs hot.** After minutes
   of sustained inference on socket 1's cores, its local DRAM bandwidth halves
   (99 → 49 GB/s by a user-space probe with the server idle), latency rises
   11%, core clocks stay put, and it clears after 60 s idle. It tracks the
   BMC's inlet temperature: above about 47 °C the throttle sets in. BIOS
   APBDIS / fixed SoC P-state did nothing. A fan blowing straight onto
   socket 1 keeps fourteen back-to-back 700-token prefills flat at 31-32 s.
   `docs/DSV41-PREFILL-NODE1-BANDWIDTH-20260912.md`, `tools/membw-probe.c`,
   `tools/probe-node1-ikllama.sh`, `tools/watch-bmc-sensors.sh`.
2. **Engram row fetches were serial NVMe reads.** 12 rows per call, 75-94%
   cache misses, each miss two `O_DIRECT` preads in sequence: 8-9 ms of every
   decode token. An I/O pool in the callback: 4 ms → 1 ms per call, +8%.
   `tools/engram-adapter/`.
3. **The dense FP8 GEMMs had no tuned tiling for this card.** At M=1 the
   Triton block-FP8 GEMM launched N/128 CTAs and ran at 250-500 GB/s on a
   900 GB/s card: 38 of the 70 ms of a decode step. A split-K 16-row tiling
   for M ≤ 16: 8.7 ms, +40%. `patches/fp8-skinny-splitk.patch`.
4. **The GPU expert slots held logical experts 0..N-1.** Recorded routing is
   skewed, so placing each layer's most-used experts in the GPU slots takes
   that share of the bytes off the CPU. SGLang's expert-distribution recorder
   never sees the V4 router's ids, so the KT wrapper records its own histogram
   and remaps. Rebuilt on 09-24 from V4.1's own routing: five slots now catch
   47.9% of routed tokens (the 09-13 map caught 41.8%), seven 54.6%, ten
   60.9%. Ten decode at 31.3 tok/s but shrink the KV pool to 559K, so five is
   the default. `patches/kt-ep-wrapper.patch`, `tools/record-dsv41-routing.sh`,
   `tools/build-dsv41-expert-placement.py`, `dsv41-expert-placement-{5,7,8,10}.json`.
5. **The CPU MXFP4 GEMM re-permuted its activations per weight slice.** The
   AVX2 fast path converted the m × k activation block to permuted FP32 once
   per 64-row weight slice, 36-80 times per expert: 40% of a prefill task.
   `BufferA` now carries the permuted copy, filled once per expert: prefill
   2.1×, output byte-identical. `patches/kt-mxfp4-aperm-once.patch`. Upstream
   as kvcache-ai/ktransformers
   [#2205](https://github.com/kvcache-ai/ktransformers/pull/2205) (merged 09-14).
6. **Prefill was the CPU expert GEMM, so prefill now streams the experts to
   the GPU instead.** Above a token threshold the CPU-resident experts are not
   computed on the CPU at all: kt-kernel keeps each TP part's expert weights
   and scales in memfd arenas, each GPU rank maps the part it needs, registers
   it as pinned, and DMAs a group of experts at a time straight from the
   resident copy into a device slot; the conversion and the MoE GEMM run as
   one CUDA graph per group. Two things that ate a day on the way: every CUDA
   launch costs ~250 µs of host time while the link is saturated by our own
   DMA (hence the graphs), and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments`
   plus FlashInfer's in-capture workspace allocation corrupts every prompt past
   the SWA window. `patches/kt-stream-prefill.patch`,
   `kt-mxfp4-arena-and-upfirst-combined.patch`,
   `docs/DSV41-GPU-STREAMED-PREFILL-20260913.md`.
7. **Finding 5 cost decode, and nobody looked (09-15).** The pre-permuted copy
   is written and read on every call, and for the few rows of a decode step it
   lives in a cold region while the gemm's own scratch was L1-hot. Gated to
   inputs of 16 rows or more: decode 24 → 34 tok/s with four experts
   deferred, prefill keeps the 2.1×. `patches/kt-mxfp4-aperm-decode-gate.patch`. Upstream as
   [#2209](https://github.com/kvcache-ai/ktransformers/pull/2209) (merged 09-17).
8. **The CUTLASS MXFP8 × MXFP4 kernel shortchanged outlier tokens (09-16).**
   On the hot GPU experts it returned 6-12% too small a result for tokens with
   outlying activations. The hot experts moved to Marlin (rel. error 0.39%),
   and the streamed prefill computes in bf16 (`KT_GPU_STREAM_W16=1`, threshold
   512 tokens, groups of 4). `docs/DSV41-MARLIN-W16-20260916.md`.
9. **Arenas pinned at load, SWA prefix tails (09-16).** Pinning moved from the
   first request to startup (`KT_GPU_STREAM_EARLY_INIT`), and
   `--swa-prefix-tails 6` cuts a 48K prompt from 37 to 24 chunks: 283 → 198 s.
   The W16 path needs `KT_EXPERT_SHM`, `KT_GPU_STREAM_ZEROCOPY` and the graph
   path exported together, or the scheduler aborts at startup (09-18).
10. **No peer access, so all-reduce goes through host memory (09-22, 09-24).**
    Prefill-sized all-reduces are staged through pinned host memory on the
    copy engines (+3.2% on a 4K prefill); decode-sized ones are one kernel
    that both GPUs run on mapped host memory, 22-25 µs → 5 µs per call. On
    09-26 a race was found and fixed: for messages of 1 MiB and up the
    in-place add could start before this rank's own copy out had finished, so
    long prefills between 09-22 and 09-26 could occasionally come out wrong.
    `patches/sglang-host-staged-allreduce-race.py`, `patches/sglang-ll-allreduce.py`;
    upstream as [sgl-project/sglang#39605](https://github.com/sgl-project/sglang/pull/39605) (open).
11. **kt-kernel handed results to the GPU through host-function callbacks
    (09-24).** Replaced by `cuStreamWriteValue32` / `cuStreamWaitValue32` flags
    on the GPU stream: decode 25.4-26.1 → 27.0 tok/s, NLL unchanged;
    with the decode all-reduce above, 29.4. `patches/kt-stream-memops.py`.
12. **Startup took over an hour because the arenas sat on 4 KB pages (09-24).**
    The host's shmem THP had been `never`, so kt-kernel's `MADV_HUGEPAGE` did
    nothing; at `advise`, the late layers still fell back to 4 KB because the
    checkpoint's page cache had eaten every free 2 MB block. Dropping the page
    cache before each layer's experts load puts 328 of 337 GB on huge pages:
    startup 60+ min → 11 min. The load barrier timeout went 480 → 1800 s for
    the slow cases. `patches/sglang-kt-drop-cache-per-layer.py`,
    `patches/sglang-load-barrier-timeout.py`.
13. **A 48K prompt ran out of VRAM in the indexer (09-24).** The sparse
    indexer's candidate selection grew with prompt length. It now scores the
    query 512 rows at a time and drops the padding (426 checks identical), and
    a 51K prefill peaks at 15.3 GB per card, flat. Five GPU slots and
    `--mem-fraction-static 0.81` leave a 1,121,536-token pool, capped at 1M.
    `patches/sglang-dsv4-indexer-query-step.py`,
    `patches/sglang-dsv4-candidate-blocks-nopad.py`.
14. **The streamed prefill is PCIe-bound, which the 09-16 notes missed
    (09-25).** One layer moves 4.19 GB per rank in 185 ms, 22.7 GB/s, against
    25-26 GB/s for any pinned host-to-device copy on these cards. The "10.4
    GB/s" written earlier miscounted the bytes.
15. **The CPU group-32 kernel, prefill side (09-25).** Loop order and
    k-blocking: 1.33×, byte-identical. `patches/kt-mxfp4-g32-loop-order.py`,
    `patches/kt-mxfp4-g32-kblock.py`, and the decode shuffle without
    `vinserti128`, `patches/kt-mxfp4-decode-noinsert.py`.

16. **Lending the GPU weights to the prefill (10-06, 10-07).** A
    streamed-prefill forward costs about 7.5-9 s whatever the chunk, so 3x
    the chunk is close to 3x fewer forwards, but at chunk 2048 the 1M pool,
    the hot experts and the streamer's buffers left no room for more. Now,
    during a streamed prefill, the decoder weights give up their VRAM
    (torch_memory_saver keeps the addresses, so the decode CUDA graphs stay
    valid) and come back one layer at a time from a pinned host copy; the
    streamer's buffers exist only then. How it works and the other models it
    runs on: [kt-prefill-lend](https://github.com/05yuki/kt-prefill-lend).
    - Chunk: 6144, the widest that keeps the KV pool whole. A wider chunk
      widens the SWA cap, and here each SWA slot costs 14.33 full ones: at
      10240 the full pool fell from 1,048,576 to 896,000 (114K prompts 790
      tok/s). The lend takes nothing from the KV pool by default; the
      launcher's `DSV41_LEND_MAX_CHUNK` and kt-lend-auto's KV guard hold it
      there.
    - The SWA cap counts two chunks in flight, but the overlap scheduler
      still holds the previous chunk's slots while it builds the next, so
      every other batch came in at 3072 of 6144 rows. The launcher reads the
      last run's pool budget (1,090,304 at 6 tails for a 1,048,576 cap) and
      adds prefix tails as far as that slack pays: 13 tails, every batch
      5888-6144 rows, the pool still 1,048,576.
    - Calibration: the first launch of a setup runs the lend probe (finding
      17) at prefix 0, 524K and the context's end; it hinted 9216 / 8192 /
      6144 and kept 6144.
    - Quality: paired per-token NLL against chunk 2048 without the lend, 2,477
      tokens, two lend launches at chunk 4096 (not yet at 6144): delta -0.0153
      (z -1.42) and -0.0213 (z -1.90). Only the prose pieces are long enough
      to go through the lend.
    - The engram stays resident. Back-to-back 38K prefills at chunk 4096
      alternated 87 / 125 s with the inlet at 51.5 °C: finding 1, not the lend.
    - Files: `patches/kt_lend.py`, `patches/sglang-dsv41-kt-prefill-lend.py`,
      `patches/sglang-kt-stream-shared-graph-inputs.py`,
      `patches/kt-kernel-shared-gpu-output.py`,
      `patches/kt-kernel-lend-gpu-output.py`, `tools/kt-lend-auto.sh`.
17. **A 1M-token prompt, all the way (10-06, 10-07).** With the lend at
    chunk 6144 a 261K prompt left 5 MiB at its peak and a ~512K one ran out
    of memory. In order of the fixes:
    - The two-level indexer's candidate mask was a [rows, prefix] bool table
      per request (3.1 GB at 512K), built slice by slice and joined with
      `torch.cat` (two copies at once). It is now written in place and kept
      per block, not per position, and the consumer layers sink non-candidate
      blocks through a [rows, blocks, block_size] view of their scores
      (`sglang-dsv4-candidate-mask-inplace.py`,
      `sglang-dsv4-candidate-mask-blocks.py`).
    - A slice's fp32 logits were 512 rows x prefix (708 MiB at ~356K); the
      rows of a slice now shrink so they stay within 256 MiB
      (`sglang-dsv4-indexer-slice-budget.py`).
    - Checked with `SGLANG_DSV41_INDEXER_QUERY_CHECK=1` (sliced against
      whole-chunk on the same inputs): 288 layer checks, source, consumer
      and prefix-tail layers, all "rows differing 0, masks same True".
    - The prompt's last chunk, below the streaming threshold, ran with the
      weights and KV pool back and could not gather every compressed K
      (484 MiB at 989K). An extend over 65536 tokens or more now stays lent.
    - The engram adapter kept a staging set per power-of-two lookup size, so
      prompts of new lengths piled up sets (170 MiB after a 38K prompt);
      prefill-sized lookups now share one set, in the lend scratch.
    - The streamer kept a CUDA graph per row count, and a prompt's last chunk
      brings a new one: 12-14 MiB outside torch per prompt of a new length.
      It now keeps the largest and the 8 most recently used
      (`sglang-kt-stream-graph-keep.py`).
    - Result: 510K prompts peak at 3,270 MiB (38K: 3,292) and a 989,154-token
      prompt prefills to the end in 2,433.9 s (406 tok/s, the server's first
      prefill), KV pool 1,048,576.
    - The probe (`kt_lend_probe.py`, `sglang-kt-lend-probe.py`): a few
      extend forwards over a prefix whose KV is allocated but never computed
      give the same peak a real chunked prefill reaching it would, in seconds
      instead of 40 minutes (991K, 6144 rows: 5,286 MiB, against 6,272 for a
      real 989K prefill). Of that, 2,921 MiB is allocated; the rest is pieces
      in torch's cache from buffers that grow chunk by chunk. Emptying the
      cache before each chunk past 64K removes most of it between forwards;
      `expandable_segments`, which would remove the rest, is refused by
      torch_memory_saver.

Also in `docs/DSV41-DECODE-PROFILE-20260913.md`: the per-token budget as of
09-13, and the profiler gotcha — SGLang's `/start_profile` wants the activity
named `"GPU"`; `"CUDA"` is silently ignored and records no kernels.

## Layout

- `patches/` — against `sgl-project/sglang` branch `dsv4.1` (commit
  1aa0e962) and `kvcache-ai/ktransformers` (commit 95009ea):
  - `sglang-dsv41-tree-20261007.patch` — the whole SGLang tree as running on
    10-07, prefill lend, probe and the indexer fixes of finding 17 included
    (supersedes `sglang-dsv41-tree-20260916.patch`, the SGLang `.py` patchers
    and the SGLang parts of the older `.patch` files; it also carries the
    V4-Flash-Vision path and a few switches that are off by default)
  - `kt-kernel-tree-20261006.patch` — the whole kt-kernel tree as running on
    10-06 (supersedes every kt-kernel `.patch` and `.py` below; they stay as
    the record of what each change was)
  - `sglang-dsv41-tree-20260916.patch` and the `.py` patchers — the same
    changes one at a time, each with its write-up in its docstring; they
    apply on top of the 09-16 tree patch in the order listed under
    Reproducing, and a second run changes nothing
  - `fp8-skinny-splitk.patch`, `kt-ep-wrapper.patch`, `kt-stream-prefill.patch`,
    `kt-mxfp4-arena-and-upfirst-combined.patch` (`KT_EXPERT_SHM=1` memfd
    expert arenas, and `KT_WRITE_UP_FIRST=1`; `kt-mxfp4-writer-upfirst.patch`
    is that part alone), `mxfp4-avx2-gemv.patch` (m == 1 GEMV, upstream
    #2175), `kt-taskqueue-timing.patch`, `kt-prefill-stage-timing.patch`
  - `kt-mxfp4-aperm-once.patch`, `kt-mxfp4-aperm-decode-gate.patch` — upstream
    since #2205 / #2209
  - none of the kt-kernel `.patch` files applies to kvcache-ai's current main
    (its #2175 rewrote the MXFP4 fast path they edit); use the tree patch on
    95009ea
- `tools/start-dsv41-engram-nvme.sh` — the launcher (all knobs are env vars;
  set `KT_GPU0_UUID` / `KT_GPU1_UUID`); with `tools/kt-lend-auto.sh` it
  calibrates the prefill chunk with the probe on a setup's first launch and
  keeps the KV pool whole (`KT_PREFILL_LEND=0` turns the lend off,
  `SGLANG_CHUNKED_PREFILL_SIZE` overrides the chunk, `DSV41_LEND_MAX_CHUNK`
  the ceiling, `SGLANG_SWA_PREFIX_TAILS` the tails)
- `tools/engram-adapter/` — engram rows from the checkpoint shards over NVMe
  with a bounded RAM cache and an I/O pool (derived from 0xSero's adapter)
- `tools/build-kt-dsv41.sh` — rebuild `kt_kernel_ext` into the venv
  (`KTRANSFORMERS_ROOT`, default `~/KTransformers`)
- `tools/sglang-ladder.py` — the depth ladder and depth-0 prefill against
  SGLang's `/generate`
- `tools/measure-dsv41-stream-prefill.sh` — streamed-prefill A/B; the prompt
  ends in knowledge questions because a model whose streamed experts arrived
  as zeros still copies the paragraph back as a "summary"
- `tools/memguard-dsv41.sh` / `restart-memguard.sh`, `measure-dsv41-prefill-order.sh`,
  `torch-profile-dsv41-*.sh`, `record-dsv41-routing.sh`,
  `build-dsv41-expert-placement.py`, `verify-dsv41-swa-long.sh` — the harnesses
- `tools/membw-probe.c`, `memlat-probe.c`, `clock-probe.c`, `watch-dsv41-*.sh`,
  `probe-dsv41-tripped.sh`, `probe-node1-ikllama.sh`, `watch-bmc-sensors.sh`
  — how the socket-1 throttle was cornered without root
- `dsv41-expert-placement-{5,7,8,10}.json` — hot-expert maps recorded on
  Japanese prose (09-24); record your own for your workload
- `docs/` — the write-ups, dated

## Reproducing

1. SGLang `dsv4.1` at 1aa0e962 with `patches/sglang-dsv41-tree-20261007.patch`
   (`git apply`). To build it up one change at a time instead: the 09-16 tree
   patch, then `sglang-host-staged-allreduce-race.py`, `sglang-ll-allreduce.py`,
   `sglang-kt-drop-cache-per-layer.py`, `sglang-load-barrier-timeout.py`,
   `sglang-dsv4-indexer-query-step.py`, `sglang-dsv4-candidate-blocks-nopad.py`,
   `sglang-kt-stream-shared-graph-inputs.py`, `sglang-dsv41-kt-prefill-lend.py`,
   `sglang-dsv4-swa-pool-floor.py`, `sglang-dsv4-candidate-mask-inplace.py`,
   `sglang-dsv4-candidate-mask-blocks.py`, `sglang-dsv4-indexer-slice-budget.py`,
   `sglang-kt-stream-graph-keep.py`, `sglang-kt-lend-probe.py`, each with the
   tree as its argument. That gives the same files V4.1 runs; the 10-07
   patch adds the Vision path.
2. kvcache-ai/ktransformers at 95009ea with
   `patches/kt-kernel-tree-20261006.patch` (`git apply`; the submodules
   `third_party/llama.cpp` and `third_party/pybind11` at that commit's
   pins), built by `tools/build-kt-dsv41.sh`. The Python side of the
   installed `kt_kernel` package must match the tree's `kt-kernel/python/`.
   `pip install torch_memory_saver` (0.0.9.post1) in the same venv.
3. `vm.min_free_kbytes = 8388608` (a node-bound OOM from the CUDA JIT
   otherwise kills the scheduler while the experts sit in DRAM).
4. Engram manifest per `docs/DSV41-ENGRAM-MXFP4.md`; `DSV41_ENGRAM_DIR`,
   `DSV41_ENGRAM_BASE`.
5. As root, made persistent (e.g. through tmpfiles.d):
   `echo advise > /sys/kernel/mm/transparent_hugepage/shmem_enabled`,
   `echo defer > /sys/kernel/mm/transparent_hugepage/defrag`, and swap off.
   Without the first, pinning 4 KB shmem pages takes most of an hour; the
   launcher warns when it is `never`.
6. `KT_GPU0_UUID=... KT_GPU1_UUID=... tools/start-dsv41-engram-nvme.sh`. The
   streamed prefill, the Marlin hot experts, memops, both all-reduce paths and
   the prefill lend are on by default. The first launch of a setup starts the
   server once more for calibration (the lend probe at prefix 0, half the
   context and its end, about two minutes once loaded) and keeps the chunk it
   measured in `~/.cache/kt-lend/`.
7. Keep air moving over socket 1. Really.

## Not done

- The kt-kernel binary behind these figures was built on 09-25. The tree
  patch is the 10-06 source; what it adds since (one-byte scale codes, the
  experimental group-32 paths, load-all) is off unless its switch is set. A
  binary rebuilt from it ran the 10-06 checks with the same fingerprint
  spread (top-1 unchanged) and decode 27.3-28.1 tok/s.
- A chunk wider than 6144 costs KV pool (10240: 896,000 of 1,048,576, 114K
  prompts 790 instead of about 600 tok/s); left as a choice.
- The first prompt after a launch is two to four times slower: the engram
  rows it needs come from NVMe until the adapter's cache warms (the first
  38K prompt took 253-599 s, the next 66 s).
- Inside one forward at a long prefix, torch's cache still holds pieces
  (991K, 6144 rows: 5,286 MiB reserved for 2,921 allocated).
- Speculative decoding pays badly while the CPU expert path costs per row.
- One-byte group-scale codes for the CPU experts (0.53 instead of 0.625 byte
  per weight) make the kernel 7% faster at V4.1's shape, byte-identical, but
  no gain shows through V4.1's run-to-run spread; not enabled here.
- The ~1 in 12 samples with a stray Latin fragment before "。" is the model's
  own at t=1.0 and needs a writer-side filter.
