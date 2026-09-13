# V4.1 decode: where the 80 ms per token go (2026-09-13, first pass)

Deferral 0 (deferral 4 breaks output and is not used for writer work) and one
running request; the socket-1 fan is in place. `tools/torch-profile-dsv41-decode.sh`
through SGLang's `/start_profile`, plus timing counters added to the engram
row store (`tools/dsv41-engram-adapter/row_store.cpp`, `DSV41_ENGRAM_TIMING=<s>`
logs them to `server.log`).

## Throughput

| run | decode |
|---|---|
| 300 tokens, graph on | 10.7-10.9 tok/s (server log), 27 s wall for 300 |
| 600 tokens, graph on, engram timing on | 12.1-12.3 tok/s |
| 120 tokens, `--disable-cuda-graph` | 3.5 tok/s — launch-bound, 245 ms of CPU-side op launches per step; useless for the split, kept as a data point |

So deferral 0 sits at 11-12 tok/s, not 15.8 (that figure was deferral 4).

## The step

With the graph on, the trace splits a step into `step[DECODE]` = 2.6 ms (the
graph launch and its bookkeeping) and `scheduler.process_batch_result` =
74 ms median, which is the `aten::item` sync waiting for the graph to finish.
Scheduler-side overhead is ~3 ms per token; nothing to win there. Everything
else is inside the graph, serialised on the stream: GPU attention and the 7
GPU experts, the 40 CPU-expert submit/sync host callbacks, and the two engram
row fetches.

13 of 301 steps took over 100 ms (max 275 ms) and cost 9% of the window.

## Engram: 8-9 ms per token, 75-94% cache misses, one NVMe read at a time

Two engram layers (1 and 14), one host callback each per token, 12 rows per
call. Each layer's table is 384M rows × 264 B ≈ 101 GB (50 GB per rank); the
direct-mapped cache is 2 GiB per layer per rank (7.9M slots, 2% of the rows),
so **74-94% of lookups miss** and every miss is two synchronous `O_DIRECT`
preads (weights, scales) at ~170 µs each, issued one after another inside the
callback:

```
[engram-timing layer 1]  calls=128 rows=1536 miss=1172 (76%) lookup=438 ms (3.4 ms/call) miss_io=401 ms
[engram-timing layer 14] calls=128 rows=1536 miss=1448 (94%) lookup=454 ms (3.5 ms/call) miss_io=444 ms
```

3.3-4.7 ms per call per layer → **~8-9 ms per token, 10-11% of the step, and
essentially all of it is NVMe latency in series.** A larger cache does not
help (the 64 GiB variant halved throughput through reclaim, and 200 GB of
tables do not fit beside 306 GB of experts). Issuing the 24 reads of a call
concurrently (a small thread pool or io_uring in `row_store_lookup`) would
bring a call from ~4 ms to ~0.4 ms: **+10% decode for a change confined to
the adapter.** The 100+ ms outliers are likely the same reads hitting a slow
NVMe moment; the same fix bounds them.

### Done: misses in parallel (01:15)

`row_store_lookup` now serves hits in place, collects the misses, runs them on
a per-store pool of I/O threads (`DSV41_ENGRAM_IO_THREADS`, default 16) and
fills the cache afterwards. `test_adapter_cpu.py` (now reading the shard via
`base_dir`) stays byte-exact against plain preads. Same 600-token decode:

| | before | after |
|---|---|---|
| lookup per call per layer | 3.3-4.7 ms | **0.83-1.05 ms** |
| decode | 12.1-12.3 tok/s | **13.1-13.2 tok/s** |
| step median / p90 / max | 69.6 / 77.5 / 119 ms | **64.1 / 65.2 / 98.8 ms** |

+8%, the tail gone. The remaining ~2 ms per token of engram is the 12
concurrent NVMe reads' latency (~0.7 ms) twice; the two layers' hash ids are
computed together before the forward, so layer 14's rows could be fetched
during layer 1's call — another ~1 ms, later.

## The kt-kernel clock: CPU experts are 28 ms and already at 82% of bandwidth

`KT_TASKQUEUE_TIMING=400` (task-queue timestamps added to kt-kernel,
`native-ubuntu/patches/kt-taskqueue-timing.patch`, rebuilt with
`native-ubuntu/build-kt-dsv41.sh`):

```
[kt-timing] 400 syncs: tasks=400 queue_wait=11 us/task run=700 us/task | sync_wait=660 us/sync
```

One task per layer: **700 µs × 40 = 28 ms per token of CPU expert compute**,
against the 23 ms the bytes demand at 226 GB/s — the MXFP4 GEMV runs at 82%
of the two-socket ceiling. Dispatch latency is 11 µs. `sync_wait` ≈ `run`:
the GPU submits and waits, nothing overlaps (decode is layer-serial). So the
CPU side has no hidden fixed cost; the rest of the step is the GPU.

## The GPU clock: the dense FP8 GEMMs were 38 of 70 ms

SGLang's `/start_profile` wants the activity named **"GPU"**, not "CUDA" —
with "CUDA" it silently records no kernels, which is what yesterday's "no CUDA
events on this stack" was. With kernels visible, one decode step (70 ms):

| | ms | |
|---|---|---|
| `_w8a8_block_fp8_matmul` | **38.0** | 197 calls, 193 µs each: the Triton block-FP8 GEMM for every dense projection, at M=1 |
| all other kernels | 10 | NCCL all-reduce 2.1 (81 calls), cuBLAS gemv 2.5, hc/mhc, rmsnorm, sparse MLA decode 0.5, GPU experts 0.8 |
| GPU idle | 22 | the CPU expert waits |

The GEMM had no tuned config for the RTX 5070 Ti, so it ran the default
64×128×128 tiling: at M=1 that is N/128 CTAs each streaming a K×128 strip
alone, 250-500 GB/s effective on a 900 GB/s card. Real shapes per layer:
N×K = 25600×6144, 16384×1280, 5120×4096, 4096×1280, 5120×1152, 2304×5120,
1792×5120 — ~231 MB per layer, 9.2 GB per token, 10 ms at the bandwidth
floor.

### Done: skinny split-K tiling for M ≤ 16 (01:54)

`native-ubuntu/patches/fp8-skinny-splitk.patch` on the worktree's
`fp8_kernel.py`: when no tuned config exists and M ≤ 16, run the split-K
kernel (`_w8a8_block_fp8_matmul_hopper`, plain Triton) with a 16×128×128 tile
and SPLIT_K=4. Standalone device time at M=1: N=4096 K=7168 63 → 16.5 µs,
N=2048 58 → 10, N=16384 176 → 145; numerics identical to the reference.
`SGLANG_FP8_SKINNY=0` restores the old path.

| | before | after |
|---|---|---|
| fp8 GEMM per step | 38.0 ms | **8.7 ms** (202 calls) |
| GPU busy per step | 48 ms | 18 ms |
| step span | 70 ms | **48 ms** |
| decode | 12.5-13.2 tok/s | **17.7-18.2 tok/s** |

One layer now reads: attention chain ~120 µs, the two shared-expert GEMMs on
the alt stream overlapping the router (41 µs), GPU experts 12 µs, **a 713 µs
gap waiting for the CPU**, then all-reduce and the post-MoE kernels ~130 µs.
The GEMMs sit at the bandwidth floor. What is left per layer on the GPU:
two NCCL all-reduces (47 µs; a custom all-reduce would be ~15), a 41 µs
cuBLAS bf16 gemv, and ~45 tiny kernels at launch cost.

## Where the token goes now (48-51 ms)

| | ms |
|---|---|
| CPU experts (bandwidth-bound) | 28 |
| GPU: dense GEMMs at the floor | 8.7 |
| GPU: all-reduces | 2.4 |
| GPU: everything else | 7 |
| engram | 2 |
| scheduler | 3 |

The CPU expert bytes are the wall: 58% of the token, and the kernel is at
82% of what the sockets can deliver.

Per token the CPU reads 7 × 35.4M × 40 ≈ 9.9 G MXFP4 parameters ≈ 5.3 GB;
at the 226 GB/s two-socket ceiling that is 23 ms. The other ~40 ms is not
visible from outside the graph. Threads 64 → 112 made no difference
yesterday, so it is not compute; the candidates are per-layer fixed cost —
40 × (D2H of the layer input, `cudaLaunchHostFunc` scheduling, CPUInfer
dispatch to two pools, both pools' completion, H2D of the output) — and the
GPU side (attention over the 1M-capable KV layout with the indexer, the
all-reduces: 83 `nccl:all_reduce` per step at ~65 µs each ≈ 5.5 ms). Both
need a clock inside: kt-kernel's CPUInfer with enqueue/start/end timestamps
per task (a rebuild of `kt_kernel_ext` from `source/ktransformers-gemma4/kt-kernel`
with a timing hook), and CUDA events around the attention block outside the
graph. That is the next measurement.

## Hot experts on the GPU's seven slots: 18 → 23.5 tok/s (02:42)

SGLang's expert distribution recorder never sees the V4 router's ids (its
dump was all zeros), so the KT wrapper now keeps its own [layers, experts]
histogram on the GPU (`KT_ROUTING_DUMP=<prefix>`, a captured `index_add_`,
saved every 30 s). On the four writer prompts (3128 tokens):

| experts per layer on the GPU | share of routed tokens caught | uniform |
|---|---|---|
| 7 | **45.5%** | 1.8% |
| 16 | 59% | 4.2% |
| 32 | 71% | 8.3% |
| 64 | 83% | 17% |

Median 11 experts per layer cover half the routing. The VRAM budget cannot
grow (`--kt-num-gpu-experts` is per layer: 7 already cost ~2.7 GB per rank),
but the seven slots were physical 0..6 = logical 0..6, i.e. random. The map
from `tools/build-dsv41-expert-placement.py --gpu 7` puts the top seven of
each layer there. Both weight loaders honour `--init-expert-location`; what
did not was the routing — the V4 router hands the KT wrapper logical ids and
the topk remap in `topk.py` runs only under `--enable-eplb` — so the first
run with the map was 18.3 tok/s with wrong weights under every token. The
wrapper now remaps logical → physical itself (`patches/kt-ep-wrapper.patch`).

| | before | hot-7 |
|---|---|---|
| CPU expert task | 700 µs/layer | **381 µs** |
| decode | 17.7-18.2 tok/s | **23.4-23.7 tok/s** |

Numerics: 45% of the routed expert compute now runs through the GPU path
(FlashInfer CUTLASS W4A8 with MXFP8 activations) instead of the CPU's MXFP4 ×
BF16. Greedy outputs diverge from the identity placement within 0-28
characters (the identity placement reproduces itself exactly). Scoring six
fixed texts under both servers: mean logprob unchanged (+0.04 nats per token
in hot-7's favour), per-token |Δ logp| median 0.07, mean 0.38, p90 ~1.0, max
7. Two 400-token prose samples at t=1.0 read fine, with one stray token
("ulin") and one dropped kanji — a rate that needs the 12-sample gate
against the baseline before it means anything (`logs/dsv41-hot7-prose-gate.txt`,
running 03:05). Adoption is a fidelity call for the user: +30% decode for a
measurable but so far not visibly harmful change in which kernel computes
the hot experts.

### The 12-sample gate (03:05-03:31) and the stray tokens

`tools/verify-dsv41-swa-long.sh` now takes `AB_EXTRA_ARGS` (the "on" side
launches with those flags, replay off on both sides). Base vs hot-7 at
t=1.0 / p=0.95, 2 reps: essay, prose, list all **[ok]** — repetition, comma
rate and length indistinguishable (base's prose stopped at 8 and 133 tokens,
hot-7's at 359 and 275; that prompt stops short on every run). What the
scorer does not see: **stray Latin tokens inside Japanese text — 0 in the 6
base samples, 1 in the 6 hot-7 samples ("3～ Examin（未確定）割以上") plus
the "ulin" in the earlier 400-token sample, 2 in 8.** Small numbers, but the
mechanism is real: on SM120 the FlashInfer CUTLASS MoE is W4A8 only, so the
45% of expert compute moved to the GPU runs on MXFP8 activations where the
CPU used BF16.

The W4A16 alternative, `--moe-runner-backend marlin`, loads (Marlin repacks
the MXFP4 experts) but faults with an illegal memory access in
`fused_marlin_moe`'s swiglu during CUDA graph capture under the KT wrapper's
masked ids — the same partial-expert Marlin path the V4-Flash stack needed
`patches/mxfp4-deepseek-marlin-partial.patch` for. Porting that patch to the
dsv4.1 worktree is the next step toward a hot-expert placement without the
precision change.

### Morning (08:40-09:00): the stray tokens are the model's, not the map's

- Marlin W4A16 runs once the KT wrapper stops handing it -1 ids
  (`KT_GPU_MASK_ZERO=1`: CPU-bound rows go to GPU expert 0 with a zero
  routing weight, ids as int64 so `fused_marlin_moe` takes its generic
  alignment). Decode 23.4 tok/s, CPU task 363 µs — and **the same |Δlogp|
  against base as W4A8 (mean 0.377, median 0.06)**. So the deviation is not
  activation precision; it is the GPU MXFP4 kernel versus the CPU AVX2
  MXFP4 kernel, now applied to 45% of the routing instead of 2%. Both are
  the same 4-bit approximation of the same weights.
- Stray Latin fragments, 12 samples × 500 tokens per side, same prompts,
  t=1.0 / p=0.95: **base 2 ("笑わないTags。", "伸ばしているacs。"), hot-7 1
  ("吸い込まれていくopy。")**. The artifact is the model's own — a Latin
  fragment glued before "。" — and the earlier 0-of-6 was small-sample luck.

**Standing: hot-7 shows no measurable degradation** — gate passes, mean
logprob unchanged, stray-token rate equal to base — for +30% decode. Whether
it becomes the launcher default is the user's call; the evidence no longer
argues against it. (The "…Tags。" artifact itself is a separate writer-side
issue worth a logit filter or a post-pass.)

## Levers, ranked by evidence

1. ~~Engram misses in parallel~~ — done, +8%.
2. ~~Dense FP8 GEMM tiling at M=1~~ — done, +40%.
3. ~~Hot experts on the GPU~~ — done, +30%, pending the prose gate and the
   user's fidelity call. More slots would catch more (16 → 59%) but need
   VRAM the cards do not have at 1M context.
4. All-reduces: re-test `--enable-custom-all-reduce` (SGLang's) — ~1.2 ms.
5. Engram: fetch layer 14's rows during layer 1's call — ~1 ms.
6. Hardware: an 8-CCD 7002 halves the 28 ms. The one lever left with a 2x
   in it, once the DIMM airflow is real.

## Prefill (09-13 morning): the CPU expert GEMM re-permuted its activations per slice

`tools/torch-profile-dsv41-prefill.sh` with GPU activities and the kt clock on a
737-token prompt: **12.6 s, of which GPU kernels 0.41 s and 12.2 s waiting for
the CPU experts** — 300 ms per layer. The GPU-side indexer is not the problem.
Hot-7 had already taken 30 s → 13-15 s (45% of the rows moved to the GPU).

Stage clock inside `forward_prefill` (`KT_PREFILL_STAGE_TIMING=1`,
`patches/kt-prefill-stage-timing.patch`), tp0, 770 tokens, one layer:
bookkeeping 0.2, copy 1.6, pack 5.2, **gate/up 159-207, act 3, pack 1,
down 93-161**, sum 2 ms. Only the GEMMs matter.

A standalone bench of `gemm_mxfp4` (single core, V4.1 shapes): 23 GMAC/s at
m ≥ 4 whether n is whole or split 4 ways — but **14 GMAC/s when n is split
36 ways, the server's task size**, and 11 GMAC/s with all 32 cores of a socket
busy. The fast path converts the whole m × k activation block to permuted
FP32 inside every call, i.e. once per 64-row weight slice, 36-80 times per
expert.

Fix (`patches/kt-mxfp4-aperm-once.patch`): `BufferA` carries the permuted
FP32 copy, filled once per expert in `from_mat` (the pack stage), and both
fast paths (group 32 and the NVFP4 group 16) read it. Same values, same FMA
order — greedy outputs are byte-identical across the change (3/3).

| | before | after |
|---|---|---|
| slice bench, m=16, n/36 | 14.6 GMAC/s | 24.3 |
| prefill 737 tokens (wall) | 15.5 s | **9.1 s** |
| prefill 2000 tokens, 2 chunks (wall) | 58.8 s | **27.0 s** (74 tok/s) |
| tp0 gate/up + down, 40 layers, 737 tok | 10.7 + 2.7 s | 4.8 + 2.3 s |
| decode | 23.4 tok/s | 24.0 |

The same code sits in upstream kvcache-ai/ktransformers via #2175 (AVX2
MXFP4 fast path) and #2176 (NVFP4 group 16) — both carry the per-call
conversion — and in the V4-Flash venvs on this host. A follow-up PR and a
rebuild of the V4-Flash extension are owed.

What is left in prefill on the CPU path: gate/up at 120 ms/layer against a
48 ms single-core-rate bound (all-core clocks, experts with 1-3 rows on the
single-row path, stage barriers); maybe 1.5x more. The order-of-magnitude
lever is still streaming expert weights to the GPU per expert for prefill
(handover item 7).

