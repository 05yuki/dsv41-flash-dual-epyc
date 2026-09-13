# DeepSeek V4.1-Flash on a dual EPYC 7452 + 2 × RTX 5070 Ti, 1M context

Serving DeepSeek's 552 B-backbone MoE (plus 196 B of Engram conditional memory; 8 B parameters active per token in prefill, 16 B in decode, per the model card) from a used dual-socket EPYC Rome board and
two 16 GB gaming cards: 384 routed experts in MXFP4 on 512 GB of host DRAM
through [KTransformers'](https://github.com/kvcache-ai/ktransformers)
kt-kernel, attention and the seven hottest experts per layer on the GPUs
through [SGLang's `dsv4.1` branch](https://github.com/sgl-project/sglang), the
200 GB of engram tables read straight from the checkpoint shards on NVMe.
Deferral off (every routed expert computed), one request at a time, the
writer's sampling (t=1.0, top_p 0.95).

This repository is the patches, launchers, probes and write-ups from getting
it from *works* to *usable*. Nothing here is a fork; every change is a small
patch against the two upstream trees, and the measurements say what each one
bought.

| | before (09-11) | after (09-13) | streamed prefill (09-14) |
|---|---|---|---|
| decode, single stream | 11-12 tok/s | **24 tok/s** | 24 tok/s |
| 3564-token prompt + 96 decoded tokens, wall | — | 45 s | **17.3 s** |
| 6945-token prompt + 96 decoded tokens, wall | — | 85-92 s | **36.5 s** |
| prefill after ~5 min of continuous load | 2× slower, indefinitely | flat | flat (the CPU is idle during prefill) |

(Measured with SWA Bounded Replay off, which is the launcher's default again:
with it on, the third request of a session comes back as garbage on the CPU
path too — an issue in the dsv4.1 tree's replay, not in anything here. With
replay on and only the 21 non-SWA layers streaming, the 3564-token prompt
took 12.8 s, but a session cannot rely on it.)
| numerics | — | greedy output byte-identical across the kernel changes; the hot-expert map is the one change that moves logits (mean logprob unchanged, per-token median |Δ| 0.06 nats) |

## Hardware

- 2 × AMD EPYC 7452 (Rome, 32 cores, **4 CCDs each** — the memory ceiling is
  the CCD count, not the DIMMs: ~113 GB/s per socket, 226 GB/s total, see
  `docs/CCD-BANDWIDTH-20260910.md`)
- 16 × 32 GB DDR4-2666 (512 GB), HUANANZHI H12D-16D
- 2 × RTX 5070 Ti 16 GB (SM120), PCIe, no NVLink, no P2P
- one NVMe for the checkpoint (306 GiB of MXFP4 experts are loaded to DRAM;
  the engram tables stay on disk)

## What was wrong, in the order it was found

1. **Socket 1's memory path throttles under sustained load.** After 5-7
   minutes of any real inference on socket 1's cores — V4.1 through kt-kernel
   or a 9B dense model through ik_llama.cpp — socket 1's local DRAM bandwidth
   halves (99 → 49 GB/s by a user-space probe with the server idle), latency
   rises 11%, core clocks stay put, and it clears after 60 s idle. The
   node-0 expert pool looked idle because it was waiting for node 1. BIOS
   APBDIS / fixed SoC P-state did nothing; no synthetic load reproduces it; a
   desk fan on socket 1's DIMM bank removes it entirely (case inlet 50 →
   39 °C). `docs/DSV41-PREFILL-NODE1-BANDWIDTH-20260912.md`,
   `tools/membw-probe.c`, `tools/probe-node1-ikllama.sh`.
2. **Engram row fetches were serial NVMe reads.** 12 rows per call, 75-94%
   cache misses (a 2 GiB cache in front of a 101 GB table), each miss two
   `O_DIRECT` preads in sequence: 8-9 ms of every decode token. An I/O pool
   in the callback: 4 ms → 1 ms per call, +8%. `tools/engram-adapter/`.
3. **The dense FP8 GEMMs had no tuned tiling for this card.** At M=1 the
   Triton block-FP8 GEMM launched N/128 CTAs and ran at 250-500 GB/s on a
   900 GB/s card: 38 of the 70 ms of a decode step. A split-K 16-row tiling
   for M ≤ 16: 8.7 ms, +40%. `patches/fp8-skinny-splitk.patch`.
4. **The seven GPU expert slots held logical experts 0..6.** Recorded
   routing on writer prompts is skewed: the top seven experts per layer take
   45% of routed tokens (uniform would be 1.8%). Placing them in the GPU
   slots takes 45% of the bytes off the CPU: +30% decode, and prefill along
   with it. SGLang's expert-distribution recorder never sees the V4 router's
   ids and the router hands the wrapper logical ids, so the KT wrapper
   records its own histogram and remaps. `patches/kt-ep-wrapper.patch`,
   `tools/record-dsv41-routing.sh`, `tools/build-dsv41-expert-placement.py`.
5. **The CPU MXFP4 GEMM re-permuted its activations per weight slice.** The
   AVX2 fast path converts the m × k activation block to permuted FP32 on
   every call — once per 64-row weight slice, 36-80 times per expert. Under
   the server's task size that was 40% of a prefill task. `BufferA` now
   carries the permuted copy, filled once per expert: prefill 2.1×, output
   byte-identical. `patches/kt-mxfp4-aperm-once.patch`. The same rebuild
   takes DeepSeek V4-Flash's prefill from 56.5 to 31.1 s per 2000 tokens on
   this host. Upstream kvcache-ai/ktransformers #2175 and #2176 carry the
   same structure; the follow-up is
   [#2205](https://github.com/kvcache-ai/ktransformers/pull/2205).

6. **Prefill was the CPU expert GEMM, so prefill now streams the experts
   to the GPU instead.** Above a token threshold the CPU-resident experts
   are not computed on the CPU at all: kt-kernel keeps each TP part's expert
   weights and scales in memfd arenas, each GPU rank maps the part it needs,
   registers it as pinned (2 MB shmem pages: 0.2 s per layer; 4 KB pages
   took 3-38 s through the IOMMU) and DMAs eight experts at a time straight
   from the resident copy into a device slot; the CUTLASS repack, the e8m0
   scale conversion and the MoE GEMM run as one CUDA graph per group. Only
   layers 0..20 stream — under the model's SWA design layers 21..39 see 128
   tokens per chunk and stay on the CPU (with replay off, the default, all
   40 stream). 182 ms per streamed layer, 85% of the PCIe 4.0 x16 link; the
   3564-token prompt goes from 45 s to 17.3 s wall including 96 decoded
   tokens, the 6945-token one from ~90 s to 36.5 s. Two
   things that ate a day on the way: every CUDA launch costs ~250 µs of host
   time while the link is saturated by our own DMA (hence the graphs), and
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` plus FlashInfer's in-capture
   workspace allocation corrupts every prompt past the SWA window (the runner
   patch carries the one arrangement that survives).
   `patches/kt-stream-prefill.patch`, `kt-mxfp4-arena-and-upfirst-combined.patch`,
   `tools/measure-dsv41-stream-prefill.sh`, `docs/DSV41-GPU-STREAMED-PREFILL-20260913.md`.

Also in `docs/DSV41-DECODE-PROFILE-20260913.md`: the per-token budget after
all of it (CPU experts 15 ms at 82% of the socket bandwidth, GPU 18 ms,
engram 2, scheduler 3), and the profiler gotcha — SGLang's `/start_profile`
wants the activity named `"GPU"`; `"CUDA"` is silently ignored and records
no kernels.

## Layout

- `patches/` — against `sgl-project/sglang` branch `dsv4.1` (commit
  1aa0e962) and `kvcache-ai/ktransformers` kt-kernel:
  - `fp8-skinny-splitk.patch` — split-K tiling for block-FP8 GEMMs at M ≤ 16
  - `kt-ep-wrapper.patch` — kt-kernel 0.7 mask shim, `KT_NUMA_NODES`,
    routing histogram (`KT_ROUTING_DUMP`), logical→physical remap,
    zero-weight masking for the Marlin backend (`KT_GPU_MASK_ZERO`)
  - `kt-mxfp4-aperm-once.patch` — permuted activations once per expert
  - `kt-stream-prefill.patch` — `kt_stream_prefill.py` (the streamed prefill),
    the runner's shared CUTLASS workspace, the E-sizing fix; needs
    `kt-ep-wrapper.patch` (the hook, `KT_GPU_STREAM_PREFILL=<tokens>`)
  - `kt-mxfp4-arena-and-upfirst-combined.patch` — kt-kernel: `KT_EXPERT_SHM=1`
    memfd expert arenas + `expert_arena_infos()`, and `KT_WRITE_UP_FIRST=1`
    for the writer path (`kt-mxfp4-writer-upfirst.patch` is that part alone)
  - `mxfp4-avx2-gemv.patch` — the m==1 GEMV fast path (08-23; upstream #2175)
  - `kt-taskqueue-timing.patch`, `kt-prefill-stage-timing.patch` — clocks
    (`KT_TASKQUEUE_TIMING=<syncs>`, `KT_PREFILL_STAGE_TIMING=1`)
- `tools/start-dsv41-engram-nvme.sh` — the launcher (all knobs are env vars;
  set `KT_GPU0_UUID` / `KT_GPU1_UUID`)
- `tools/engram-adapter/` — engram rows from the checkpoint shards over
  NVMe with a bounded RAM cache and an I/O pool (derived from 0xSero's
  adapter)
- `tools/build-kt-dsv41.sh` — rebuild `kt_kernel_ext` with the patches
- `tools/measure-dsv41-stream-prefill.sh` — streamed-prefill A/B; the prompt
  ends in three knowledge questions because a model whose streamed experts
  arrived as zeros still copies the paragraph back as a "summary"
- `tools/memguard-dsv41.sh` / `restart-memguard.sh` — logs host memory per
  NUMA node every 5 s and kills the server at 6 GB of swap (the arenas fill
  both nodes to ~15 GB free at the end of load)
- `tools/measure-dsv41-prefill-order.sh`, `torch-profile-dsv41-*.sh`,
  `record-dsv41-routing.sh`, `build-dsv41-expert-placement.py`,
  `verify-dsv41-swa-long.sh` — the harnesses behind the numbers
- `tools/membw-probe.c`, `memlat-probe.c`, `clock-probe.c`,
  `watch-dsv41-thermal.sh`, `watch-dsv41-pools.sh`, `probe-dsv41-tripped.sh`,
  `probe-node1-ikllama.sh`, `watch-bmc-sensors.sh` — how the socket-1
  throttle was cornered without root
- `dsv41-expert-placement-7.json` — the hot-expert map recorded on Japanese
  prose prompts; record your own for your workload
- `docs/` — the write-ups, dated

## Reproducing

1. SGLang `dsv4.1` worktree at 1aa0e962 with `patches/fp8-skinny-splitk.patch`
   and `patches/kt-ep-wrapper.patch`; kt-kernel with the three kt patches,
   built by `tools/build-kt-dsv41.sh` into the venv.
2. `vm.min_free_kbytes = 8388608` (a node-bound OOM from the CUDA JIT
   otherwise kills the scheduler while 306 GiB of experts sit in DRAM).
3. Engram manifest per `docs/DSV41-ENGRAM-MXFP4.md`; `DSV41_ENGRAM_DIR`,
   `DSV41_ENGRAM_BASE`.
4. `KT_GPU0_UUID=... KT_GPU1_UUID=... tools/start-dsv41-engram-nvme.sh`.
5. Keep air moving over socket 1's DIMMs. Really.
6. For the streamed prefill (`ZEROCOPY=1` in the harness, i.e.
   `KT_EXPERT_SHM=1 KT_GPU_STREAM_ZEROCOPY=1 KT_GPU_STREAM_PREFILL=1024
   KT_GPU_STREAM_GROUP=8`), as root and not persistent across reboots:
   `echo advise > /sys/kernel/mm/transparent_hugepage/shmem_enabled`,
   `echo defer > /sys/kernel/mm/transparent_hugepage/defrag`,
   `sysctl vm.swappiness=1`. Without the first the pinning of 4 KB shmem
   pages takes minutes per request; without the other two the loader
   swaps the desktop out.

## Not done

- The streamed prefill's fixed cost is per chunk (40 layers × 182 ms), so a
  4096-token chunk would halve it per token; the 1M-token KV pool leaves
  too little VRAM for it on 16 GB cards (512K context or a smaller
  resident-expert workspace). The first request after launch pays ~5 s of
  registration and graph capture that belongs at startup.
- Speculative decoding (DSpark) pays badly while the CPU expert path costs
  per row; the m=2..8 tiling is the next kernel job.
- The ~1 in 12 samples with a stray Latin fragment before "。" is the model's
  own at t=1.0 and needs a writer-side filter.
