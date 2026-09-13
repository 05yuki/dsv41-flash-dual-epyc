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

| | before (09-11) | after (09-13) |
|---|---|---|
| decode, single stream | 11-12 tok/s | **24 tok/s** |
| prefill (2048-token chunks) | 25 tok/s | **74 tok/s** |
| prefill after ~5 min of continuous load | 2× slower, indefinitely | flat |
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
  - `mxfp4-avx2-gemv.patch` — the m==1 GEMV fast path (08-23; upstream #2175)
  - `kt-taskqueue-timing.patch`, `kt-prefill-stage-timing.patch` — clocks
    (`KT_TASKQUEUE_TIMING=<syncs>`, `KT_PREFILL_STAGE_TIMING=1`)
- `tools/start-dsv41-engram-nvme.sh` — the launcher (all knobs are env vars;
  set `KT_GPU0_UUID` / `KT_GPU1_UUID`)
- `tools/engram-adapter/` — engram rows from the checkpoint shards over
  NVMe with a bounded RAM cache and an I/O pool (derived from 0xSero's
  adapter)
- `tools/build-kt-dsv41.sh` — rebuild `kt_kernel_ext` with the patches
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

## Not done

- Prefill is still the CPU expert GEMM (97% of a prefill step); the
  order-of-magnitude path is streaming expert weights to the GPU per expert
  for prefill, which the 16 GB cards can hold one expert at a time.
- Speculative decoding (DSpark) pays badly while the CPU expert path costs
  per row; the m=2..8 tiling is the next kernel job.
- The ~1 in 12 samples with a stray Latin fragment before "。" is the model's
  own at t=1.0 and needs a writer-side filter.
