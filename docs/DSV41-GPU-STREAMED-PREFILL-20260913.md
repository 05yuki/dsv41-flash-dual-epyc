# Design: prefill V4.1 by streaming expert groups to the GPU (2026-09-13)

Handover item 7 / P4. Not started; this is the plan and the numbers it rests on.

## Why

After the 09-13 kernel fix prefill is 74 tok/s and 97% of it is the CPU
expert GEMM (`DSV41-DECODE-PROFILE-20260913.md`). The AVX2 MXFP4 GEMM runs at
~50% of the FMA peak per core; the CPU path has at most ~1.5x left. A 60K
writer prompt is ~13 minutes. On the GPU the same GEMMs are a few hundred
milliseconds per chunk; what stops us is that the experts live in DRAM and
16 GB holds seven of them per layer.

## What exists

The KT fork's `kt_ep_wrapper.py` (`source/ktransformers/third_party/sglang`,
4477 lines) already has an **MXFP4 layerwise prefill pipeline**: kt-kernel
exports its resident expert weights through shared memory (`kt_buf_*`), the
wrapper stages one **complete layer image** (`_Mxfp4PrefillSlot`, two slots,
raw w13/w2 + scale tensors) H2D, Marlin-prepares it on device, runs the whole
layer's routed experts through the Marlin MoE, and cycles the slot for the
next layer; `--kt-gpu-prefill-token-threshold` turns it on above a chunk size.
It was measured on V4-Flash and is closed here because a layer image is
384 × ~20 MB / 2 ranks ≈ 3.75 GB per rank, ×2 slots = 7.5 GiB, on top of the
1M KV pool and 14 GB of resident weights on a 16 GB card.

The dsv4.1 worktree's `kt_ep_wrapper.py` is the 400-line upstream minimum
plus our patches: it has none of this.

## The change: slot = expert group, not layer

Keep the pipeline, shrink the unit. A slot holds **G experts** (G = 32:
32 × 20 MB / 2 ranks = 320 MB, two slots = 640 MB per rank — fits beside the
1M pool today). Per layer per chunk:

```
for group g in 0..384/G:            # 12 groups at G=32
    H2D raw(g+1) into the other slot        # async, overlaps with compute of g
    marlin-prepare(g) on device
    rows_g = tokens whose topk hit experts in g   # filter topk ids, remap to 0..G-1
    out[rows_g] += marlin_moe(rows_g, slot g)
```

The seven hot experts stay resident and are served by the existing GPU path;
the CPU experts are not called at all for chunks above the threshold. Decode
is untouched (below threshold → kt-kernel as today).

## Cost model (per rank, per 2048-token chunk, per layer)

| | |
|---|---|
| bytes over PCIe | 377 experts × 20 MB / 2 ranks ≈ **3.8 GB** |
| PCIe 4.0 x16, measured 28 GB/s (h2d+d2h pinned, 09-12) | ≈ 135-150 ms |
| Marlin prepare + GEMM for 2048 × 6 rows | tens of ms (GPU) |
| per layer, transfer-bound | ~150 ms |
| **per chunk, 40 layers** | **~6 s** |
| CPU path today (74 tok/s) | 27 s |

The transfer cost is per chunk and independent of chunk length, so larger
chunks amortize it: at 4096 tokens ~3 s per 2048-equivalent, at 8192 ~1.5 s.
Chunk size is bounded by GPU workspace (activations 8192 × 5120 × 2 B = 84 MB
are nothing; the sparse-attention/indexer workspace at long context is the
limit — 0xSero's 4×96 GB setup failed at a 4096 chunk near 399K, and our
`--enable-dynamic-chunking` exists for exactly that). Realistic target for a
60K prompt: **1-2 minutes** against 13 today and 40 two days ago.

Both ranks stream in parallel (each holds half of every expert), so the PCIe
budget above is per card, not shared.

## Work, in order

1. Port the layerwise pipeline from the fork into the dsv4.1 worktree's
   `kt_ep_wrapper.py` as-is (slot = layer) and confirm it runs at a small
   chunk with a reduced KV pool, just to have the plumbing alive on this
   branch. The fork's `mxfp4_deepseek.py` gate and `v4_marlin_moe` come with
   it (`patches/mxfp4-deepseek-marlin-partial.patch` history).
2. Regroup: slot holds G experts; per-group topk filtering/remap (the same
   logical→physical machinery the hot-expert patch added); output
   accumulation across groups; the async H2D of group g+1 during group g on
   a second stream with events.
3. Interaction with hot-7: the resident seven are physical 0..6; the
   streamed groups cover the rest; keep the wrapper's remap consistent.
4. Measure: 2048 chunk, then 4096, then dynamic chunking at 60K; the
   pass is a 60K prefill under 2 minutes with output matching the CPU path
   (Marlin W4A16 numerics are the same as the CUTLASS path within the
   0.06-nat noise measured 09-13).
5. Interplay with the socket-1 throttle: with prefill off the CPU, the
   sustained CPU load that trips socket 1 disappears for prefill entirely.

Estimate: two to three days of work, most of it in step 1's port and step
2's stream/event correctness under CUDA graph capture (prefill graph is off
for V4, which helps).

## What was actually ported (09-13, evening)

Reading the fork's pipeline settled the cut: steps 1 and 2 collapse into one
new module, because the only pieces worth carrying over are two primitives
that already exist in the dsv4.1 tree's kt-kernel build:

- `AVX2MXFP4_MOE.write_weight_scale_to_buffer_task(gpu_tp, expert, ptrs…)`
  (`operators/avx2/mxfp4-moe.hpp:1132`, bound by SFINAE in
  `ext_bindings.cpp:473`) — CPU tp part *i* memcpy's expert *e*'s packed
  e2m1 bytes and its scales (fp32 → **bf16**) into GPU rank *i*'s host
  pointer, `[gate; up]` rows for w13, `[K, N/2]` for w2;
- `NativeMoEWrapper.submit_write_weight_scale_to_buffer` /
  `sync_write_weight_scale_to_buffer` (`kt_kernel/utils/amx.py:1026`).

Everything else in the fork — `SharedFullContext`, the per-expert TP
consensus all-reduces, the Marlin repack, the epoch/round scheduler — is
either the 7.5 GiB layer image or error plumbing around it. Not ported.

`python/sglang/srt/layers/moe/kt_stream_prefill.py` (dsv4.1 worktree, ~300
lines) instead:

- **slot = G experts** (`KT_GPU_STREAM_GROUP`, 32): per rank two device
  slots in the layout the resident seven already use. That method is
  `Mxfp4FlashinferCutlassMoEMethod` (`mxfp4_flashinfer_cutlass_moe.py`),
  not the GPT-OSS `Mxfp4MoEMethod` in `mxfp4.py` I first read: w13
  `[G, 2N, K/2]` u8 loaded as `[up; gate]`, w2 `[G, K, N/2]`, e8m0 scales
  block-interleaved in place by `flashinfer.block_scale_interleave`, no
  biases, no swiglu alpha/beta, a `[G]` clamp-limit tensor and a `[G]` ones
  global scale. V4.1-Flash (N = 1152 per rank, K = 5120) needs no padding,
  so the DMA lands straight in the final layout — the half swap is two
  contiguous copies per expert, the scales get one `bf16 → exponent byte`
  pass on the device.
- **three pinned SHM host buffers** of G experts per rank (`kt_stream_*`,
  cudaHostRegister'd, opened by rank 0 like the fork's `kt_buf_*`), so the
  kt-kernel write of group g+2 runs while group g+1's DMA is in flight and
  group g's GEMM runs. The ranks hand groups over through a small SHM flag
  block (`write_seq`, per-rank `consumed`), **not** through
  torch.distributed: the first attempt used one gloo barrier per group on
  the TP cpu_group, and that is the group the scheduler's request broadcast
  runs on between forwards — the tags drifted and both ranks sat in
  `broadcast_pyobj` forever.
- **per group**: `topk_ids ∈ [base, base+n)` → local ids, else −1 (the
  masking the resident path already relies on), then the same
  `FlashInferCutlassMxfp4MoeQuantInfo` the resident `apply` builds, with the
  slot's tensors, into `gpu_method.runner.run`; outputs accumulate into one
  `[tokens, K]` bf16.
- **only layers 0..20 stream.** In prefill the SWA layers 21..39 see 128
  tokens per chunk (the window; `kv_source_layer_ids [2, 8, 14, 20]`), so
  the threshold leaves them on the CPU path, which is right: 128 × 6 / 384
  is two rows per expert.
- **hook** in `KTEPWrapperMethod.apply`: `KT_GPU_STREAM_PREFILL=<tokens>`;
  at or above it the CPU `submit`/`sync` are skipped and the streamed sum
  is added to the resident experts' output; below it (decode, graph
  capture) nothing changes. The physical ids are cloned before
  `mask_cpu_expert_ids` mutates them in place. Hot-7 needs nothing extra:
  the groups are physical 7..383, which is the index space the kt wrapper
  was loaded with.
- `KT_GPU_STREAM_TIMING=1` prints per-layer write/total ms;
  `tools/measure-dsv41-stream-prefill.sh` is the A/B harness (STREAM=0 vs
  2048, greedy text + prefill throughput per request).

Memory per rank: host 3 × 320 MB pinned, device 2 × ~340 MB (weights 283,
raw + converted scales 55) — inside the 2.8 GB left after the 1M pool and
graph capture, but only with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`:
without it the third request of the first streamed run died on a 244 MB
CUTLASS workspace allocation with 139 MB free, the per-request token
counts (2048/1482/1805) and the 7- and 32-expert shapes having fragmented
the caching allocator.

### First numbers (09-13 evening, G = 32, chunk 2048, threshold 1024)

| prompt | CPU path (09-13 morning) | streamed |
|---|---|---|
| 26 paragraphs, 1805 tokens, +48 decode | 17.0 s | 10.9-12.1 s |
| 51 paragraphs, 3530 tokens, +48 decode | — | 20.5-23.5 s |
| 100 paragraphs, 6911 tokens, +48 decode | — | 37.5 s |

Greedy text is the same summary as the CPU path (one run of the 51-paragraph
prompt chose 課し where the other chose 義務づけ; whether that is the
CUTLASS finalize's atomics or something of mine is the next gate to run).

Per streamed layer: 365 ms, of which the kt-kernel write is 196 ms
(16 ms per group, 319 MB per rank: ~15 GB/s). The DMA at 28 GB/s would be
11 ms per group and the GEMM a few, so a fully overlapped pipeline should
sit at the write's 16 ms per group ≈ 200 ms per layer; the extra 170 ms was
the group loop serialising, and finding out why took the rest of the day.

### Why the loop serialised: kernel launches cost 250 µs while the DMA runs

The per-group MoE call blocked the host for ~11 ms (G = 32) / ~6 ms (G = 16)
— the DMA time of one group — in the server, and 0.24 ms standalone. It was
not a stream sync (torch.profiler saw none), not the runner's workspace
allocation (a persistent `workspace_buffer=` changed nothing), not the SGLang
runner wrapper (calling FlashInfer directly changed nothing), not the launch
count (`KT_WRITE_UP_FIRST` cut the copies 64 → 1 per group, nothing), not
`CUDA_DEVICE_MAX_CONNECTIONS`, and not CPython's 16 KiB data-stack chunk
churn (real, but all at import time). It reproduced standalone with a
memory hog: **while host DRAM or the PCIe link is saturated, every CUDA
launch costs ~250 µs of host time instead of 8** (`x.add_(1)` 8 → 283 µs,
event record 8 → 240 µs, memset 6 → 70 µs; rusage shows it as pure user
time, no faults, no context switches — the driver's pushbuffer/fence path).
A group was ~30 launches (scale conversion, interleave, ids, the MoE's
dozen kernels), and our own DMA keeps the link saturated by design.

Fix: one CUDA graph per (token count, group), replayed after
`wait_event(dma_done)`; the graph body is the scale prep + MoE + accumulate
on persistent buffers, all graphs in one mempool. `graph_enq` per layer:
140 ms → 1 ms. The resident-expert path pays the same launch tax once per
layer; giving the SGLang runner a shared persistent workspace
(`flashinfer_cutlass.py`) removed its per-call 244 MB allocation, which was
also what OOM'd the third request.

### Numbers (09-13 16:00, G = 16, up-first writer, graphs, chunk 2048)

| prompt | CPU path | streamed | |
|---|---|---|---|
| 51 paragraphs, 3530 tokens (+48 decode) | 25.3-27.6 s | **14.4 s** | 1.8× |
| 100 paragraphs, 6911 tokens (+48 decode) | 50.7 s | **28.7 s** | 1.8× |

Prefill only (decode of 48 tokens is ~2.5 s of each): ~157 → ~290 tok/s.
Per streamed layer now: write 203 ms (8.5 ms per 16-expert group,
19 GB/s) + host wait for a free host buffer 40 ms + DMA enqueue 10 + graph
1 = 256 ms; 42 layer-calls (2 chunks × 21 layers) = 11 of the 12 s. The
DMA under the writer's memory traffic runs at 17 GB/s (26 alone), so the
writer and the DMA are now the same size and the whole prefill is those two.
The first request after launch pays SGLang's warm-up prefill (a 256-token
batch that takes 30-75 s on this host) plus ~1 s of graph captures.

Greedy text is the same summary on both paths; the exact wording differs
between the first request after launch and later ones on the CPU path too
(申勤交代 / 参勤交代), so that is not the streamer's doing.

### Next

1. Zero-copy: register kt-kernel's expert storage and DMA from it, no
   writer, no host buffers, no handshake; DMA back at 26 GB/s → ~6 ms per
   group, ~150 ms per layer, ~3 s per 2048 chunk. Needs the pointer export
   and ~300 GB of cudaHostRegister.
2. Chunk 4096 once VRAM allows (the 1M pool leaves 2.8 GB; G = 16 with the
   shared workspace fits, G = 32 did not).
3. Pre-capture the group graphs at startup so the first request does not
   pay for them.

### Where the time will go, and the next cut

The writer is a DRAM→DRAM memcpy of the same 3.8 GB per layer per rank the
PCIe carries, so on this host the CPU copy and the DMA are the same order
(≈10-15 ms per group each) and the pipeline is bounded by whichever is
slower, ≈12 ms × 12 groups × 40 layers ≈ 6 s per 2048 chunk. The
zero-copy step after this — register kt-kernel's own expert storage
(`gate_bb_[e]->b/d`, contiguous per expert and per tp part, exactly one
rank's half) as pinned and DMA from it directly — removes the writer, the
host buffers and the barrier; it needs a small pybind to export the
pointers and ~300 GB of `cudaHostRegister`, which is the open question.

## Zero-copy landed (09-14 01:31)

kt-kernel `KT_EXPERT_SHM=1`: the per-expert MXFP4 loader keeps each TP
part's packed weights in three memfd arenas (gate/up/down, expert *e* at
`base + e * stride`) and the fp32 group scales in three scale-only arenas;
`expert_arena_infos()` reports `(fd, size, stride)` ×6 per part. Rank 0 sends
the fds to the other ranks over a unix socket (`send_fds`); each rank mmaps
its part, `madvise(MADV_HUGEPAGE)`s it and `cudaHostRegister`s it on the
layer's first streamed group, then DMAs a group as six contiguous copies
into a raw landing slot; the CUTLASS repack ([up; gate], e8m0 scales,
`block_scale_interleave`) runs inside the per-group graph. No writer, no
host buffers, no rank handshake.

| | CPU path | writer + graphs (09-13 16:00) | zero-copy |
|---|---|---|---|
| 26 paragraphs, 1839 tokens (+96 decode) | 17.4 s | — | **13.4 s** (first request, incl. registration + captures) |
| 51 paragraphs, 3564 tokens (+96 decode) | 25-27 s | 14.4 s (+48 decode) | **12.8 s** |

Per streamed layer 180 ms at G = 8 (48 groups × 89 MB at ~25 GB/s: the
PCIe 4.0 x16 link, 85% of the measured 26 GB/s H2D). Prefill of the
3564-token prompt ≈ 8 s ≈ 440 tok/s (was ~157 on the CPU). Output verified
with the harness's knowledge questions: the model summarises, then answers
キャンベラ / 391 / H₂O (and, on the 51-paragraph version, calls the prompt
でたらめ before answering — a live V4.1, not a copy-back).

Host requirements, all volatile until put in sysctl.d / rc.local:

- `/sys/kernel/mm/transparent_hugepage/shmem_enabled = advise`: with 4 KB
  shmem pages `cudaHostRegister` of a 4.2 GB arena took 3-38 s per layer
  through the IOMMU (the first request blew the 20-minute watchdog); with
  huge pages 0.2 s. `defrag = defer` (never let the arena's page faults do
  direct compaction: with `madvise` they swapped the desktop out).
- `vm.swappiness = 1`: at 60 the loader's page-cache pressure swapped 8 GB
  of anonymous memory with 165 GB available. Even so the two NUMA nodes
  bottom out at 15-20 GB free at the end of load (322 GB shmem + the kt
  staging pools + KV); `tools/memguard-dsv41.sh` logs node0/node1 free every
  5 s and kills the server at 6 GB of swap.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for VRAM — only with
  the runner variant in `kt-stream-prefill.patch` (see the regression note
  above): FlashInfer's own in-capture workspace allocation plus expandable
  segments produced garbage past the SWA window.

Next: chunk 4096 (VRAM: the 1M pool leaves too little; 512K context or a
smaller resident-expert workspace), pre-registration and graph capture at
startup so the first request does not pay 5 s, and the streamed layers'
DMA at G = 16 once the raw landing fits.

## 09-14 02:50: the multi-request corruption was SWA Bounded Replay, not ours

With replay on, the third request of a session (100 / 51 / 100 paragraphs,
96-token completions) comes back as garbage on the **plain CPU path** as
well; with it off the same three answer correctly and the first and third
are byte-identical. The 09-13 gate measured one request per launch and
never saw it. The launcher default is back to off; the bug belongs to the
dsv4.1 tree's replay implementation and is reported as such.

Consequences for the streamed prefill: with replay off every layer sees the
whole chunk, so all 40 layers stream (not 21) and the CPU path is ~2×
slower too. Final numbers, replay off, zero-copy, G = 8, shared workspace
preallocated before capture, no expandable segments, four requests in one
session all correct (キャンベラ / 391 / H₂O each time):

| prompt | CPU path | streamed |
|---|---|---|
| 26 paragraphs, 1839 tokens (+96 decode) | ~30 s | **11.0 s** |
| 51 paragraphs, 3564 tokens (+96 decode) | 45.1 s | **17.3 s** |
| 100 paragraphs, 6945 tokens (+96 decode) | 85-92 s | **36.5 s** (first request 96 s: registration + 1920 graph captures) |

182 ms per streamed layer, 40 layers, ≈ 7.3 s per 2048-token chunk. The
runner's shared workspace is now allocated once in `create_moe_runner`
(before any capture, sized for the chunk and `KT_GPU_STREAM_GROUP`) and
never replaced; the 09-13 "grow at the first prefill" form left the decode
graphs writing into freed memory and that, too, showed up only on later
requests.
