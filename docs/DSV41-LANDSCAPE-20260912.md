# Who else is running DeepSeek V4.1-Flash, and what transfers — 2026-09-12

Survey of the public V4.1-Flash (and closest V4-Flash) deployments as of this
morning, read for techniques that apply to this host: dual EPYC, 2 x 16 GB
SM120 cards, TP=2, MXFP4 experts on CPU through kt-kernel, engram from NVMe
(`DSV41-HANDOVER-20260911.md`). The prefill problem on this host is a CPU
expert pool stalling (`DSV41-PREFILL-SCALING-20260911.md`); nobody below has
that problem because nobody below runs experts on the CPU.

## The deployments

| who | hardware | engine | experts | prefill | decode (1 stream) | notes |
|---|---|---|---|---:|---:|---|
| [0xSero](https://github.com/0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000) | 4 x RTX PRO 6000 96 GB (SM120), 128 GB DDR5 | SGLang TP4/EP4 | all on GPU | 5.8-7.4K tok/s | 75-231 tok/s | engram rows: 64 GiB DDR5 cache + NVMe on miss (our adapter derives from this); DSpark on; 2048 chunk (4096 failed near 399K); sparse prefill split into 64-token pages for SM120 |
| [tonyd2wild](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark) | 4 x DGX Spark (GB10, SM121) | vLLM TP4 | all on GPU | 0.9-1.5K tok/s | 38-74 tok/s | engram on disk, later local NVMe per rank; 64-state/page indexer paging for SM12x; `top_k_per_row_decode` instead of `persistent_topk` (1.6-3.6x); GB10 has a two-state GEMV slowdown (70 vs 230 GB/s) reproduced outside the engine |
| [sfxnz](https://github.com/sfxnz/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark) | 2 x DGX Spark | vLLM TP2 | all on GPU, EXL3 2.0 bpw | TTFT 0.37 s (short) | 21 tok/s | DSpark-5; 2.0 bpw is a quality question for prose |
| [shi3z](https://note.com/shi3zblog/n/nd5fc5341b342) | 8 x A100 80 GB, 2 x Xeon 4410Y, 1.7 TB | custom | GPU (FP4 -> INT8/BF16 in registers), CPU INT8 VNNI tried | not reported | 165-673 tok/s batched, 1467 aggregate with 2 replicas | task-aware batching: same-kind prompts share experts (216 -> 156 distinct per layer); MTP; kernel-launch reduction |
| [Tono_Ken3](https://x.com/Tono_Ken3) | 16 GB VRAM + CPU | official `inference/` | CPU | — | 23 tok/s on code | tilelang `act_quant` non-deterministic on SM120; reference implementation makes the stack bit-exact; Japanese speculative breaks |
| [dnhkng](https://dnhkng.github.io/posts/gh200-benchmarking-part-4-dsv4-released/) (V4-Flash) | 2 x GH200 | SGLang vs vLLM | all on GPU | 10.3K tok/s (8K), 2.9K at 1M | 178-317 tok/s | adaptive chunking `738M / context`, floor 1280, cut 1M prefill 895 -> 358 s; SGLang +15% over vLLM |
| [KTransformers upstream](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepSeek-V4-Flash.md) (V4-Flash) | 1 x RTX 5090 32 GB, >=200 GB RAM | SGLang + kt-kernel | 10 on GPU, rest CPU | layerwise GPU prefill >= 2048 tokens | 20+ tok/s, 32.7 with MTP | the only other CPU-expert stack; `--kt-threadpool-count 2`; layerwise prefill needs a layer's experts in VRAM |
| [vcruz305](https://huggingface.co/vcruz305/DeepSeek-V4.1-Flash-GGUF) | — | llama.cpp `runtime/deepseek41` (WIP) | CPU | — | — | loader, engram, hyper-connections verified; sparse attention still missing; no mainline support, no tok/s yet |
| this host | 2 x EPYC (4 CCD each), 2 x 16 GB SM120 | SGLang + kt-kernel TP2 | 7 on GPU, 377 on CPU | 25 tok/s clean, 2x worse after ~7K tokens back-to-back | 14.6-15.8 tok/s | engram from NVMe, 0 RAM; bounded replay 1.6-3.4x on prefill, unverified for prose |

The all-GPU deployments prefill at 1-10K tok/s. This host prefills at 25.
The gap is the CPU expert stage, not the GPU, not the indexer, not the engram.

## What transfers, in order of value

### 1. Stream expert weights to the GPU for prefill instead of computing them on the CPU

Two independent sources describe the same idea. KTransformers' own "layerwise
prefill" (`--kt-gpu-prefill-token-threshold`, default on for prompts >= 2048)
uploads a layer's experts to the GPU and runs the chunk there; and
[arXiv 2606.10493](https://arxiv.org/abs/2606.10493) ("Stream-Loading Prefill")
reports 1,200 tok/s on DeepSeek-V3 by streaming expert weights through PCIe
during prefill, 1,800 with expert parallelism, and 32K-45K prompts in under 30 s.

The arithmetic for this host: 306 GiB of MXFP4 experts across the MoE layers is
roughly 7-8 GiB per layer, 20 MB per expert. Over PCIe 4 at ~25 GB/s a whole
layer streams in ~0.3 s; forty layers is ~12 s per prefill **regardless of
prompt length**, plus the GPU compute. A 60K-token writer prompt costs ~40 min
on the CPU path today (60K / 25 tok/s) and would cost tens of seconds streamed.
This is the only technique in the survey that changes the order of magnitude.

Why it was closed here: the HANDOVER records that layerwise prefill needs ~7 GiB
per rank for V4-Flash and more for V4.1, which does not fit beside the model on
a 16 GB card. The way around is finer granularity — stream per expert, not per
layer: sort the chunk's tokens by routed expert, upload one expert (20 MB), run
its tokens, discard, next. Peak VRAM is a handful of experts, the PCIe volume
is the same, and the CPU pool leaves the prefill path entirely (which also
removes the node-0 stall from prefill). This is a kt-kernel/SGLang change, not a
flag. Worth a design pass before anything else on the prefill side.

### 2. Put the hot experts on the GPU

shi3z's observation that same-kind prompts pick the same experts (216 -> 156
distinct per layer) is what `tools/record-dsv41-routing.sh` and
`tools/build-dsv41-expert-placement.py` are for (added this morning, not yet
run). With 7 GPU slots out of 384, uniform routing gives the GPU 1.8% of the
expert work; prose routing concentrated enough to give it 10-20% takes that
much off the CPU stage for free, via `--init-expert-location`. Small, but free.

### 3. Check which indexer kernels the SM120 path actually runs

tonyd2wild found two SM12x-specific wins in vLLM on GB10: `persistent_topk`
over-subscribes 48 SMs and needs 99 KB of shared memory, and swapping to
`top_k_per_row_decode` gave 1.6-3.6x on decode; and the default 1:128 indexer
paging does not work with 32/64 block sizes, fixed by a 64-state/page layout.
0xSero splits sparse prefill into independent 64-token pages with separate
scratch on SM120. This host's SGLang fork has its own SM120 handling (the DSV4
prefill graph is disabled by architecture, `_low_ratio_index_topk_dense` is the
path taken, `arg_groups/cuda_graph_hook.py:287`). Which top-k kernel decode
uses here, and whether the dense prefill path could take 0xSero's page split,
are both one grep away and neither has been looked at.

### 4. Expect little from speculative decoding on prose

tonyd2wild's DSpark acceptance on TP4: code and tables ~6 tokens/step, **prose
and narrative ~2**. KTransformers upstream reports MTP at 1.2x on V4-Flash. The
memory note that DSpark blocks speculative here should be revisited eventually,
but for a writer workload the ceiling is ~1.5-2x on decode and nothing on
prefill. Not this week.

### 5. Adaptive chunking only matters past ~100K

dnhkng's policy (`738M / context`, floor 1280) exists to keep the indexer
workspace bounded at 1M; at 3530 tokens it was measured here as a no-op. The
writer's 60K prompts sit under the point where it starts to bite. Keep the
flag in mind for the 1M experiments, ignore it for prefill speed.

### 6. Engram: our adapter already has what the others learned the hard way

tonyd2wild's Boot 3 audit found ranks 1-3 silently reading rank 0's engram rows
(offset ignored), and Boot 10 found NFS-served rows costing 2-3x the local ones
per step. 0xSero's adapter, which ours derives from, keeps per-rank row
ownership and a local NVMe path; `row_store_range` here is per rank and the
tables are read straight from the local checkpoint shards. Nothing to adopt;
worth a one-time check that the rank-1 row range in the log is not rank 0's.

### 7. Two-state slowdowns are a thing on other hardware too — measure, do not theorise

tonyd2wild's GB10 flips between 70 and 230 GB/s GEMV and stays slow after long
idle, reproduced with a standalone PyTorch script outside the engine
(`tools/gpuflip.py`), and a separate clock latch at 630-950 MHz that only a
power-cycle clears. Different hardware, different cause, same lesson as
yesterday: a sampler that reads clocks and utilisation *during* the slow state
found ours in ten minutes after a day of reading pool code. The
`tools/watch-dsv41-thermal.sh` habit stays.

### Not applicable

- Tono's `act_quant` fix: this stack is SGLang + DeepGEMM, and greedy output is
  byte-identical across runs (`DSV41-DECODE-TUNING-20260911.md`).
- shi3z's INT8 VNNI / AMX CPU experts and register-level FP4 -> INT8: Xeon
  instructions; this host is EPYC on kt-kernel's AVX2 MXFP4 path.
- shi3z's kernel-launch reduction: decode here is already at the memory
  bandwidth the CCD count allows (`DSV41-DECODE-VS-BANDWIDTH`).
- EXL3 2.0 bpw: fits two Sparks, but 2 bpw on a writer model needs a quality
  gate nobody has published.
- llama.cpp: no V4.1 sparse attention yet; when it lands, ik_llama.cpp's V4
  numbers (2x mainline on CPU) make it the thing to re-measure against.

## SGLang upstream, specifically

- **No release serves V4.1 yet.** The cookbook
  ([DeepSeek-V4_1](https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1))
  runs off the preview image `lmsysorg/sglang:dev-dsv41`, with verified cells
  for GB300 / H200 / B200 / B300 and MI350X only. SM120 is not a verified
  target; this host's `source/sglang-dsv41` worktree is ahead of upstream on
  SM120, not behind it.
- **Upstream ships `--enable-decoder-swa-bounded-replay` in its own recipe** (the
  H200 low-latency cell) and describes it as *numerically equivalent* and faster
  prefill. The launcher comment here says "not numerically equivalent" because
  greedy output diverges at character 0 with the flag on; the cookbook's claim
  means that divergence is floating-point ordering flipping an argmax, not a
  lossy approximation — replay reconstructs the same SWA KV states from the
  last window. That moves the prior on adopting it for prose from "prove it is
  harmless" to "confirm once at t=1.0/0.95 and switch". The prose gate
  (`tools/verify-dsv41-swa-long.sh` at the card's sampling) is still the step,
  but it is a confirmation, not a trial.
- **DSpark block 5 is the default speculative setting** upstream, and it is
  incompatible with PD disaggregation. The 1M context is not documented in the
  cookbook at all.
- **SM120 prefill is an open upstream problem for V4 too.**
  [#33422](https://github.com/sgl-project/sglang/issues/33422): 4 x RTX PRO
  6000, dsv4 backend, 2-7K tok/s prefill vs vLLM's 12.5K on the same cards,
  prefill CUDA graphs off for capture-pool pressure (the same reason as here),
  and throughput falling from 6.5K to ~2K as the sequence position grows — the
  out-of-window KV read that bounded replay removes. No maintainer answer.
  [#31578](https://github.com/sgl-project/sglang/issues/31578) asks for native
  SM120 `flash_mla_sparse_fwd` for sparse prefill;
  [#23657](https://github.com/sgl-project/sglang/issues/23657) reports no SM120
  fallback in the compressed attention backend;
  [#19637](https://github.com/sgl-project/sglang/issues/19637) is the SM120
  plan. Nothing in them about CPU experts.
- Nothing upstream about the encoder-side replay flag
  (`enable_encoder_swa_bounded_replay`, present in this worktree); the cookbook
  only sets the decoder one.

## What nobody else has

A CPU expert pool. Every V4.1 deployment above with numbers is all-GPU, and the
one other CPU-expert stack (KTransformers upstream) documents V4-Flash on a
32 GB card with layerwise GPU prefill doing the heavy lifting. The node-0 pool
stall is therefore this host's to solve, and the survey's answer to "how do the
others prefill fast" is item 1: they do not run prefill through a CPU pool at
all.

Sources: the links in the table, plus
[Unsloth's V4 guide](https://unsloth.ai/docs/models/deepseek-v4),
[the vLLM V4 post](https://github.com/vllm-project/vllm-project.github.io/blob/main/_posts/2026-04-24-deepseek-v4.md)
(KV compression layout, kernel fusions),
[llama.cpp discussion #22376](https://github.com/ggml-org/llama.cpp/discussions/22376),
[kt-kernel issue #2084](https://github.com/kvcache-ai/ktransformers/issues/2084)
(GPU-resident experts still loaded into CPU RAM — check host RSS here too).
