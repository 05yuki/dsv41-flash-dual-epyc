#!/usr/bin/env bash
# DeepSeek V4.1-Flash on the dual-EPYC host: sgl-project/sglang branch dsv4.1
# (worktree source/sglang-dsv41), kt-kernel MXFP4 CPU experts, and the engram
# tables served from NVMe through tools/dsv41-engram-adapter instead of the
# upstream 203 GB anonymous-RAM host table. See native-ubuntu/DSV41-ENGRAM-MXFP4.md.
set -Eeuo pipefail
export LC_ALL=C
export PYTHONUNBUFFERED=1

root="${KTRANSFORMERS_ROOT:-$HOME/KTransformers}"
mode="${1:---foreground}"
# Prefer the NVMe copy staged by tools/stage-dsv41-to-nvme.sh; loading the
# 476 GiB checkpoint off the archive HDD costs ~50 minutes per attempt.
if [[ -z "${DSV41_MODEL:-}" && -f "$root/models/deepseek-v41-flash/config.json" ]]; then
  DSV41_MODEL="$root/models/deepseek-v41-flash"
fi
model="${DSV41_MODEL:-$HOME/models/deepseek-v41-flash}"
engram_dir="${DSV41_ENGRAM_DIR:-$root/models/dsv41-engram}"
adapter_dir="$root/tools/dsv41-engram-adapter"
source_dir="$root/source/sglang-dsv41/python"
alias_name="deepseek-v41-flash-engram-nvme"
log_name="sglang-dsv41-engram"
port="${SGLANG_PORT:-8080}"
# V4.1 spends 1,630 bytes of KV per token, so the full million fits in the same
# ~2 GB budget 65k used and costs no measurable decode speed.
context_length="${SGLANG_CONTEXT_LENGTH:-1048576}"
max_total_tokens="${SGLANG_MAX_TOTAL_TOKENS:-1048576}"
# Measured (logs/dsv41-decode-sweep.txt): 7 vs 12 GPU experts is a wash, since
# 12 of 384 routed experts is 3% of activations; 64/96/112 CPU threads are all
# the same and 128 collapses to 1.5 tok/s by starving the GPU-side threads.
# 5 (was 7) since 09-16: hot-7 Marlin + streamer + 1M pool left 0.5 GB and OOMed
# at 22K context; five hottest experts still catch 40% of routed slots (uniform 1.3%).
# Stays at 5 (user decision 09-24: the 1M pool comes first). The placement is
# rebuilt from V4.1's own routing under the writer load: the 09-13 map caught
# 41.8% of routed slots, the new 5-slot map 47.9% (decode 27 -> 29.2 tok/s).
# More slots eat the pool: 10 slots left it 559K tokens (31.3 tok/s), 8 slots
# ~540K. Maps for 7/8/10 are in native-ubuntu/ for a decode-first run.
gpu_experts="${KT_GPU_EXPERTS:-5}"
cpu_threads="${KT_CPU_THREADS:-64}"
# Deferring experts lets the GPU run ahead instead of blocking on the CPU
# stage, and 0/2/4/6 measured 11.91/13.57/15.78/19.12 tok/s -- but 6, which is
# num_experts_per_tok, DESTROYS the output: kana ratio 0.01, Chinese text,
# wrong facts, then a repetition loop. That speed is bought by dropping expert
# contributions. Do not raise this without reading the generated text.
# 0 (was 4) since 09-23. On V4-Flash-Vision 4 deferred made decode diverge from
# prefill: teacher-forced the right continuation byte after the lead piece of
# 掴/挿 sits at logprob ~0, yet generation at t=0.3 still emitted wrong ones,
# 0.71-1.09 U+FFFD per 1k tokens against 0-0.11 with none deferred. Same kernel
# path here, and the writer needs clean Japanese. Not re-measured on V4.1.
deferred_experts="${KT_DEFERRED_EXPERTS:-0}"
# kt hands MoE work over with stream memops and a spinning poller instead of
# host functions (native-ubuntu/patches/kt-stream-memops.py). 09-24 arms:
# held-out prose 25.4/26.1 -> 27.0/27.0 tok/s, paired NLL -0.0021 (SE 0.011,
# n=3832). Two cores spin.
export KT_STREAM_MEMOPS="${KT_STREAM_MEMOPS:-1}"
export KT_TASKQUEUE_SPIN_US="${KT_TASKQUEUE_SPIN_US:-300}"
# Decode all-reduces through host memory in NCCL's LL form
# (patches/sglang-ll-allreduce.py; Vision-Exp gained 4-6%). Here 29.2 -> 29.4
# tok/s only: the CPU experts dominate the token. Sums identical to NCCL's.
export SGLANG_LL_ALLREDUCE="${SGLANG_LL_ALLREDUCE:-1}"
# One expert pool per NUMA node by default. KT_THREADPOOL_COUNT=1 puts the
# whole pool on node 0 (kt-kernel's sequential default); KT_NUMA_NODES=1 moves
# it to node 1 (the worktree's kt_ep_wrapper reads it, patch of 09-12). Both
# are for isolating the node-0 pool stall in DSV41-PREFILL-SCALING-20260911.md.
# KT_TP=1 KT_NUMA_NODE_LIST=0 runs a single rank on GPU0 (attention and the GPU
# experts must then fit one 16 GB card: KT_GPU_EXPERTS=0 and a shorter
# SGLANG_CONTEXT_LENGTH), for isolating whether socket 1's GPU rank is part of
# the prefill-decay trigger.
# 0.87 (was 0.85): the streamer now takes its device slots before the pool is
# sized (KT_GPU_STREAM_EARLY_INIT), which at 0.85 truncated the pool to
# 1,022,464 tokens; 0.87 keeps the full 1M.
# 0.81 (was 0.87) since 09-24: a 48K prefill died in the indexer's candidate
# selection at 0.85-0.87 whatever the slot count (its [2048, prefix] scores
# grow with the prompt; patches/sglang-dsv4-candidate-blocks-nopad.py removed
# a second copy of them). With 5 slots the pool has 1.1 GB to spare above 1M
# tokens, so the fraction goes to the prefill instead: 0.80 kept 1,023,744
# tokens and passed 51K in 284 s; 0.81 keeps the full 1M.
mem_fraction="${SGLANG_MEM_FRACTION:-0.81}"
kv_cache_dtype="${SGLANG_KV_CACHE_DTYPE:-fp8_e4m3}"
# Prefill chunk. With 6-of-384 routing each expert sees chunk*6/384 rows per
# chunk (32 at 2048), and the CPU expert stage streams every touched expert's
# weights once per chunk, so larger chunks amortize the same bytes over more
# rows. Bounded above by what the KT staging buffers and VRAM tolerate.
chunked_prefill="${SGLANG_CHUNKED_PREFILL_SIZE:-2048}"
# SGLANG_EXTRA_ARGS passes arbitrary flags through, word-split on purpose, so a
# one-off A/B needs no launcher edit (e.g. --enable-dynamic-chunking).
extra_args=(${SGLANG_EXTRA_ARGS:-})
# SWA pool headroom (09-16). With --disable-radix-cache the tree caps the SWA
# pool at one request (5120 tokens) and the scheduler then cuts prefill into
# 2048/768/1792 chunks; the streamed prefill costs ~7.5 s per chunk whatever
# its size, so a 48K prompt took 37 chunks / 283 s. Six prefix tails raise
# the cap to 7424 and every chunk is 2048: 24 chunks / 198 s. Costs ~1 GB of
# VRAM per rank (SWA slots carry the c4 state), which hot-5 leaves free.
swa_prefix_tails="${SGLANG_SWA_PREFIX_TAILS:-6}"
# The radix cache stays off in production. DSV41_RADIX=1 keeps it, for the
# depth ladder (tools/sglang-ladder.py), where each stage must reuse the
# previous stage's context to time only the new chunk.
radix_args=(--disable-radix-cache)
[[ "${DSV41_RADIX:-0}" == "1" ]] && radix_args=()
# Hot-expert placement (09-13): the seven GPU slots per layer hold the seven
# experts the writer's routing hits most (45% of routed tokens) instead of
# logical 0..6. Decode 18 -> 23.5 tok/s, no measurable degradation
# (DSV41-DECODE-PROFILE-20260913.md). The map is built by
# tools/build-dsv41-expert-placement.py from a KT_ROUTING_DUMP histogram and
# needs the worktree's kt_ep_wrapper remap (patches/kt-ep-wrapper.patch).
# KT_EXPERT_PLACEMENT=none turns it off; a path overrides the default file.
placement="${KT_EXPERT_PLACEMENT:-$root/dsv41-expert-placement-${gpu_experts}.json}"
if [[ "$placement" != "none" ]]; then
  if [[ -f "$placement" ]]; then
    extra_args+=(--init-expert-location "$placement")
  else
    echo "expert placement $placement missing; running with logical 0..6 on the GPU" >&2
  fi
fi
# Kernel for the resident experts (09-16, found on V4-Flash): the tree's
# default FlashInfer CUTLASS MXFP8 x MXFP4 kernel returns outlier tokens 6-12%
# short; Marlin is W4A16 and matches the checkpoint within 0.5%. Needs the
# wrapper's KT_GPU_MASK_ZERO (Marlin cannot take -1 ids). KT_HOT_BACKEND=cutlass
# restores the 09-13 behaviour.
hot_backend="${KT_HOT_BACKEND:-marlin}"
if [[ "$gpu_experts" != "0" && "$hot_backend" == "marlin" ]]; then
  extra_args+=(--moe-runner-backend marlin)
  export KT_GPU_MASK_ZERO=1
fi
# W16 streaming reads the bf16 experts straight from kt-kernel's shared expert
# arenas (KT_EXPERT_SHM=1, KT_GPU_STREAM_ZEROCOPY=1) under a captured CUDA graph
# (KT_GPU_STREAM_GRAPH=1). kt_stream_prefill rejects W16 without the zero-copy
# graph mode, so these three are W16 preconditions, not options -- the same set
# tools/start-dsv4-flash-stream.sh exports. (09-18: they were only on the
# V4-Flash launcher; V4.1 had W16 alone and the scheduler aborted at startup
# with "KT_GPU_STREAM_W16 needs the zero-copy graph mode".)
export KT_EXPERT_SHM="${KT_EXPERT_SHM:-1}"
# Same tree as V4-Flash, which has had this on since 09-16: the
# prefill-sized all-reduce goes through a host staging buffer instead
# of NCCL's SHM transport, which measures 1.9 GB/s on these two cards.
export SGLANG_HOST_STAGED_ALLREDUCE="${SGLANG_HOST_STAGED_ALLREDUCE:-1}"
export KT_GPU_STREAM_ZEROCOPY="${KT_GPU_STREAM_ZEROCOPY:-1}"
export KT_GPU_STREAM_GRAPH="${KT_GPU_STREAM_GRAPH:-1}"
# Streamed prefill in bf16 (dequantized on the GPU) for the same reason; the
# packed MXFP8 x MXFP4 path is KT_GPU_STREAM_W16=0.
export KT_GPU_STREAM_W16="${KT_GPU_STREAM_W16:-1}"
# Streamed prefill: a 2048-token chunk streams the 377 CPU experts of a layer
# in ~185 ms (7.5 s per chunk, ~270 tok/s) whatever the chunk size, against
# 74 tok/s on the CPU path, so it pays above ~550 tokens. G=8: the bf16 group
# buffer is G x 35 MB and G=16 did not fit beside the seven Marlin experts.
# The arenas are pinned at load time (EARLY_INIT): lazily, the 4K-page third
# of them cost the first request 500 s on 09-16.
export KT_GPU_STREAM_PREFILL="${KT_GPU_STREAM_PREFILL:-512}"
export KT_GPU_STREAM_GROUP="${KT_GPU_STREAM_GROUP:-4}"
export KT_GPU_STREAM_EARLY_INIT="${KT_GPU_STREAM_EARLY_INIT:-1}"
# Each rank's pinning takes 70-1550 s with fragmented memory, and the ranks can
# finish more than the stock 480 s apart: the post-load barrier then killed the
# launch (09-22, 09-23). patches/sglang-load-barrier-timeout.py reads this.
export SGLANG_UNBALANCED_LOAD_TIMEOUT_S="${SGLANG_UNBALANCED_LOAD_TIMEOUT_S:-1800}"
# The arenas want 2 MB pages (kt-kernel madvises them): on 4K pages the pinning
# above took 20-60 min and wandered, on 2 MB it is 0.2 s a layer. Two things
# are needed: the host's shmem THP at advise (resets on reboot), and the
# checkpoint's page cache dropped before each layer's experts load, or the
# late layers find no free 2 MB block
# (patches/sglang-kt-drop-cache-per-layer.py). 09-24: 328 of 337 GB huge,
# startup 60+ min -> 11 min, NLL and decode unchanged.
export KT_GPU_STREAM_DROP_CACHE="${KT_GPU_STREAM_DROP_CACHE:-$model}"
if grep -q '\[never\]' /sys/kernel/mm/transparent_hugepage/shmem_enabled 2>/dev/null; then
  echo "WARNING: shmem THP is 'never'; the expert arenas will be 4K and pinning takes 20-60 min." >&2
  echo "         echo advise | sudo tee /sys/kernel/mm/transparent_hugepage/shmem_enabled" >&2
fi
# SWA Bounded Replay is part of the V4.1 design (model card: decoder SWA KV
# reconstructed by replaying the last n_win tokens) and halves prefill here.
# Gate 09-13 (t=1.0/p=0.95, 12 samples a side): replay off had 2 comma-shredded
# and 1 looping sample, replay on had none. Default on; =0 turns it off.
# 09-14: back to off by default. With replay on, the third request of a
# session (100/51/100-paragraph prompts, 96-token completions) comes back as
# garbage on the plain CPU path too; with it off the same three answer
# correctly and the first and third are byte-identical. The 09-13 gate
# measured one request per launch and never saw it. Prefill is ~2x slower
# with it off (every layer sees the whole chunk); the streamed prefill
# covers all 40 layers then.
if [[ "${SGLANG_SWA_BOUNDED_REPLAY:-0}" == "1" ]]; then
  # Opt-in, prefill-only speedup; not numerically equivalent to full prefill.
  extra_args+=(--enable-decoder-swa-bounded-replay)
fi
# Engram row cache: DSV41_CACHE_GIB is the total across both tables and both
# TP ranks (engram_backend.py splits it). 64 GiB holds ~250M rows, far more
# than any single session touches (43 new rows per token).
export DSV41_ENGRAM_DIR="$engram_dir"
# The engram tables are served straight out of the checkpoint shards, so the
# rows follow wherever the checkpoint lives; this overrides the manifest's
# base_dir so moving it needs no edit.
export DSV41_ENGRAM_BASE="${DSV41_ENGRAM_BASE:-$model}"
# 64 GiB here means 16 GiB per store, and four stores of anonymous huge pages
# on top of 306 GiB of experts push the host into constant reclaim: the same
# deferred=4 config measured 15.78 tok/s at 8 GiB and 7.63 at 64.
export DSV41_CACHE_GIB="${DSV41_CACHE_GIB:-8}"
export DSV41_ENGRAM_MODE="${DSV41_ENGRAM_MODE:-nvme}"
# FlashInfer autotune re-picks kernel tactics on every launch (the per-rank
# caches were seen to disagree and get discarded), and different tactics give
# different greedy output. With it off, two launches of the same config
# reproduce a 300-token greedy completion byte for byte, at no measured cost
# (14.97 / 15.27 vs 15.78 tok/s). Keep it off whenever outputs are compared.
autotune_args=()
if [[ "${SGLANG_DISABLE_FLASHINFER_AUTOTUNE:-1}" == "1" ]]; then
  autotune_args=(--disable-flashinfer-autotune)
fi
spec_args=()
if [[ "${SGLANG_ENABLE_DSPARK:-0}" == "1" ]]; then
  spec_args=(--speculative-algorithm DSPARK --speculative-attention-mode "${SGLANG_SPEC_ATTENTION_MODE:-decode}")
fi
venv="$root/venv-dsv41"
python="$venv/bin/python"
sysroot_lib="$root/runtime/sysroot/usr/lib/x86_64-linux-gnu"
compat_header="$root/tools/cuda13-glibc-compat.h"
# Note the wiring: the launcher's first device (TP0, whose scheduler runs on
# node 0) is nvidia-smi's GPU 1 at 81:00.0, which hangs off socket 1; the
# second is GPU 0 at 21:00.0 on socket 0. Every TP0 GPU<->host transfer for
# the CPU expert path therefore crosses xGMI into socket 1's IOD.
# KT_GPU_UUIDS overrides the list (e.g. the 21:00.0 UUID alone for a
# single rank on the socket-0 card).
gpu0_uuid="${KT_GPU0_UUID:?set KT_GPU0_UUID}"
gpu1_uuid="${KT_GPU1_UUID:?set KT_GPU1_UUID}"
gpu_devices="${KT_GPU_UUIDS:-$gpu0_uuid,$gpu1_uuid}"

if [[ "$mode" != "--foreground" && "$mode" != "--background" ]]; then
  echo "Usage: $0 [--foreground|--background]" >&2
  exit 2
fi
[[ -x "$python" ]] || { echo "venv missing: $venv" >&2; exit 3; }
[[ -f "$model/config.json" ]] || { echo "model missing: $model" >&2; exit 4; }
[[ -f "$engram_dir/engram-manifest.json" ]] || { echo "engram tables missing: $engram_dir (run tools/extract_engram_tables.py)" >&2; exit 4; }
[[ -f "$adapter_dir/librow_store.so" ]] || { echo "build $adapter_dir/librow_store.so first (g++ -O2 -std=c++17 -shared -fPIC)" >&2; exit 4; }
[[ -f "$source_dir/sglang/srt/layers/engram.py" ]] || { echo "dsv4.1 worktree missing: $source_dir" >&2; exit 4; }

# Match the server processes themselves, not any shell whose command line
# mentions a log path like logs/sglang-.../server.log (09-13: a waiting
# `grep` on that path tripped this and the launch was refused).
if pgrep -af 'llama-server|-m sglang[. ](serve|launch_server)' | grep -F "$root" >/dev/null; then
  echo "A native local-LLM server is already running; port $port is exclusive." >&2
  exit 6
fi
if pgrep -af '^sglang::(scheduler|detokenizer)' >/dev/null; then
  echo "An orphaned SGLang worker is still running; use run.sh stop first." >&2
  exit 6
fi

# KT prefill lend (patches/kt_lend.py through sglang-dsv41-kt-prefill-lend.py,
# on by default since 10-06, KT_PREFILL_LEND=0 turns it off): the decoder
# weights (not the engram's) go to the prefill while it runs, so the chunk can
# grow past what the KV pool leaves. The chunk comes from what the server
# measured (tools/kt-lend-auto.sh): the first launch calibrated 16384 (out of
# memory, the indexer's scores grow with chunk x prefix) and settled at 8192.
# 10-06 against chunk 2048 without it: 8K 257 -> 602 tok/s, 38K 255 -> 636-639,
# 114K 259 -> 594 (441 -> 192 s), decode 25.5 -> 27.4-27.9, paired NLL z -1.90
# (at chunk 4096). The SWA pool needs nothing here (with prefix tails its cap
# already counts the chunk), but it grows with the chunk: the full-attention
# KV pool is 1,031,680 tokens at 8192 instead of 1M.
KT_PREFILL_LEND="${KT_PREFILL_LEND:-1}"
[[ -f "$source_dir/sglang/srt/layers/moe/kt_lend.py" ]] || KT_PREFILL_LEND=0  # tree without the patch
export KT_PREFILL_LEND
if [[ "$KT_PREFILL_LEND" == 1 ]]; then
  # The KV pool comes first (10-07): a wider chunk needs a wider SWA cap, and on
  # V4.1 each SWA slot costs several full ones. Chunk 10240 (20 tails) took the
  # full pool to 896,000 (114K prompts 790 tok/s); 6144 with the six default
  # tails keeps the whole 1,048,576. The lend chunk stays at or below
  # DSV41_LEND_MAX_CHUNK unless SGLANG_CHUNKED_PREFILL_SIZE is given.
  lend_max_chunk="${DSV41_LEND_MAX_CHUNK:-6144}"
  export KT_PREFILL_LEND_MAX_CHUNK="$lend_max_chunk"
  # the calibration measures with the lend probe up to the context's end (a
  # 1M-token prompt is the case the chunk must survive; 10-07 the probe put a
  # 6144-row chunk at 5,286 MiB of 5,609 free there, 2,921 of it allocated)
  export KT_PREFILL_LEND_PROBE_CONTEXT="${KT_PREFILL_LEND_PROBE_CONTEXT:-$max_total_tokens}"
  # shellcheck source=kt-lend-auto.sh
  source "$root/tools/kt-lend-auto.sh"
  # the lend takes nothing from the KV pool: kt-lend-auto's KV guard narrows a
  # chunk whose wider SWA pool cost full-pool tokens on the last launch
  export KT_PREFILL_LEND_KV_CAP="${KT_PREFILL_LEND_KV_CAP:-$max_total_tokens}"
  kt_lend_auto "$0" SGLANG_CHUNKED_PREFILL_SIZE "$port" "$model" "$context_length" "$max_total_tokens" \
    "$mem_fraction" "$gpu_devices" "$gpu_experts" "$placement" "${SGLANG_SWA_PREFIX_TAILS:-6}" || exit $?
  chunked_prefill="$KT_LEND_CHUNK"
  if [[ -z "${SGLANG_CHUNKED_PREFILL_SIZE:-}" ]] && (( chunked_prefill > lend_max_chunk )); then
    chunked_prefill="$lend_max_chunk"
  fi
  # The SWA cap counts two chunks in flight, but with the overlap scheduler the
  # previous chunk's slots are still held while the next is built, so with six
  # tails every other batch gets 3072 of 6144 rows. Tails of chunk / 768 more
  # fill them (10-07, chunk 6144: 14 tails, every batch 5888-6144 rows), but
  # they come out of the KV pool (1,046,272 instead of 1,048,576; 22 tails cost
  # 4.4%), so only with DSV41_LEND_FILL_TAILS=1: by default the lend takes
  # nothing from the KV pool.
  if [[ -z "${SGLANG_SWA_PREFIX_TAILS:-}" && "${DSV41_LEND_FILL_TAILS:-0}" == 1 ]]; then
    swa_prefix_tails=$(( 6 + (chunked_prefill + 767) / 768 ))
  elif [[ -z "${SGLANG_SWA_PREFIX_TAILS:-}" ]]; then
    # As many of those tails as the KV pool's slack pays for, read from the
    # last run: the pool sizer logs the full pool its budget allows before
    # the cap cuts it to max_total_tokens (chunk 6144, 6 tails: 1,090,304 for
    # 1,048,576), and each SWA slot costs 14.33 full ones here (6 -> 14 -> 22
    # tails took 1,090,304 -> 1,046,272 -> 1,002,240), a tail 384 slots.
    last_log="$(ls -td "$root/logs/$log_name"/2*/ 2>/dev/null | head -1)server.log"
    if [[ -f "$last_log" ]]; then
      swa_prefix_tails="$(python3 - "$last_log" "$max_total_tokens" "$chunked_prefill" \
        "${DSV41_SWA_FULL_COST:-14.33}" <<'EOF'
import re, sys
log, cap, chunk, cost = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
text = open(log, errors="replace").read()
sizes = re.findall(r"DSV4 pool sizes: full=(\d+)", text)
tails = re.findall(r"DSV4 SWA sizing: mode=cap, .*prefix_tails=(\d+)", text)
chunks = re.findall(r"'chunked_prefill_size': (\d+)", text)
want = 6 + (chunk + 767) // 768
if not sizes or not tails or not chunks:
    print(6)
    sys.exit()
budget, ran, ran_chunk = int(sizes[0]), int(tails[0]), int(chunks[0])
# SWA slots the slack pays for, less the two chunks in flight growing from the
# last run's chunk to this one's; whole tails of what is left
spare_slots = (budget - cap) / cost - 2 * (chunk - ran_chunk)
print(max(6, min(want, ran + int(spare_slots // 384))))
EOF
)"
    fi
  fi
  echo "prefill lend: chunk $chunked_prefill, SWA prefix tails $swa_prefix_tails" >&2
fi
extra_args+=(--swa-prefix-tails "$swa_prefix_tails")

timestamp="$(date +%Y%m%d-%H%M%S)"
run_dir="$root/logs/$log_name/$timestamp"
mkdir -p "$run_dir"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$gpu_devices"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=0
export NCCL_DEBUG=WARN
export KT_MXFP4_BACKEND=avx2
export TORCH_CUDA_ARCH_LIST="12.0+PTX"
export FLASHINFER_CUDA_ARCH_LIST=12.0a
export CUDA_HOME=/usr/local/cuda-13.1
export PATH="$CUDA_HOME/bin:$venv/bin:$PATH"
export LD_LIBRARY_PATH="$sysroot_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:+$NVCC_PREPEND_FLAGS }-U_GNU_SOURCE -D_DEFAULT_SOURCE -include $compat_header"
export OMP_NUM_THREADS="$cpu_threads"
# SGLang's NUMA bind v2 launches each rank under `numactl --membind=N`, a hard
# MPOL_BIND that every child inherits. With 306 GiB of CPU experts resident,
# both nodes sit at their free floor, and the CUDA JIT compiler (cicc) spawned
# during attention-backend autotune then dies on a node-constrained OOM that
# takes the scheduler with it. v1 keeps the CPU pinning but uses
# numa_set_preferred, so allocations still prefer the local node and can spill
# instead of failing. kt-kernel binds its own expert pools either way.
export SGLANG_NUMA_BIND_V2="${SGLANG_NUMA_BIND_V2:-0}"
# The worktree shadows the venv's sglang; the adapter dir supplies
# sitecustomize.py, which installs the engram hook in every worker.
export PYTHONPATH="$source_dir:$adapter_dir${PYTHONPATH:+:$PYTHONPATH}"

command=(
  "$python" -m sglang.launch_server
  --host 0.0.0.0
  --port "$port"
  --model-path "$model"
  --load-format safetensors
  --trust-remote-code
  --served-model-name "$alias_name"
  --tensor-parallel-size "${KT_TP:-2}"
  --numa-node ${KT_NUMA_NODE_LIST:-0 1}
  --disable-custom-all-reduce
  --kt-weight-path "$model"
  --kt-method MXFP4
  --kt-num-gpu-experts "$gpu_experts"
  --kt-cpuinfer "$cpu_threads"
  --kt-threadpool-count "${KT_THREADPOOL_COUNT:-2}"
  --kt-max-deferred-experts-per-token "$deferred_experts"
  --context-length "$context_length"
  --max-total-tokens "$max_total_tokens"
  --kv-cache-dtype "$kv_cache_dtype"
  --mem-fraction-static "$mem_fraction"
  --max-running-requests 1
  --chunked-prefill-size "$chunked_prefill"
  --max-prefill-tokens "$chunked_prefill"
  --sampling-backend pytorch
  --watchdog-timeout 1200
  --disable-shared-experts-fusion
  "${radix_args[@]}"
  --enable-metrics
  "${autotune_args[@]}"
  "${extra_args[@]}"
  "${spec_args[@]}"
)
# (SGLANG_EXTRA_ARGS is already in extra_args above; it used to be appended a
# second time here, which made every extra flag appear twice.)

printf '%s\n' "DeepSeek V4.1-Flash: dsv4.1 SGLang + KT MXFP4 experts + NVMe engram"
printf '%s\n' "ctx: $context_length; GPU experts: $gpu_experts; CPU threads: $cpu_threads; engram cache: ${DSV41_CACHE_GIB} GiB ($DSV41_ENGRAM_MODE)"
printf '%s\n' "Run directory: $run_dir"

ln -sfn "$run_dir" "$root/logs/$log_name/current"
if [[ "$mode" == "--background" ]]; then
  nohup "${command[@]}" >>"$run_dir/server.log" 2>&1 &
  pid=$!
  printf '%s\n' "$pid" >"$run_dir/server.pid"
  echo "Started PID $pid"
else
  exec "${command[@]}" 2>&1 | tee -a "$run_dir/server.log"
fi
