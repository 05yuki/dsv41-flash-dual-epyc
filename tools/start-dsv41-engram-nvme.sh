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
gpu_experts="${KT_GPU_EXPERTS:-7}"
cpu_threads="${KT_CPU_THREADS:-64}"
# Deferring experts lets the GPU run ahead instead of blocking on the CPU
# stage, and 0/2/4/6 measured 11.91/13.57/15.78/19.12 tok/s -- but 6, which is
# num_experts_per_tok, DESTROYS the output: kana ratio 0.01, Chinese text,
# wrong facts, then a repetition loop. That speed is bought by dropping expert
# contributions. Do not raise this without reading the generated text.
deferred_experts="${KT_DEFERRED_EXPERTS:-4}"
# One expert pool per NUMA node by default. KT_THREADPOOL_COUNT=1 puts the
# whole pool on node 0 (kt-kernel's sequential default); KT_NUMA_NODES=1 moves
# it to node 1 (the worktree's kt_ep_wrapper reads it, patch of 09-12). Both
# are for isolating the node-0 pool stall in DSV41-PREFILL-SCALING-20260911.md.
# KT_TP=1 KT_NUMA_NODE_LIST=0 runs a single rank on GPU0 (attention and the GPU
# experts must then fit one 16 GB card: KT_GPU_EXPERTS=0 and a shorter
# SGLANG_CONTEXT_LENGTH), for isolating whether socket 1's GPU rank is part of
# the prefill-decay trigger.
mem_fraction="${SGLANG_MEM_FRACTION:-0.85}"
kv_cache_dtype="${SGLANG_KV_CACHE_DTYPE:-fp8_e4m3}"
# Prefill chunk. With 6-of-384 routing each expert sees chunk*6/384 rows per
# chunk (32 at 2048), and the CPU expert stage streams every touched expert's
# weights once per chunk, so larger chunks amortize the same bytes over more
# rows. Bounded above by what the KT staging buffers and VRAM tolerate.
chunked_prefill="${SGLANG_CHUNKED_PREFILL_SIZE:-2048}"
# SGLANG_EXTRA_ARGS passes arbitrary flags through, word-split on purpose, so a
# one-off A/B needs no launcher edit (e.g. --enable-dynamic-chunking).
extra_args=(${SGLANG_EXTRA_ARGS:-})
# Hot-expert placement (09-13): the seven GPU slots per layer hold the seven
# experts the writer's routing hits most (45% of routed tokens) instead of
# logical 0..6. Decode 18 -> 23.5 tok/s, no measurable degradation
# (DSV41-DECODE-PROFILE-20260913.md). The map is built by
# tools/build-dsv41-expert-placement.py from a KT_ROUTING_DUMP histogram and
# needs the worktree's kt_ep_wrapper remap (patches/kt-ep-wrapper.patch).
# KT_EXPERT_PLACEMENT=none turns it off; a path overrides the default file.
placement="${KT_EXPERT_PLACEMENT:-$root/dsv41-expert-placement-7.json}"
if [[ "$placement" != "none" ]]; then
  if [[ -f "$placement" ]]; then
    extra_args+=(--init-expert-location "$placement")
  else
    echo "expert placement $placement missing; running with logical 0..6 on the GPU" >&2
  fi
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
# Set these to your two cards (nvidia-smi -L). The first is TP0, whose
# scheduler runs on NUMA node 0.
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
  --disable-radix-cache
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
