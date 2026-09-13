#!/usr/bin/env bash
# Prompt order matters and the descending runs could not separate "a large
# prompt degrades later ones" from "1805 is simply a 145 s prompt". ORDER sets
# the paragraph counts to send, so the same lengths can be run ascending
# (matching the original sweep) and each length can repeat. Host free memory is
# sampled alongside VRAM: the earlier runs only watched the card.
set -Eeuo pipefail
root="$HOME/KTransformers"
out="${OUT:?set OUT}"
order="${ORDER:-26 38 51 26 38}"
exec >"$out" 2>&1
echo "start $(date +%H:%M:%S) ALLOC_CONF=${ALLOC_CONF:-unset} frac=${FRAC:-0.85} flush=${FLUSH:-0} idle=${IDLE:-0} idle_at=${IDLE_AT:-} defer=${KT_DEFERRED_EXPERTS:-4} chunk=${SGLANG_CHUNKED_PREFILL_SIZE:-2048} order=$order"
# Only ever stop our own previous server, by its pid file. A blanket pkill of
# "sglang" once came within a sleep of killing a Qwen server another session
# had just started with `sglang serve` (09-12 17:17).
own_pid=$(cat "$root/logs/sglang-dsv41-engram/current/server.pid" 2>/dev/null || true)
if [ -n "$own_pid" ] && kill -0 "$own_pid" 2>/dev/null; then
  pkill -TERM -P "$own_pid" 2>/dev/null || true; kill -TERM "$own_pid" 2>/dev/null || true
  sleep 20
  pkill -KILL -P "$own_pid" 2>/dev/null || true; kill -KILL "$own_pid" 2>/dev/null || true
fi
# Refuse to launch beside another GPU job: a training run or someone else's
# server on the same cards would be slowed and would contaminate every number
# here. Anything still holding the GPUs after our own server is gone is not ours.
clear=0
for i in $(seq 1 40); do
  u0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
  u1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)
  if [ "${u0:-9999}" -lt 400 ] && [ "${u1:-9999}" -lt 100 ]; then clear=1; break; fi
  sleep 5
done
if [ "$clear" != 1 ]; then
  echo "DIED gpu busy at $(date +%H:%M:%S): $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd/ -)MiB; other GPU processes:"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
  exit 2
fi
echo "gpu clear at $(date +%H:%M:%S): $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd/ -)MiB"
[ -n "${ALLOC_CONF:-}" ] && export PYTORCH_CUDA_ALLOC_CONF="$ALLOC_CONF"
SGLANG_MEM_FRACTION="${FRAC:-0.85}" "$root/tools/start-dsv41-engram-nvme.sh" --background >/dev/null
L="$root/logs/sglang-dsv41-engram/current/server.log"
P=$(cat "$root/logs/sglang-dsv41-engram/current/server.pid")
for _ in $(seq 1 80); do sleep 15; grep -q "Uvicorn running" "$L" && break; kill -0 "$P" 2>/dev/null || { echo DIED; tail -15 "$L"; exit 1; }; done
echo "UP $(date +%H:%M:%S)"

cpuclk() { awk '/^cpu MHz/{s+=$4; n++} END{if(n) printf "%.0f", s/n}' /proc/cpuinfo; }

vmst() { awk '/^(compact_stall|compact_fail|pgscan_direct|pgsteal_direct|allocstall_normal|thp_split_pmd) /{printf "%s=%s ", $1, $2}' /proc/vmstat; }

# Automatic NUMA balancing (kernel.numa_balancing=1 on this host) unmaps the
# process's pages in scan-sized chunks and takes a hinting fault on the next
# touch; on a 330 GB RSS the sweep is slow and its cost lands on whichever
# threads touch the range being scanned. The direct-reclaim counters above
# never moved across the decay, so record the NUMA side: the global hint
# fault / PTE update / migration counters and TP0's own per-task fault totals.
numast() {
  awk '/^(numa_pte_updates|numa_hint_faults|numa_hint_faults_local|numa_pages_migrated|pgmigrate_fail) /{printf "%s=%s ", $1, $2}' /proc/vmstat
  local sp
  sp=$(pgrep -f '^sglang::scheduler_TP0' | head -1)
  [ -n "$sp" ] && awk '/numa_scan_seq|total_numa_faults|numa_preferred_nid/{gsub(/ /, ""); printf "%s ", $0}' "/proc/$sp/sched" 2>/dev/null
  echo
}

nvmeread() { awk '/nvme/{s+=$6} END{print s+0}' /proc/diskstats; }

hostmem() { awk '/MemFree/{f=$2} /^Cached/{c=$2} /MemAvailable/{a=$2} END{printf "free=%.0f cached=%.0f avail=%.0fGiB", f/1048576, c/1048576, a/1048576}' /proc/meminfo; }

send() {
  local body B t0 t1 resp pt S
  body=$(python3 - "$1" <<'PY'
import json, sys
para = "江戸時代の参勤交代は、諸大名に定期的な江戸在府と国元帰還を義務づけた制度であり、街道と宿場町の整備、藩財政の慢性的な逼迫、そして江戸の消費経済の膨張という三つの帰結をもたらした。"
text = "以下の文章を要約してください。\n\n" + "\n".join(f"{i+1}. {para}" for i in range(int(sys.argv[1])))
print(json.dumps({"model":"deepseek-v41-flash-engram-nvme","messages":[{"role":"user","content":text}],"max_tokens":1,"temperature":0}))
PY
)
  B=$(wc -l <"$L"); local rd0=$(nvmeread)
  echo "  pre  clk=$(cpuclk)MHz $(vmst)"
  echo "  pre  numa $(numast)"
  ( for i in $(seq 1 900); do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0; sleep 1; done ) > /tmp/o-$2.txt 2>/dev/null &
  S=$!
  t0=$(date +%s); resp=$(curl -s -m 3600 http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d "$body"); t1=$(date +%s)
  kill "$S" 2>/dev/null || true
  sleep 1
  pt=$(python3 -c "import json,sys;print(json.loads(sys.argv[1]).get('usage',{}).get('prompt_tokens','ERR'))" "$resp" 2>/dev/null || echo ERR)
  echo "--- $2 paras=$1 prompt_tokens=$pt wall=$((t1-t0))s"
  tail -n +"$B" "$L" | grep -E "Prefill batch" \
    | sed -E 's/^\[[0-9-]+ ([0-9:]+) (TP[01])\].*#new-token: ([0-9]+),.*input throughput \(token\/s\): ([0-9.]+).*/  done_at=\1 \2 newtok=\3 thr=\4/'
  awk '$1 ~ /^[0-9]+$/{if($1>m)m=$1; if(n==0||$1<n)n=$1; c++} END{printf "  vram gpu0 %s-%s MiB (span %s) over %d samples\n", n,m,m-n,c}' /tmp/o-$2.txt
  echo "  host $(hostmem) nvme_read=$(( ($(nvmeread) - rd0) * 512 / 1048576 ))MiB"
  echo "  post clk=$(cpuclk)MHz $(vmst)"
  echo "  post numa $(numast)"
}

# FLUSH=1 resets SGLang's KV token pool between requests (/flush_cache), to
# test whether the order-dependent slowdown lives in the pool's page free list
# rather than in CUDA memory.
i=0
for paras in $order; do
  i=$((i+1))
  # IDLE sleeps before every request after the first; IDLE_AT lists request
  # indexes to sleep before (IDLE_AT_SECS each, default 15), so a backlog can
  # be built back-to-back and then given one gap.
  if [ "$i" -gt 1 ] && { [ -n "${IDLE:-}" ] || [[ " ${IDLE_AT:-} " == *" $i "* ]]; }; then
    d="${IDLE:-${IDLE_AT_SECS:-15}}"
    echo "  idle ${d}s from $(date +%H:%M:%S)"; sleep "$d"
  fi
  if [ "${FLUSH:-0}" = "1" ] && [ "$i" -gt 1 ]; then
    r=$(curl -s -m 60 -X POST http://127.0.0.1:8080/flush_cache); echo "  flush_cache -> ${r:-ok}"
  fi
  send "$paras" "r$i-p$paras"
done
# KEEP=1 leaves the server up in whatever state the run put it in, so the
# tripped state can be probed by hand (per-node clocks, bandwidth, threads).
if [ "${KEEP:-0}" = "1" ]; then
  echo "kept server pid $P"
else
  pkill -TERM -P "$P" 2>/dev/null || true; kill -TERM "$P" 2>/dev/null || true; sleep 20
  pkill -KILL -P "$P" 2>/dev/null || true; kill -KILL "$P" 2>/dev/null || true
fi
echo "done $(date +%H:%M:%S)"
