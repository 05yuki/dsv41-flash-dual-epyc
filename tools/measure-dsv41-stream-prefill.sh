#!/usr/bin/env bash
# GPU-streamed prefill A/B (P4, 09-13). Launches the V4.1 server with
# KT_GPU_STREAM_PREFILL=$STREAM (0 = the CPU expert path), sends the same
# summarisation prompts greedily with a short completion, and prints per
# request: prompt tokens, wall, SGLang's prefill throughput lines, the
# [kt-stream] per-layer clocks and the greedy text — so a STREAM=0 run and a
# STREAM=2048 run can be diffed for both speed and output.
set -Eeuo pipefail
root="$HOME/KTransformers"
out="${OUT:?set OUT}"
order="${ORDER:-26 51 100}"
stream="${STREAM:-2048}"
max_tokens="${MAX_TOKENS:-96}"
exec >"$out" 2>&1
echo "start $(date +%H:%M:%S) alloc=${ALLOC_CONF:-unset} frac=${FRAC:-0.85} zerocopy=${ZEROCOPY:-0} graph=${GRAPH:-1} upfirst=${UPFIRST:-1} stream=$stream group=${GROUP:-16} hostbufs=${HOSTBUFS:-3} chunk=${SGLANG_CHUNKED_PREFILL_SIZE:-2048} ctx=${SGLANG_CONTEXT_LENGTH:-default} order=$order max_tokens=$max_tokens"
# ATTACH=1: a server launched by hand (or by a harness that died) is already
# loading or up; skip the stop/launch and just wait for it.
if [ "${ATTACH:-0}" = "1" ]; then
  L="$root/logs/sglang-dsv41-engram/current/server.log"
  P=$(cat "$root/logs/sglang-dsv41-engram/current/server.pid")
  for _ in $(seq 1 80); do grep -q "Uvicorn running" "$L" && break; kill -0 "$P" 2>/dev/null || { echo DIED; tail -30 "$L"; exit 1; }; sleep 15; done
  echo "ATTACHED $(date +%H:%M:%S) pid=$P vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd/ -)MiB"
fi
if [ "${ATTACH:-0}" != "1" ]; then
# Only ever stop our own previous server, by its pid file (09-12 rule).
own_pid=$(cat "$root/logs/sglang-dsv41-engram/current/server.pid" 2>/dev/null || true)
if [ -n "$own_pid" ] && kill -0 "$own_pid" 2>/dev/null; then
  pkill -TERM -P "$own_pid" 2>/dev/null || true; kill -TERM "$own_pid" 2>/dev/null || true
  sleep 20
  pkill -KILL -P "$own_pid" 2>/dev/null || true; kill -KILL "$own_pid" 2>/dev/null || true
fi
clear=0
for i in $(seq 1 40); do
  u0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
  u1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)
  if [ "${u0:-9999}" -lt 400 ] && [ "${u1:-9999}" -lt 100 ]; then clear=1; break; fi
  sleep 5
done
if [ "$clear" != 1 ]; then
  echo "DIED gpu busy at $(date +%H:%M:%S): $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd/ -)MiB"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
  exit 2
fi
# Knobs, all read by kt_stream_prefill.py (KT_WRITE_UP_FIRST also by the
# patched kt-kernel writer): GRAPH (per-group CUDA graphs, default on),
# UPFIRST (writer emits [up; gate], one copy per group), GROUP (experts per
# slot), HOSTBUFS, and the diagnostics NOMOE / SYNCDMA / PROFILE / DEBUG.
export KT_GPU_STREAM_SELFTEST="${SELFTEST:-0}" KT_EXPERT_SHM="${ZEROCOPY:-0}" KT_GPU_STREAM_ZEROCOPY="${ZEROCOPY:-0}" KT_GPU_STREAM_GRAPH="${GRAPH:-1}" KT_GPU_STREAM_SYNC_DMA="${SYNCDMA:-0}" KT_WRITE_UP_FIRST="${UPFIRST:-1}" KT_GPU_STREAM_NOMOE="${NOMOE:-0}" KT_GPU_STREAM_PROFILE="${PROFILE:-0}" KT_GPU_STREAM_DEBUG="${DEBUG:-0}" KT_GPU_STREAM_PREFILL="$stream" KT_GPU_STREAM_GROUP="${GROUP:-16}" KT_GPU_STREAM_HOST_BUFS="${HOSTBUFS:-3}" KT_GPU_STREAM_TIMING="${TIMING:-1}"
# ALLOC_CONF=expandable_segments:True: the per-request token counts and the
# 7- and 16-expert shapes fragment the ~2.8 GB left beside the 1M pool.
[ -n "${ALLOC_CONF:-}" ] && export PYTORCH_CUDA_ALLOC_CONF="$ALLOC_CONF"
old_pid="${own_pid:-}"
SGLANG_MEM_FRACTION="${FRAC:-0.85}" "$root/tools/start-dsv41-engram-nvme.sh" --background >/dev/null
L="$root/logs/sglang-dsv41-engram/current/server.log"
# --background returns before the new run dir's pid file is written; the
# stale pid of the previous server read here once made the loop below declare
# DIED while the new server was loading (09-13).
P=""
for _ in $(seq 1 60); do
  P=$(cat "$root/logs/sglang-dsv41-engram/current/server.pid" 2>/dev/null || true)
  if [ -n "$P" ] && [ "$P" != "$old_pid" ] && kill -0 "$P" 2>/dev/null; then break; fi
  P=""; sleep 1
done
[ -n "$P" ] || { echo "DIED: no new server pid after launch"; exit 1; }
for _ in $(seq 1 80); do sleep 15; grep -q "Uvicorn running" "$L" && break; kill -0 "$P" 2>/dev/null || { echo DIED; tail -30 "$L"; exit 1; }; done
echo "UP $(date +%H:%M:%S) vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd/ -)MiB"
fi

send() {
  local body B t0 t1 resp pt S
  body=$(python3 - "$1" "$max_tokens" <<'PY'
import json, sys
para = "江戸時代の参勤交代は、諸大名に定期的な江戸在府と国元帰還を義務づけた制度であり、街道と宿場町の整備、藩財政の慢性的な逼迫、そして江戸の消費経済の膨張という三つの帰結をもたらした。"
# The three questions at the end need the routed experts: a model whose
# streamed experts arrived as zeros still copies the paragraph back as a
# "summary" (09-13), but cannot answer these.
text = ("以下の文章を一文で要約し、そのあと次の三つに一行ずつ答えてください: オーストラリアの首都、17×23、水の化学式。\n\n"
        + "\n".join(f"{i+1}. {para}" for i in range(int(sys.argv[1]))))
print(json.dumps({"model":"deepseek-v41-flash-engram-nvme","messages":[{"role":"user","content":text}],"max_tokens":int(sys.argv[2]),"temperature":0}))
PY
)
  B=$(wc -l <"$L")
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 -lms 1000 > /tmp/o-$2.txt 2>/dev/null &
  S=$!
  t0=$(date +%s.%N); resp=$(curl -s -m 3600 http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d "$body"); t1=$(date +%s.%N)
  kill "$S" 2>/dev/null || true
  sleep 1
  python3 - "$resp" "$2" "$1" "$t0" "$t1" <<'PY'
import json, sys
try:
    r = json.loads(sys.argv[1])
    u = r.get("usage", {})
    txt = r["choices"][0]["message"]["content"]
except Exception as e:
    print(f"--- {sys.argv[2]} paras={sys.argv[3]} ERR {e}: {sys.argv[1][:300]}"); sys.exit(0)
wall = float(sys.argv[5]) - float(sys.argv[4])
print(f"--- {sys.argv[2]} paras={sys.argv[3]} prompt_tokens={u.get('prompt_tokens')} completion_tokens={u.get('completion_tokens')} wall={wall:.1f}s")
print("  text: " + txt.replace("\n", "\\n"))
PY
  tail -n +"$B" "$L" | { grep -E "Prefill batch" || true; } \
    | sed -E 's/^\[[0-9-]+ ([0-9:]+) (TP[01])\].*#new-token: ([0-9]+),.*input throughput \(token\/s\): ([0-9.]+).*/  done_at=\1 \2 newtok=\3 thr=\4/'
  tail -n +"$B" "$L" | { grep -E "\[kt-stream\] rank0 layer" | grep -v below || true; } | awk '{for(i=1;i<=NF;i++){if($i~/^write=/){sub("write=","",$i);sub("ms","",$i);w+=$i} if($i~/^total=/){sub("total=","",$i);sub("ms","",$i);t+=$i}} n++} END{if(n) printf "  kt-stream rank0: %d layer-calls, write %.1fs, total %.1fs\n", n, w/1000, t/1000}'
  awk '$1 ~ /^[0-9]+$/{if($1>m)m=$1; if(n==0||$1<n)n=$1; c++} END{printf "  vram gpu0 %s-%s MiB (span %s) over %d samples\n", n,m,m-n,c}' /tmp/o-$2.txt
  tail -n +"$B" "$L" | { grep -iE "Scheduler hit an exception|MemoryError|illegal|out of memory" || true; } | head -n 5 | sed "s/^/  LOG: /"
}

i=0
for paras in $order; do
  i=$((i+1))
  send "$paras" "r$i-p$paras"
done
if [ "${KEEP:-0}" = "1" ]; then
  echo "kept server pid $P"
else
  pkill -TERM -P "$P" 2>/dev/null || true; kill -TERM "$P" 2>/dev/null || true; sleep 20
  pkill -KILL -P "$P" 2>/dev/null || true; kill -KILL "$P" 2>/dev/null || true
fi
echo "done $(date +%H:%M:%S)"
