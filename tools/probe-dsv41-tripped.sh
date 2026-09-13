#!/usr/bin/env bash
# Probe a server the order harness left up with KEEP=1 after tripping the
# prefill decay. Sends one 701-token request while sampling per-node busy
# clocks and the pool thread states, then measures each socket's read
# bandwidth with tools/membw-probe while the server is idle, sends another
# request, then idles in RECOVER-second steps, measuring bandwidth and one
# request after each step, until a request comes back at the clean time.
# Usage: OUT=... [RECOVER="60 60 60 60 60"] [CLEAN=34] tools/probe-dsv41-tripped.sh
set -Eeuo pipefail
root="$HOME/KTransformers"
out="${OUT:?set OUT}"
recover="${RECOVER:-60 60 60 60 60}"
clean="${CLEAN:-34}"
exec >"$out" 2>&1
pid=$(pgrep -f '^sglang::scheduler_TP0' | head -1)
[ -n "$pid" ] || { echo "no server"; exit 1; }
declare -a node
while IFS=, read -r c n; do node[$c]=$n; done < <(lscpu -p=CPU,NODE | grep -v "^#")

body=$(python3 - <<'PY'
import json
para = "江戸時代の参勤交代は、諸大名に定期的な江戸在府と国元帰還を義務づけた制度であり、街道と宿場町の整備、藩財政の慢性的な逼迫、そして江戸の消費経済の膨張という三つの帰結をもたらした。"
text = "以下の文章を要約してください。\n\n" + "\n".join(f"{i+1}. {para}" for i in range(10))
print(json.dumps({"model":"deepseek-v41-flash-engram-nvme","messages":[{"role":"user","content":text}],"max_tokens":1,"temperature":0}))
PY
)

# per-node mean MHz of the cores busy over the last 0.5 s, plus pool states
sample() {
  read -r -a a < <(awk '/^cpu[0-9]/{print $2+$3+$4+$6+$7+$8, $5}' /proc/stat | paste -sd' ' -)
  sleep 0.5
  read -r -a b < <(awk '/^cpu[0-9]/{print $2+$3+$4+$6+$7+$8, $5}' /proc/stat | paste -sd' ' -)
  mapfile -t mhz < <(awk '/^cpu MHz/{print $4}' /proc/cpuinfo)
  local n0=0 s0=0 n1=0 s1=0 i db di tot m
  for ((i=0; i<${#mhz[@]}; i++)); do
    db=$(( ${b[2*i]} - ${a[2*i]} )); di=$(( ${b[2*i+1]} - ${a[2*i+1]} ))
    tot=$(( db + di )); [ "$tot" -gt 0 ] || continue
    if [ $(( db * 100 / tot )) -ge 60 ]; then
      m=${mhz[$i]%.*}
      if [ "${node[$i]:-0}" = 0 ]; then n0=$((n0+1)); s0=$((s0+m)); else n1=$((n1+1)); s1=$((s1+m)); fi
    fi
  done
  local st
  st=$(for t in /proc/"$pid"/task/*; do c=$(cat "$t/comm" 2>/dev/null); case "$c" in numa_*_t_*) echo "${c%%_t_*} $(awk '{print $3}' "$t/stat" 2>/dev/null)";; esac; done | sort | uniq -c | awk '{printf "%s=%s(%d) ", $2, $3, $1}')
  echo "  $(date +%H:%M:%S) node0 busy=$n0 mhz=$(( n0 ? s0/n0 : 0 )) node1 busy=$n1 mhz=$(( n1 ? s1/n1 : 0 )) | $st"
}

request() {  # $1 = label
  local t0 t1
  echo "=== request $1 at $(date +%H:%M:%S)"
  ( for _ in $(seq 1 120); do sample; sleep 1.5; done ) &
  local S=$!
  t0=$(date +%s)
  curl -s -m 3600 http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d "$body" >/dev/null
  t1=$(date +%s)
  kill "$S" 2>/dev/null || true; wait "$S" 2>/dev/null || true
  echo "=== request $1 wall=$((t1-t0))s"
}

bw() {  # $1 = label
  local n
  echo "=== bandwidth $1 at $(date +%H:%M:%S) (server idle)"
  for n in 0 1; do
    echo "  node $n 8t: $(numactl --cpunodebind=$n --membind=$n "$root/tools/membw-probe" 8 512 3)"
    echo "  node $n 1t: $(numactl --cpunodebind=$n --membind=$n "$root/tools/membw-probe" 1 512 3)"
  done
  # effective core clock, one core per socket, to separate a memory-side
  # drop from a hidden core throttle
  echo "  clock cpu8: $(taskset -c 8 "$root/tools/clock-probe" 3000)   cpu40: $(taskset -c 40 "$root/tools/clock-probe" 3000)"
}

echo "start $(date +%H:%M:%S) pid $pid recover steps=$recover clean=${clean}s"
request tripped-1
bw tripped
request tripped-2
total=0
for step in $recover; do
  echo "=== idle ${step}s from $(date +%H:%M:%S)"
  sleep "$step"; total=$((total+step))
  bw "after-${total}s"
  echo "=== request after-${total}s at $(date +%H:%M:%S)"
  t0=$(date +%s)
  curl -s -m 3600 http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d "$body" >/dev/null
  w=$(( $(date +%s) - t0 ))
  echo "=== request after-${total}s wall=${w}s"
  if [ "$w" -le "$clean" ]; then echo "recovered after ${total}s idle"; break; fi
done
echo "done $(date +%H:%M:%S)"
