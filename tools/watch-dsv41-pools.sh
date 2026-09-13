#!/usr/bin/env bash
# Per-pool CPU time of TP0's kt-kernel threads, sampled every INTERVAL seconds
# over SPAN-second windows, for running beside a prefill run. One line per
# sample: for each subpool the worker threads' summed CPU ticks over the span
# and how many of them ran at all, plus the pool manager thread (numa_N_m_0)
# and the scheduler main thread. A pool whose workers show ~0 ticks while the
# other pool's workers are busy is the stall seen in the busy-core counts,
# now attributed to the pool rather than the socket.
set -Eeuo pipefail
out="${OUT:?set OUT}"
interval="${INTERVAL:-10}"
span="${SPAN:-5}"
exec >"$out" 2>&1
pid=""
while [ -z "$pid" ]; do pid=$(pgrep -f '^sglang::scheduler_TP0' | head -1); [ -n "$pid" ] || sleep 5; done
hz=$(getconf CLK_TCK)
echo "# pid $pid span ${span}s hz $hz; per pool: workers_ticks workers_active/n manager_ticks; then main_ticks"
snap() {
  local t
  for t in /proc/"$pid"/task/*; do
    [ -r "$t/stat" ] || continue
    awk -v comm="$(tr -d '\n' <"$t/comm")" '{ s=$0; sub(/^[^)]*\) /, "", s); split(s, f, " "); print $1, comm, f[12]+f[13] }' "$t/stat"
  done
}
while kill -0 "$pid" 2>/dev/null; do
  a=$(snap); sleep "$span"; b=$(snap)
  line=$(join -j1 <(echo "$a" | sort -k1,1) <(echo "$b" | awk '{print $1, $3}' | sort -k1,1) | awk -v main="$pid" '
    { d=$4-$3 }
    $2 ~ /^numa_[0-9]+_t_/ { p=$2; sub(/_t_.*/, "", p); w[p]+=d; n[p]++; if (d>0) act[p]++ }
    $2 ~ /^numa_[0-9]+_m_/ { p=$2; sub(/_m_.*/, "", p); m[p]+=d }
    $1 == main { mt=d }
    END { for (p in n) printf "%s: %d %d/%d m=%d | ", p, w[p], act[p]+0, n[p], m[p]+0; printf "main=%d", mt+0 }')
  echo "$(date +%H:%M:%S) $line"
  sleep "$interval"
done
echo "# scheduler $pid gone $(date +%H:%M:%S)"
