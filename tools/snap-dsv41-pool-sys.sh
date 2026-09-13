#!/usr/bin/env bash
# User vs system CPU time per kt-kernel pool of TP0 over SPAN seconds. A pool
# that is busy at 100% but half as productive is either starved of memory
# bandwidth (user time) or burning its time in the kernel (system time:
# page faults from NUMA hinting, futex churn). Output per pool:
# user_ticks sys_ticks and the sys share.
set -Eeuo pipefail
pid="${PID:-$(pgrep -f '^sglang::scheduler_TP0' | head -1)}"
span="${SPAN:-5}"
[ -n "$pid" ] || { echo "no sglang::scheduler_TP0" >&2; exit 1; }
snap() {
  local t
  for t in /proc/"$pid"/task/*; do
    [ -r "$t/stat" ] || continue
    awk -v comm="$(tr -d '\n' <"$t/comm")" '{ s=$0; sub(/^[^)]*\) /, "", s); split(s, f, " "); print $1, comm, f[12], f[13] }' "$t/stat"
  done
}
a=$(snap); sleep "$span"; b=$(snap)
join -j1 <(echo "$a" | sort -k1,1) <(echo "$b" | awk '{print $1, $3, $4}' | sort -k1,1) \
  | awk -v main="$pid" -v when="$(date +%H:%M:%S)" '
    { du=$5-$3; ds=$6-$4 }
    $2 ~ /^numa_[0-9]+_t_/ { p=$2; sub(/_t_.*/, "", p); u[p]+=du; s[p]+=ds }
    $2 ~ /^numa_[0-9]+_m_/ { p=$2 "_m"; sub(/_m_.*/, "", p); p=p "_m"; u[p]+=du; s[p]+=ds }
    $1 == main { u["main"]+=du; s["main"]+=ds }
    END { printf "%s", when; for (p in u) printf " %s: u=%d s=%d sys%%=%.0f |", p, u[p], s[p], (u[p]+s[p]) ? 100*s[p]/(u[p]+s[p]) : 0; printf "\n" }'
