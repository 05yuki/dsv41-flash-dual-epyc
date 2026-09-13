#!/usr/bin/env bash
# Log host memory every 5 s while the DSV41 server runs; kill it if swap use
# passes 2 GB (09-13: a run with huge-page shmem arenas pushed the box into
# swap and took the desktop down).
root="$HOME/KTransformers"
out="${1:-/tmp/memguard.txt}"
while true; do
  p=$(cat "$root/logs/sglang-dsv41-engram/current/server.pid" 2>/dev/null || true)
  read -r _ total used free shared bc avail < <(free -m | sed -n 2p)
  swapused=$(free -m | awk 'NR==3{print $3}')
  shmem=$(awk '/^Shmem:/{printf "%d", $2/1024}' /proc/meminfo)
  hp=$(awk '/^ShmemHugePages:/{printf "%d", $2/1024}' /proc/meminfo)
  n0=$(awk '/^Node 0 MemFree/{printf "%d", $4/1024}' /sys/devices/system/node/node0/meminfo)
  n1=$(awk '/^Node 1 MemFree/{printf "%d", $4/1024}' /sys/devices/system/node/node1/meminfo)
  echo "$(date +%H:%M:%S) used=${used}M shared=${shared}M shmem=${shmem}M shmem_huge=${hp}M avail=${avail}M swap=${swapused}M node0_free=${n0}M node1_free=${n1}M" >> "$out"
  if [ "$swapused" -gt 6144 ] && [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then
    echo "$(date +%H:%M:%S) SWAP ${swapused}M: killing server $p" >> "$out"
    pkill -TERM -P "$p"; kill -TERM "$p"; sleep 20; pkill -KILL -P "$p" 2>/dev/null; kill -KILL "$p" 2>/dev/null
  fi
  sleep 5
done
