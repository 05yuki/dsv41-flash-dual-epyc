#!/usr/bin/env bash
# Sample GPU clocks / temperature / throttle reasons / power and CPU busy-core
# clocks / package temperature every INTERVAL seconds into OUT, for running
# alongside a prefill run. The order harness only samples clocks at the idle
# instant before and after a request; this catches what happens inside one.
set -Eeuo pipefail
out="${OUT:?set OUT}"
interval="${INTERVAL:-2}"
exec >"$out" 2>&1
# cpu index -> NUMA node, so a pool going idle shows as one socket's count
declare -a node
while IFS=, read -r c n; do node[$c]=$n; done < <(lscpu -p=CPU,NODE | grep -v "^#")
echo "# t gpu0_sm gpu0_mem gpu0_temp gpu0_pw gpu0_throttle | gpu1_sm gpu1_temp gpu1_pw gpu1_throttle | cpu_busy_n busy_node0 busy_node1 cpu_busy_mhz_avg cpu_max_mhz tctl_socket0/socket1 | numa_hint_faults/pte_updates/migrated per sample | pkg0_W pkg1_W"
# NUMA balancing activity per sample: hinting faults, PTE unmaps, migrations.
numac() { awk '/^(numa_hint_faults|numa_pte_updates|numa_pages_migrated) /{printf "%s ", $2} END{print ""}' /proc/vmstat; }
read -r hf0 pu0 mg0 < <(numac)
# package power from RAPL (energy_uj is world-readable on this host)
rapl() { cat /sys/class/powercap/intel-rapl:0/energy_uj /sys/class/powercap/intel-rapl:1/energy_uj 2>/dev/null | paste -sd' ' -; echo; }
read -r e0a e1a < <(rapl); ta=$(date +%s%N)
while true; do
  g=$(nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,clocks_throttle_reasons.active --format=csv,noheader,nounits 2>/dev/null | paste -sd'|' -)
  # busy cores: those above 60% in a 0.5 s window, from /proc/stat deltas
  read -r -a a < <(awk '/^cpu[0-9]/{print $2+$3+$4+$6+$7+$8, $5}' /proc/stat | paste -sd' ' -)
  sleep 0.5
  read -r -a b < <(awk '/^cpu[0-9]/{print $2+$3+$4+$6+$7+$8, $5}' /proc/stat | paste -sd' ' -)
  mapfile -t mhz < <(awk '/^cpu MHz/{print $4}' /proc/cpuinfo)
  busy=0; sum=0; max=0; n0=0; n1=0
  for ((i=0; i<${#mhz[@]}; i++)); do
    db=$(( ${b[2*i]} - ${a[2*i]} )); di=$(( ${b[2*i+1]} - ${a[2*i+1]} ))
    tot=$(( db + di )); [ "$tot" -gt 0 ] || continue
    if [ $(( db * 100 / tot )) -ge 60 ]; then
      m=${mhz[$i]%.*}; busy=$((busy+1)); sum=$((sum+m)); if [ "$m" -gt "$max" ]; then max=$m; fi
      if [ "${node[$i]:-0}" = 0 ]; then n0=$((n0+1)); else n1=$((n1+1)); fi
    fi
  done
  avg=0; if [ "$busy" -gt 0 ]; then avg=$((sum/busy)); fi
  tctl=$(for h in /sys/class/hwmon/hwmon*; do if [ "$(cat "$h/name")" = k10temp ]; then printf "%s/" "$(( $(cat "$h/temp1_input") / 1000 ))"; fi; done; true)
  read -r hf1 pu1 mg1 < <(numac)
  read -r e0b e1b < <(rapl); tb=$(date +%s%N)
  dt=$(( (tb - ta) / 1000 )); [ "$dt" -gt 0 ] || dt=1
  echo "$(date +%H:%M:%S) $g | $busy $n0 $n1 $avg $max ${tctl:-?} | $((hf1-hf0)) $((pu1-pu0)) $((mg1-mg0)) | $(( (e0b - e0a) / dt )) $(( (e1b - e1a) / dt ))"
  hf0=$hf1; pu0=$pu1; mg0=$mg1; e0a=$e0b; e1a=$e1b; ta=$tb
  sleep "$interval"
done
