#!/usr/bin/env bash
# Sample the board's BMC sensors over Redfish beside a run. The H12D-16D BMC
# (OpenBMC/bmcweb, <bmc-ip>, enabled 09-12) exposes one CPU temperature,
# the Vcore and Vsoc VRM temperatures, inlet/outlet air, CPU power and the
# rails — no per-DIMM temperatures. Which socket "CPU_*" means is settled by
# loading one socket at a time and watching which run moves it.
# Usage: OUT=... [INTERVAL=20] [BMC=<bmc-ip>] [CRED=admin:admin] tools/watch-bmc-sensors.sh
set -Eeuo pipefail
out="${OUT:?set OUT}"
interval="${INTERVAL:-20}"
bmc="${BMC:-<bmc-ip>}"
cred="${CRED:-admin:admin}"
base="https://$bmc/redfish/v1/Chassis/1/Sensors"
exec >>"$out" 2>&1
echo "# t cpu_temp vcore_vr_t vsoc_vr_t inlet outlet cpu_power_W vddq_V vcore_V p12v_V"
read_one() { curl -k -s -m 8 -u "$cred" "$base/$1" | sed -n 's/.*"Reading": \([0-9.]*\).*/\1/p' | head -n 1; }
while true; do
  line="$(date +%H:%M:%S)"
  for s in temperature_CPU_Temp temperature_CPU_Vcore_VR_T temperature_CPU_Vsoc_VR_T temperature_Inlet_Temp temperature_Outlet_Temp power_CPU_Power voltage_VDDQ_SENSOR voltage_VCORE_SENSOR voltage_P12V_SENSOR; do
    v=$(read_one "$s" || true); line="$line ${v:-NA}"
  done
  echo "$line"
  sleep "$interval"
done
