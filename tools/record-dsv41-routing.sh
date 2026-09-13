#!/usr/bin/env bash
# Record which experts V4.1 routes to on writer-shaped prompts, so the hot ones
# can be placed on the GPU (tools/build-dsv41-expert-placement.py). GPU experts
# are physical ids 0..N-1 (kt_ep_wrapper.mask_cpu_expert_ids), topk emits
# physical ids through expert_location_dispatch, and the KT CPU side loads by
# physical_to_logical_map_cpu, so a static --init-expert-location map moves
# experts between GPU and CPU without touching weights.
#
# Launches the server with SGLang's expert distribution recorder in "stat"
# mode, generates from the three prose scenes plus a long argument prompt at
# the writer sampling (t=1.0 / top_p=0.95), dumps the per-layer logical counts,
# and stops. Refuses to run beside another GPU job. Run detached: OUT=...
set -Eeuo pipefail
root="$HOME/KTransformers"
out="${OUT:?set OUT}"
rec_dir="${REC_DIR:-$root/logs/expert-routing-$(date +%Y%m%d-%H%M%S)}"
max_tokens="${MAX_TOKENS:-1500}"
exec >"$out" 2>&1
mkdir -p "$rec_dir"
echo "start $(date +%H:%M:%S) rec_dir=$rec_dir max_tokens=$max_tokens"
u0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
u1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)
if [ "${u0:-9999}" -ge 400 ] || [ "${u1:-9999}" -ge 100 ]; then
  echo "DIED gpu busy: ${u0}/${u1} MiB"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader; exit 2
fi
export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR="$rec_dir"
export SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS:-} --expert-distribution-recorder-mode stat"
# SGLang's recorder never sees the V4 router's ids (its dump was all zeros on
# 09-13); the KT wrapper's own histogram (KT_ROUTING_DUMP, kt_ep_wrapper.py)
# is what build-dsv41-expert-placement.py reads.
mkdir -p "$rec_dir"
export KT_ROUTING_DUMP="$rec_dir/kt-routing"
"$root/tools/start-dsv41-engram-nvme.sh" --background >/dev/null
L="$root/logs/sglang-dsv41-engram/current/server.log"
P=$(cat "$root/logs/sglang-dsv41-engram/current/server.pid")
for _ in $(seq 1 80); do sleep 15; grep -q "Uvicorn running" "$L" && break; kill -0 "$P" 2>/dev/null || { echo DIED; tail -15 "$L"; exit 1; }; done
echo "UP $(date +%H:%M:%S)"

body() {  # $1 = prompt id
python3 - "$1" "$max_tokens" <<'PY'
import json, sys
pid, max_tokens = sys.argv[1], int(sys.argv[2])
prompts = {
  "march":  "参勤交代で国元へ帰る大名行列に随行する下級武士を主人公に、三人称の小説の一場面を"
            "書いてください。地の文は常体、情景と身体感覚を具体的に。説明的な要約はせず、"
            "場面として書くこと。",
  "inn":    "参勤交代の行列が泊まる宿場の旅籠で、旅籠の女将と若い武士が交わす短い会話を軸に、"
            "三人称の小説の一場面を書いてください。地の文は常体、会話を多めに。",
  "ledger": "藩の勘定方の武士が、参勤交代の費用を帳面につけながら藩の窮状を思う場面を、"
            "三人称の小説として書いてください。地の文は常体、内心の描写を厚く。",
  "essay":  "参勤交代が江戸の消費経済と街道整備に与えた影響について、事実に即して常体で論じて"
            "ください。見出しを付け、各節は具体例を一つ以上含めること。",
}
print(json.dumps({"model": "deepseek-v41-flash-engram-nvme",
                  "messages": [{"role": "user", "content": prompts[pid]}],
                  "max_tokens": max_tokens, "temperature": 1.0, "top_p": 0.95}))
PY
}

curl -s -m 60 -X POST http://127.0.0.1:8080/start_expert_distribution_record >/dev/null; echo "recording from $(date +%H:%M:%S)"
for pid in march inn ledger essay; do
  t0=$(date +%s)
  resp=$(curl -s -m 3600 http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d "$(body "$pid")")
  t1=$(date +%s)
  echo "--- $pid $(python3 -c "import json,sys;u=json.loads(sys.argv[1]).get('usage',{});print('prompt=%s completion=%s'%(u.get('prompt_tokens'),u.get('completion_tokens')))" "$resp" 2>/dev/null || echo ERR) wall=$((t1-t0))s"
done
curl -s -m 600 -X POST http://127.0.0.1:8080/dump_expert_distribution_record >/dev/null; echo "dumped at $(date +%H:%M:%S)"
sleep 35  # the KT histogram is saved every 30 s
ls -la "$rec_dir"
pkill -TERM -P "$P" 2>/dev/null || true; kill -TERM "$P" 2>/dev/null || true; sleep 20
pkill -KILL -P "$P" 2>/dev/null || true; kill -KILL "$P" 2>/dev/null || true
echo "done $(date +%H:%M:%S)"
