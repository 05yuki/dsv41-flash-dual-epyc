#!/usr/bin/env bash
# Torch-profile decode on the V4.1 server: where do the ~63 ms per token go?
# Launches with KT_DEFERRED_EXPERTS=0 (the arithmetic the checkpoint specifies;
# deferral 4 is not used for writer work), warms up with one generation, then
# profiles one 300-token generation through SGLang's /start_profile and
# summarises the trace. Everything in the CUDA graph is one launch per step,
# so the split this gives is: graph replay (GPU + CPU experts + engram host
# callbacks, all serialised on the stream) versus everything the scheduler
# does around it. Run detached: OUT=... setsid nohup ...
set -Eeuo pipefail
root="$HOME/KTransformers"
out="${OUT:?set OUT}"
ntok="${NTOK:-300}"
pdir="${PROFILE_DIR:-$root/logs/torch-profile-decode-$(date +%H%M%S)}"
exec >"$out" 2>&1
mkdir -p "$pdir"
echo "start $(date +%H:%M:%S) deferral=${KT_DEFERRED_EXPERTS:-0} ntok=$ntok"
KT_DEFERRED_EXPERTS="${KT_DEFERRED_EXPERTS:-0}" "$root/tools/start-dsv41-engram-nvme.sh" --background >/dev/null
L="$root/logs/sglang-dsv41-engram/current/server.log"
P=$(cat "$root/logs/sglang-dsv41-engram/current/server.pid")
for _ in $(seq 1 80); do sleep 15; grep -q "Uvicorn running" "$L" && break; kill -0 "$P" 2>/dev/null || { echo DIED; tail -n 5 "$L"; exit 1; }; done
echo "UP $(date +%H:%M:%S)"
body=$(python3 - "$ntok" <<'PY'
import json, sys
text = "参勤交代で国元へ帰る大名行列に随行する下級武士を主人公に、三人称の小説の一場面を書いてください。地の文は常体、情景と身体感覚を具体的に。"
print(json.dumps({"model":"deepseek-v41-flash-engram-nvme","messages":[{"role":"user","content":text}],
                  "max_tokens":int(sys.argv[1]),"temperature":1.0,"top_p":0.95,"ignore_eos":True}))
PY
)
gen() {  # $1 = label
  local t0 t1 resp B
  B=$(wc -l <"$L"); t0=$(date +%s.%N)
  resp=$(curl -s -m 3600 http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d "$body")
  t1=$(date +%s.%N)
  echo "$1: $(python3 -c "import json,sys;u=json.loads(sys.argv[1])['usage'];print('prompt',u['prompt_tokens'],'completion',u['completion_tokens'])" "$resp") wall=$(python3 -c "print(f'{$t1-$t0:.1f}')")s"
  tail -n +"$B" "$L" | grep -E "Decode batch" | grep -oE "gen throughput \(token/s\): [0-9.]+" | awk '{s+=$NF; n++} END{if(n) printf "  decode log: mean %.2f tok/s over %d batches\n", s/n, n}'
}
gen warmup
sleep 5
curl -s -m 60 -X POST http://127.0.0.1:8080/start_profile -H "Content-Type: application/json" \
  -d "{\"output_dir\":\"$pdir\",\"activities\":[\"CPU\",\"GPU\"],\"with_stack\":false,\"record_shapes\":false}"; echo
gen profiled
curl -s -m 600 -X POST http://127.0.0.1:8080/stop_profile; echo
sleep 20
ls -la "$pdir" | tail -n 4
echo "== summary (divide totals by $ntok steps)"
# head closes the pipe early; under pipefail that would abort the script
# before the server is stopped (it did, 09-13 01:40: a server left on the GPUs)
"$root/venv-dsv41/bin/python" "$root/tools/summarize_torch_trace.py" "$pdir" 2>&1 | head -n 90 || true
pkill -TERM -P "$P" 2>/dev/null || true; kill -TERM "$P" 2>/dev/null || true; sleep 20
pkill -KILL -P "$P" 2>/dev/null || true; kill -KILL "$P" 2>/dev/null || true
echo "done $(date +%H:%M:%S)"
