#!/usr/bin/env bash
# Torch-profile one prefill on the V4.1 server through SGLang's /start_profile
# endpoint (no ptrace needed). Sends the same ~1.1K-token prompt twice inside
# one profiling window so the per-request first-chunk cost can be compared,
# then summarises the trace. Run detached: OUT=... setsid nohup ...
set -Eeuo pipefail
root="$HOME/KTransformers"
out="${OUT:?set OUT}"
pdir="${PROFILE_DIR:-$root/logs/torch-profile-$(date +%H%M%S)}"
exec >"$out" 2>&1
mkdir -p "$pdir"
echo "start $(date +%H:%M:%S) chunk=${SGLANG_CHUNKED_PREFILL_SIZE:-2048}"
KT_TASKQUEUE_TIMING="${KT_TASKQUEUE_TIMING:-40}" "$root/tools/start-dsv41-engram-nvme.sh" --background >/dev/null
L="$root/logs/sglang-dsv41-engram/current/server.log"
P=$(cat "$root/logs/sglang-dsv41-engram/current/server.pid")
for _ in $(seq 1 80); do sleep 15; grep -q "Uvicorn running" "$L" && break; kill -0 "$P" 2>/dev/null || { echo DIED; exit 1; }; done
echo "UP $(date +%H:%M:%S)"
para="江戸時代の参勤交代は、諸大名に定期的な江戸在府と国元帰還を義務づけた制度であり、街道と宿場町の整備、藩財政の慢性的な逼迫、そして江戸の消費経済の膨張という三つの帰結をもたらした。"
body=$(python3 - "$para" <<'PY'
import json, sys
text = "以下の文章を要約してください。\n\n" + "\n".join(sys.argv[1] for _ in range(11))
print(json.dumps({"model":"deepseek-v41-flash-engram-nvme","messages":[{"role":"user","content":text}],"max_tokens":4,"temperature":0}))
PY
)
curl -s -m 60 -X POST http://127.0.0.1:8080/start_profile -H "Content-Type: application/json" \
  -d "{\"output_dir\":\"$pdir\",\"activities\":[\"CPU\",\"GPU\"],\"with_stack\":false,\"record_shapes\":false}"; echo
for r in 1 2; do
  t0=$(date +%s.%N); B=$(wc -l <"$L")
  resp=$(curl -s -m 3600 http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d "$body")
  t1=$(date +%s.%N)
  echo "request $r: prompt_tokens=$(python3 -c "import json,sys;print(json.loads(sys.argv[1])['usage']['prompt_tokens'])" "$resp") wall=$(python3 -c "print(f'{$t1-$t0:.1f}')")s"
  tail -n +"$B" "$L" | grep -E "Prefill batch|Decode batch" | grep -oE "^\[[0-9-]+ [0-9:]+|#new-token: [0-9]+" | paste - - | head -4
done
curl -s -m 600 -X POST http://127.0.0.1:8080/stop_profile; echo
sleep 20
ls -la "$pdir" | tail -5
echo "== summary"
"$root/venv-dsv41/bin/python" "$root/tools/summarize_torch_trace.py" "$pdir" 2>&1 | head -70 || true
pkill -TERM -P "$P" 2>/dev/null || true; kill -TERM "$P" 2>/dev/null || true; sleep 20
pkill -KILL -P "$P" 2>/dev/null || true; kill -KILL "$P" 2>/dev/null || true
echo "done $(date +%H:%M:%S)"
