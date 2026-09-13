#!/usr/bin/env bash
# Does socket 1's bandwidth drop need this stack at all? Run a different model
# through a different engine with no GPU, no SGLang and no kt-kernel:
# ik_llama.cpp pinned to socket 1, decoding back-to-back for DURATION seconds,
# logging each request's decode tok/s (bandwidth-bound, so a halving shows
# directly), then the node-1 read bandwidth with the server idle.
# Usage: OUT=... [DURATION=720] [THREADS=32] [NODE=1] [MODEL=...] [SERVER=...] tools/probe-node1-ikllama.sh
set -Eeuo pipefail
root="$HOME/KTransformers"
out="${OUT:?set OUT}"
duration="${DURATION:-720}"
threads="${THREADS:-32}"
node="${NODE:-1}"
# Default: a 7.4 GB dense Q6_K 9B from the archive — every decoded token
# streams the whole model, so decode tok/s is a direct bandwidth meter.
model="${MODEL:-$root/models/gguf-probe/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q6_K.gguf}"
server="${SERVER:-$root/runtime/ik-llama-build/bin/llama-server}"
port=8092
exec >"$out" 2>&1
echo "start $(date +%H:%M:%S) node=$node threads=$threads duration=${duration}s model=$(basename "$model")"
echo "before: node$node $(numactl --cpunodebind=$node --membind=$node "$root/tools/membw-probe" 8 512 3)"
numactl --cpunodebind=$node --membind=$node "$server" \
  -m "$model" -t "$threads" -c 8192 -rtr --host 127.0.0.1 --port $port >"$out.server.log" 2>&1 &
S=$!
for _ in $(seq 1 120); do
  sleep 5
  curl -s -m 5 "http://127.0.0.1:$port/health" 2>/dev/null | grep -q '"ok"' && break
  kill -0 "$S" 2>/dev/null || { echo "DIED server"; tail -5 "$out.server.log"; exit 1; }
done
echo "UP $(date +%H:%M:%S)"
body='{"prompt":"参勤交代について、その制度の成り立ちと藩財政への影響を、具体例を挙げながら詳しく論じなさい。","n_predict":256,"temperature":0.7,"cache_prompt":false}'
t0=$(date +%s); i=0
while [ $(( $(date +%s) - t0 )) -lt "$duration" ]; do
  i=$((i+1))
  r=$(curl -s -m 600 "http://127.0.0.1:$port/completion" -H "Content-Type: application/json" -d "$body")
  echo "  $(date +%H:%M:%S) r$i $(printf '%s' "$r" | python3 -c 'import json,sys
d=json.load(sys.stdin); t=d.get("timings",{})
print("predicted=%s decode=%.2f tok/s prompt=%.1f tok/s" % (t.get("predicted_n"), t.get("predicted_per_second",0), t.get("prompt_per_second",0)))' 2>/dev/null || echo "parse-error ${r:0:80}")"
done
echo "after: node$node $(numactl --cpunodebind=$node --membind=$node "$root/tools/membw-probe" 8 512 3)"
echo "after: node$((1-node)) $(numactl --cpunodebind=$((1-node)) --membind=$((1-node)) "$root/tools/membw-probe" 8 512 3)"
kill -TERM "$S" 2>/dev/null || true
echo "done $(date +%H:%M:%S)"
