#!/usr/bin/env bash
# Can bounded replay be used for writing? On a 300-token sample it matched or
# beat full prefill on everything except 4-gram repetition, which rose 0.057 ->
# 0.106. Repetition is the failure mode that matters for long prose, and it only
# shows up at length, so generate 2000 tokens across three prompts with replay
# off and on and compare per-window repetition, not just the whole-text figure.
#
# Greedy decoding shreds V4.1 prose on its own (DSV41-PROSE-SAMPLING-20260911.md),
# so the gate that matters for writer workloads runs at the card's sampling:
# TEMP=1.0 TOP_P=0.95 REPS=2 samples each prompt twice per side and scores comma
# rate and length as well as repetition. Defaults stay greedy / one rep so the
# earlier runs are reproducible.
#
# Usage: OUT=... [TEMP=1.0 TOP_P=0.95 REPS=2] tools/verify-dsv41-swa-long.sh
set -Eeuo pipefail
root="$HOME/KTransformers"
out="${OUT:?set OUT}"
temp="${TEMP:-0}"; top_p="${TOP_P:-1}"; reps="${REPS:-1}"
exec >"$out" 2>&1
echo "start $(date +%H:%M:%S) temp=$temp top_p=$top_p reps=$reps"

mkbody() {  # $1 = prompt id
python3 - "$1" "$temp" "$top_p" <<'PY'
import json, sys
which = sys.argv[1]
temp, top_p = float(sys.argv[2]), float(sys.argv[3])
para = ("江戸時代の参勤交代は、諸大名に定期的な江戸在府と国元帰還を義務づけた制度であり、"
        "街道と宿場町の整備、藩財政の慢性的な逼迫、そして江戸の消費経済の膨張という"
        "三つの帰結をもたらした。")
ctx = "\n".join(f"{i+1}. {para}" for i in range(26))
tasks = {
  "essay": ctx + "\n\n上記を踏まえて、参勤交代が藩財政に与えた影響を論じてください。"
                 "常体で、章立てして、事実に即して詳しく。",
  "prose": ctx + "\n\n上記の制度の下で、国元へ帰る大名行列に随行する下級武士を主人公に、"
                 "三人称の小説の一場面を書いてください。地の文は常体、情景と身体感覚を"
                 "具体的に。説明的な要約はせず、場面として書くこと。",
  "list":  ctx + "\n\n上記の文章に出てくる三つの帰結それぞれについて、"
                 "具体的な事例を五つずつ挙げ、各事例に短い解説を付けてください。常体で。",
}
print(json.dumps({"model": "deepseek-v41-flash-engram-nvme",
                  "messages": [{"role": "user", "content": tasks[which]}],
                  "max_tokens": 2000, "temperature": temp, "top_p": top_p}))
PY
}

score() {  # $1 = tag, $2 = prompt id, $3 = rep
python3 - "$1" "$2" "$3" <<'PY'
import json, re, sys, hashlib
tag, pid, rep_i = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(f"/tmp/swal_{tag}_{pid}_{rep_i}.json"))
c = d["choices"][0]
t = c["message"]["content"]
open(f"/tmp/swal_{tag}_{pid}_{rep_i}.txt", "w").write(t)
u = d.get("usage", {})

def rep(s):
    g = [s[i:i+4] for i in range(len(s)-3)]
    return 1 - len(set(g))/max(1, len(g)) if g else 0.0

# Whole-text repetition hides a tail that degrades: score 400-char windows and
# report the worst, which is where a loop would start.
W = 400
wins = [rep(t[i:i+W]) for i in range(0, max(1, len(t)-W+1), W//2)]
kana = len(re.findall(r"[぀-ヿ]", t)) / max(1, len(t))
han  = len(re.findall(r"[一-鿿]", t)) / max(1, len(t))
# The same three failures check-dsv41-prose-commas.sh gates on: comma
# shredding, looping, stopping far short of the 2000 asked for.
comma = t.count("、") / max(1, len(t))
runs = [len(x) for x in t.split("、") if x.strip()]
mean_run = sum(runs) / max(1, len(runs))
got = u.get("completion_tokens") or 0
# A whole-text repetition above 0.5 is a loop even when no single 400-char
# window crosses 0.35 (the 09-13 list sample: 2000 tokens of a restated plan,
# rep_all 0.727, worst window 0.327).
if comma > 0.10 or mean_run < 8: verdict = "SHREDDED"
elif max(wins) > 0.35 or rep(t) > 0.5: verdict = "LOOPING"
elif got < 700 and c.get("finish_reason") == "stop": verdict = "SHORT"
else: verdict = "ok"
print(f"  {pid:6s} r{rep_i} completion={got} finish={c.get('finish_reason')} "
      f"chars={len(t)} sha={hashlib.sha256(t.encode()).hexdigest()[:12]}  [{verdict}]")
print(f"         rep_all={rep(t):.3f} rep_worst_window={max(wins):.3f} "
      f"windows={len(wins)} comma={comma:.3f} mean_run={mean_run:.1f} kana={kana:.3f} han={han:.3f}")
print(f"         tail: " + t[-110:].replace("\n", " / "))
PY
}

run() {  # $1 = 0|1, $2 = tag
  # stop only our own previous server, by pid file (never pkill by name)
  own_pid=$(cat "$root/logs/sglang-dsv41-engram/current/server.pid" 2>/dev/null || true)
  if [ -n "$own_pid" ] && kill -0 "$own_pid" 2>/dev/null; then
    pkill -TERM -P "$own_pid" 2>/dev/null || true; kill -TERM "$own_pid" 2>/dev/null || true
    sleep 20
    pkill -KILL -P "$own_pid" 2>/dev/null || true; kill -KILL "$own_pid" 2>/dev/null || true
  fi
  local clear=0
  for i in $(seq 1 40); do
    u0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
    u1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)
    [ "${u0:-9999}" -lt 400 ] && [ "${u1:-9999}" -lt 100 ] && { clear=1; break; }
    sleep 5
  done
  # Same refusal as the order harness: never launch beside another GPU job.
  if [ "$clear" != 1 ]; then
    echo "DIED gpu busy at $(date +%H:%M:%S): $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd/ -)MiB"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
    return 2
  fi
  # AB_EXTRA_ARGS: instead of bounded replay, the "on" side launches with
  # these SGLANG_EXTRA_ARGS (e.g. an --init-expert-location map) and replay
  # stays off on both sides — the same gate for any A/B of launch flags.
  local swa="$1" extra="${SGLANG_EXTRA_ARGS:-}"
  if [ -n "${AB_EXTRA_ARGS:-}" ]; then
    swa=0; [ "$1" = 1 ] && extra="$extra $AB_EXTRA_ARGS"
  fi
  echo "=== $2 (SGLANG_SWA_BOUNDED_REPLAY=$swa extra='$extra') launch $(date +%H:%M:%S)"
  SGLANG_SWA_BOUNDED_REPLAY="$swa" SGLANG_EXTRA_ARGS="$extra" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$root/tools/start-dsv41-engram-nvme.sh" --background >/dev/null
  local L="$root/logs/sglang-dsv41-engram/current/server.log"
  local P=$(cat "$root/logs/sglang-dsv41-engram/current/server.pid")
  for _ in $(seq 1 80); do
    sleep 15
    grep -q "Uvicorn running" "$L" && break
    kill -0 "$P" 2>/dev/null || { echo DIED; tail -15 "$L"; return 1; }
  done
  grep -oE "enable_decoder_swa_bounded_replay.: [A-Za-z]+" "$L" | head -1
  local r
  for pid in essay prose list; do
    for r in $(seq 1 "$reps"); do
      local t0=$(date +%s)
      curl -s -m 3600 http://127.0.0.1:8080/v1/chat/completions \
        -H "Content-Type: application/json" -d "$(mkbody "$pid")" \
        -o "/tmp/swal_$2_${pid}_$r.json"
      echo "  wall=$(( $(date +%s) - t0 ))s"
      score "$2" "$pid" "$r"
    done
  done
}

run 0 off
run 1 on
P=$(cat "$root/logs/sglang-dsv41-engram/current/server.pid" 2>/dev/null || true)
[ -n "$P" ] && { pkill -TERM -P "$P" 2>/dev/null || true; kill -TERM "$P" 2>/dev/null || true; }

echo "=== verdict"
python3 - "$reps" <<'PY'
import re, sys
reps = int(sys.argv[1])
def rep(s):
    g = [s[i:i+4] for i in range(len(s)-3)]
    return 1 - len(set(g))/max(1, len(g)) if g else 0.0
W = 400
def stats(t):
    wins = [rep(t[i:i+W]) for i in range(0, max(1, len(t)-W+1), W//2)]
    runs = [len(x) for x in t.split("、") if x.strip()]
    return dict(rep=rep(t), worst=max(wins), chars=len(t),
                comma=t.count("、") / max(1, len(t)),
                mean_run=sum(runs) / max(1, len(runs)))
# Under sampling each rep is a different text, so a side is judged on its
# worst rep: one shredded or looping sample at the card's sampling is a
# failure a writer would hit.
for pid in ("essay", "prose", "list"):
    sides = {}
    for tag in ("off", "on"):
        rows = []
        for r in range(1, reps + 1):
            try:
                rows.append(stats(open(f"/tmp/swal_{tag}_{pid}_{r}.txt").read()))
            except FileNotFoundError:
                pass
        if not rows:
            sides = None; break
        sides[tag] = dict(worst=max(x["worst"] for x in rows),
                          rep=max(x["rep"] for x in rows),
                          comma=max(x["comma"] for x in rows),
                          mean_run=min(x["mean_run"] for x in rows),
                          chars=min(x["chars"] for x in rows), n=len(rows))
    if not sides:
        print(f"  {pid}: missing a side"); continue
    a, b = sides["off"], sides["on"]
    bad = (b["worst"] - a["worst"] > 0.10 or b["worst"] > 0.35
           or b["rep"] > 0.5 or b["comma"] > 0.10 or b["mean_run"] < 8)
    print(f"  {pid:6s} n={a['n']}/{b['n']} rep_max {a['rep']:.3f}->{b['rep']:.3f}  "
          f"worst_window {a['worst']:.3f}->{b['worst']:.3f}  "
          f"comma_max {a['comma']:.3f}->{b['comma']:.3f}  "
          f"min_chars {a['chars']}->{b['chars']}  [{'WORSE' if bad else 'ok'}]")
print("  threshold: worst 400-char window above 0.35, or +0.10 over baseline,")
print("  or any rep with comma rate above 0.10 / mean run under 8, blocks")
print("  adoption for prose.")
PY
echo "done $(date +%H:%M:%S)"
