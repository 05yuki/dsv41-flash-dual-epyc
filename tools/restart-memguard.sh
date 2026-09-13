#!/usr/bin/env bash
# Restart the memory guard as a single instance. Runs from a file so the
# caller's command line never contains the guard's name (pkill/pgrep -f
# otherwise matches the caller itself).
for p in $(pgrep -f "memguard-dsv41.sh"); do [ "$p" != "$$" ] && kill "$p" 2>/dev/null; done
sleep 1
cd "$HOME/KTransformers"
nohup bash tools/memguard-dsv41.sh /tmp/memguard.txt >/tmp/memguard.err 2>&1 &
sleep 1
echo "guards: $(pgrep -f "memguard-dsv41.sh" | grep -v "^$$\$" | wc -l)"
