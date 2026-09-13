"""Aggregate a torch-profiler chrome trace: where did the wall time go?

Prints the longest single events (the 7-minute culprit should be one or a few
huge spans), then total time per op name for CPU ops and CUDA kernels.
"""
import gzip, json, sys, collections, pathlib

d = pathlib.Path(sys.argv[1])
files = sorted(d.glob("**/*.json*"), key=lambda p: p.stat().st_size, reverse=True)
if not files:
    sys.exit("no trace files under %s" % d)
for f in files[:2]:
    print("== %s (%.1f MB)" % (f.name, f.stat().st_size / 1e6))
    opener = gzip.open if f.suffix == ".gz" else open
    with opener(f, "rt") as fh:
        data = json.load(fh)
    ev = data["traceEvents"] if isinstance(data, dict) else data
    ev = [e for e in ev if e.get("ph") == "X" and "dur" in e]
    t0 = min(e["ts"] for e in ev) if ev else 0
    print("   %d complete events, span %.1f s" % (len(ev), (max(e["ts"] + e["dur"] for e in ev) - t0) / 1e6))
    print("-- longest 15 single events (name, start s, dur s, cat)")
    for e in sorted(ev, key=lambda e: -e["dur"])[:15]:
        print("   %-60s %8.1f %8.2f %s" % (e["name"][:60], (e["ts"] - t0) / 1e6, e["dur"] / 1e6, e.get("cat", "")))
    for cat in ("cpu_op", "kernel", "cuda_runtime", "user_annotation"):
        tot = collections.Counter(); cnt = collections.Counter()
        for e in ev:
            if e.get("cat") == cat:
                tot[e["name"]] += e["dur"]; cnt[e["name"]] += 1
        if not tot:
            continue
        print("-- top %s by total time" % cat)
        for name, t in tot.most_common(12):
            print("   %-60s %8.2f s  x%d" % (name[:60], t / 1e6, cnt[name]))
