# The V4.1 prefill decay is socket 1's memory bandwidth halving (2026-09-12)

Continues `DSV41-PREFILL-SCALING-20260911.md`, which ended with the node-0
expert pool idling ~60% of the time after 6-8K back-to-back prompt tokens and
nothing on the memory side moving. Today's runs locate it: **the node-1 pool is
the slow one — socket 1's local DRAM read bandwidth drops to half and stays
there for minutes — and the node-0 pool is merely waiting for it.** It is
measurable with a plain user-space probe while the server sits idle, so it is
hardware or firmware state, not anything in the process.

## What was run

All on `tools/start-dsv41-engram-nvme.sh` defaults (64 threads, two pools,
7 GPU experts, deferral 4, chunk 2048) unless noted, 701-token prompts
back-to-back through `tools/measure-dsv41-prefill-order.sh`.

| run | config | clean | tripped at | tripped |
|---|---|---|---|---|
| `700x14-pool1` | `KT_THREADPOOL_COUNT=1 KT_CPU_THREADS=32` (one pool on node 0) | — | — | **OOM-killed at layer 30 of 40**: `CONSTRAINT_MEMORY_POLICY nodemask=0`, 254 GB anon on a 257 GB node. 306 GiB of experts cannot live on one socket. Same for a node-1 pool. Dead end. |
| `700x14-swap` | `KT_NUMA_NODES=1,0` (subpool 0 on node 1, subpool 1 on node 0) | 30-31 s | r10 | 52-56 s |
| `700x14-numa` | default, NUMA-balancing counters recorded | 25-28 s | r11 | 45-46 s |
| `700x12-keep` | default, server kept up after the run | 32 s | r8 | 59 s |
| `700x14-keepalive1` | default + a 2-thread stream on node 1 the whole time | 31-33 s | **r4-5** | 60-61 s |

`KT_NUMA_NODES` is a one-line passthrough added to the worktree's
`kt_ep_wrapper.py` (kt-kernel's `KTMoEWrapper` already takes `numa_nodes`).

## Which pool is slow

Two new samplers beside the runs: `tools/watch-dsv41-pools.sh` (CPU ticks per
kt-kernel subpool from `/proc/<TP0>/task/*/stat`) and
`tools/snap-dsv41-pool-sys.sh` (user/system split). In every tripped phase,
whichever subpool sits on **node 1** is at 100% user time on all 31 workers,
and the pool on node 0 has 20-31 of its 31 workers sleeping in `futex_do_wait`
(the pool workers spin 50 ms after a job, then sleep on their cv — so they have
been idle for tens of ms at least). Swapping the subpool-to-node map moved
nothing: the idle pool is the node-0 pool either way. System time is ~0% on
both pools throughout, NUMA hinting faults are 0-80 per 2.5 s sample across the
trip (`numa_hint_faults`, `numa_pte_updates` in the thermal log), direct
reclaim / compaction counters do not move. Kernel-side causes are out.

kt-kernel's `NumaJobDistributor::do_numa_job` hands the same job to every
subpool and spin-waits for all of them; each TP part computes every activated
expert over its own slice of the intermediate dimension, so the work is
symmetric. A node-0 pool that sleeps while the node-1 pool runs flat out means
**the node-1 pool takes twice as long for the same job**, at full core clock
(3200-3290 MHz on its busy cores in the tripped state, effective 3.3 GHz by
`tools/clock-probe`).

## Socket 1's memory bandwidth halves

`tools/membw-probe` (16 MiB-per-thread streaming reads, first-touched on the
probing thread, run under `numactl --cpunodebind=N --membind=N`) with the server
**idle** between requests, `tools/probe-dsv41-tripped.sh`:

| state | node 0, 8 threads | node 0, 1 thread | node 1, 8 threads | node 1, 1 thread |
|---|---|---|---|---|
| host idle, no server | 99.2 | 19.1 | 99.0 | 19.4 |
| tripped (14:29, right after a 59 s request) | 107.4 | 18.9 | **49.1** | 16.1 |
| after 300 s idle (14:34) | 91.8 | 18.8 | 97.2 | 19.1 |
| tripped again (15:16, keepalive run) | 92.3 | — | **51.6** | — |
| after 300 s idle (15:21) | 99.7 | 19.1 | 101.8 | 19.4 |

The requests bracket it: 59 s / 59 s before the idle, 30 s / 31 s after. Cross-
socket reads (`cpu0->mem1`, `cpu1->mem0`) are 50-54 GB/s in both states, i.e.
capped by xGMI, unchanged. So the drop is specifically **socket 1's local
memory path**, and only when the model has been running back-to-back.

## What does not reproduce it without the model

- 16-thread sequential streaming on node 1 at 116 GB/s for 7 minutes: flat.
- 32-thread random 64-byte reads on node 1 (~97 GB/s of lines, maximal row
  activations) for 5 minutes, sequential check each minute: flat at 101-104.
- GPU1 at 300 W / 84 °C for 7 minutes with node 1 streaming: 117 → 111 GB/s,
  a 5% drift, not a halving. During the actual runs the GPUs draw 46-55 W at
  47-52 °C anyway.
- Socket temperatures (Tctl) are 64/44 °C during runs, socket 1 the cooler.

And an extra 2-thread stream on node 1 *during* a run (`keepalive1`) trips it
at r4-5 instead of r8-11, so the trigger accumulates with node-1 memory
activity — but only the model's activity counts, not a probe's.

## Socket power and the socket-heat test

RAPL `energy_uj` is world-readable here, so `tools/watch-dsv41-thermal.sh` now
logs package watts. During a clean 701-token prefill both sockets sit at
129-134 W with 30-32 busy cores each; in the tripped phase socket 1 *drops* to
114-119 W with 32 cores busy (stalled on memory, drawing less), socket 0 to
107-117 W with 13-17 busy. No power cap is in play (7452 TDP 155 W). A
32-thread fp32 matmul on socket 1 holding it at 127-128 W (Tctl 48-49 °C) for
six minutes leaves node 1 at 99.9 GB/s afterwards — socket heat alone is not
the trigger either, and neither is socket heat plus DRAM traffic: the same
matmul on 24 cores with 8 cores of random reads beside it (130 W, 6 min)
leaves node 1 at 93-96 GB/s.

## Recovery takes 60 s, not 300

`tools/probe-dsv41-tripped.sh` with `RECOVER="60 60 60 60 60"`: after two 59-60 s
requests, one 60 s idle brings node 1 from 49.2 to 102.0 GB/s and the next
request to 30 s. Yesterday's 300 s was the first interval tried, not the
threshold; 15 s is known not to be enough.

## What bounded replay actually does

`--enable-decoder-swa-bounded-replay` is the model card's **SWA Bounded
Replay** ("reconstructs missing SWA KV states by replaying only the most
recent n_win tokens", part of the V4.1 design, not an approximation added
here). In `models/deepseek_v4.py` it sets `late_layer_start =
max(kv_source_layer_ids)+1` = 21 and runs layers 21-39 over each request's
last `sliding_window` = 128 extend tokens only. Nineteen of forty layers' worth of CPU expert work shrinks
from 701 (or 2048) rows to 128, which is why the 701-token request goes from
30 s to 15-16 s and why the 09-11 note attributing the cost to "out-of-window
KV reads" was wrong: the saving is in the MoE, on the CPU. It does not avoid
the trip either: 28 such requests back-to-back trip at **r19** (15-16 s → 28 s,
node 1 at 55 GB/s afterwards), i.e. after ~4.7 min of continuous load, against
r8-11 (4-5.5 min) for the default path. **The trigger tracks time under the
model's load, roughly 4.5-5 minutes continuous, not tokens** (12.6K vs 5.6-7.7K).
Yesterday's "no decay with replay" was four requests, too few.

## Single rank: the second rank and the GPU's socket are not the trigger

The launcher's device order puts TP0 on nvidia-smi's GPU 1 at 81:00.0 — the
card on **socket 1** — and TP1 on 21:00.0 (socket 0), so in every run above
the busy rank's GPU<->host traffic for the CPU expert path crossed xGMI into
socket 1's IOD. Two single-rank runs (`KT_TP=1 KT_NUMA_NODE_LIST=0
KT_GPU_EXPERTS=1 FRAC=0.92 SGLANG_CONTEXT_LENGTH=8192`; zero GPU experts
asserts in FlashInfer's MoE kernel, two do not fit at 0.85) settle both
readings:

| run | card | clean | tripped at | after |
|---|---|---|---|---|
| `700x20-tp1` | 81:00.0 (socket 1) | 30 s | r13 | 58-59 s, node 1 at 53.7 GB/s |
| `700x20-tp1-gpu21` | 21:00.0 (socket 0) | 30-31 s | r10-11 | 59 s, node 1 at 50.3 GB/s |

No second rank, either card, same trip, same socket. The trigger is on the
CPU side of socket 1 — the kt-kernel pool's own work — or on whatever the
scheduler on node 0 does with it. The launcher now takes `KT_GPU_UUIDS`.

## The thread-count curve in the tripped state

Node 1, `tools/membw-probe N 512 3`, right after `700x20-tp1`:

| threads | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| tripped | 15.0 | 28.4 | 42.8 | 53.7 | 57.8 |
| clean | 19.4 | — | — | 100 | 116 |

A ceiling at half, reached from 8 threads, with the single thread down 23%.
That is a clock (memory or fabric P-state), not a latency change.

## It is the socket, not the stack: ik_llama.cpp reproduces it on socket 1 only

`tools/probe-node1-ikllama.sh`: a dense 9B Q6_K GGUF (7.4 GB, from the
archive) through ik_llama.cpp `-rtr`, 32 threads, pinned with `numactl` to one
socket, no GPU, no SGLang, no kt-kernel, 256-token completions back-to-back
for 12 minutes. Every decoded token streams the whole model, so decode tok/s
is a bandwidth meter.

| socket | r1-r23 | then | node bandwidth after |
|---|---|---|---|
| **1** | 16-17 s per 256 tokens (15.7 tok/s) | **r24 30 s, r25-r30 38-39 s (2.2x)**, ~6.5 min in | node 1 **47.9**, node 0 98.8 |
| 0 | 15.68-15.75 tok/s for all 44 requests, 12 min, 140 W | flat | node 0 101.5, node 1 100.7 |

Socket 1 halves under a different model and a different engine; socket 0 does
not under the identical load. This is a property of socket 1 on this host —
firmware (the SMU's memory/fabric P-state management for that socket) or the
board/DIMM population behind it — and not of KTransformers, SGLang or the
V4.1 configuration. Why no synthetic load trips it and every real model does
remains unexplained and no longer matters: the 12-minute ik_llama run on
socket 1 is the test bench for whatever is tried next, at a fraction of the
cost of a V4.1 launch.

## BIOS: APBDIS=1 / Fixed SOC Pstate P0 / DF Cstates off — no change

Set on 09-12 evening (AMD CBS → NBIO Common Options → SMU Common Options:
Determinism Manual/Performance, APBDIS 1, DF Cstates Disabled, Fixed SOC
Pstate P0, HSMP Support Enabled; `min_free_kbytes` survived the reboot). The
same 12-minute ik_llama run on socket 1: 16.1 tok/s (up from 15.7, the fixed
P0) through r27, then **r28 14.6, r29-r36 7.13 tok/s**, node 1 at 54.0 GB/s
after, socket 1 power 136 → 113 W across the trip, Tctl 45-47 °C. The APB
governor is not it: with the fabric P-state pinned, socket 1's memory path
still halves after ~7.5 minutes. What remains is a *protection* the SMU
applies regardless of the P-state setting (SoC-rail VRM or IOD thermal,
DIMM thermal throttling in the UMC, PROCHOT) or a defect behind socket 1.

With HSMP now enabled, `tools/hsmp-query.py` reads FCLK/MCLK, PROC_HOT,
socket power and the CCLK throttle limit per socket through `/dev/hsmp`
(read-only, non-root) once `amd_hsmp` is loaded — that needs one
`sudo modprobe amd_hsmp`, and `sudo modprobe jc42` puts the DIMM TSODs
into hwmon. Reading those in a tripped state is the next step.

## After the BIOS change: it toggles, and a warm socket trips sooner

Three more 11-12 minute ik_llama runs on socket 1 with APBDIS=1 / P0:

| start | state | trip | throttled for | then |
|---|---|---|---|---|
| 19:46 (fresh boot) | cold | 7.5 min | rest of run (4.5 min) | 54 GB/s after |
| 20:07 | 8 min idle | 6.3 min | 1.7 min | **recovered under load**, 3 min clean; ~20 s after the load stopped it was tripped again (53 GB/s, latency 126.5 ns vs 114.1) |
| 20:19 | warm, 1 min idle | **2 min** | 7 min | recovered under load, 3 min clean, 101 GB/s after |

Socket 1 power 136 W clean, 113 W throttled; Tctl 47 → 44-45 → 47; socket 0
untouched throughout. Three things follow. It switches both ways on its own,
under load and without it, on a timescale of minutes — a hysteresis loop. A
pre-warmed socket trips three times sooner. And the unloaded latency in the
tripped state rises only 11% (`tools/memlat-probe`, 2 MB pages, 1 GB chase):
a memory or fabric clock cut in half would add 40-50 ns, not 12, so this is
a **command-rate throttle in the memory path with the clocks intact** — the
shape of a thermal protection, applied by the SMU whatever the P-state
setting.

The host cannot see the sensor that drives it: `jc42` instantiated at
0x18-0x1f on all three PIIX4 buses finds no DIMM TSOD (the SPD bus is the
BMC's), the in-tree `amd_hsmp` refuses family 0x17 ("No such device"), and
k10temp shows only Tctl/Tccd. What is left is the BMC's sensor page and a
hand: after a 12-minute socket-1 run, the CPU1 EPS 8-pin connector and cable,
the socket-1 VRM heatsink and the DIMM banks, compared with socket 0's. A
connector or VRM that is hot to the touch on socket 1 only is the answer.

## Hands and the BMC: it is heat, and a fan removes it

09-12 late evening, with the user at the machine (CPUs are water-cooled, so
nothing but case airflow reaches the DIMMs and VRMs):

- Socket 1's DIMMs are hot to the touch under a socket-1 run; socket 0's are
  just as hot under a socket-0 run. The VRM heatsink between the sockets is
  equally hot under either load (shared). Nothing socket 1 has feels hotter
  than socket 0's counterpart.
- `dmidecode -t memory` (root): socket 0 holds 8 × Micron 36ASF4G72PZ-2G6D1;
  socket 1 holds 7 × Micron 36ASF4G72PZ-**2G6E1** plus **one Kingston
  HP26D4R9D4MEI-32**, confirmed by eye to be in socket 1. All 2Rx4 32 GB at
  2667 MT/s.
- The BMC (OpenBMC, Redfish, <bmc-ip>, enabled tonight) carries one set
  of "CPU_*" sensors that follow socket 0 only (CPU_Power stays 58 W while
  socket 1 runs at 136 W), no per-DIMM temperatures, empty SEL. What it does
  show: **inlet air 43 → 50 °C over a six-minute socket-1 run**, socket 0's
  VRMs +7 °C while idle. The case air is hot.
- **Side panel off and a desk fan on the socket-1 side: 45 requests, 12
  minutes, 16.15-16.17 tok/s throughout, node 1 at 103.5 GB/s after, inlet
  held at 38-39 °C.** Every other socket-1 run today tripped between 2 and
  7.5 minutes.

So: a thermal throttle in socket 1's memory path, cleared by cooling the
socket-1 side. Because eight channels are interleaved, one throttled channel
halves the whole socket, and the one foreign DIMM sits in socket 1 — the
Kingston's TSOD crossing the throttle threshold first in 45-50 °C case air is
the simplest account, and it explains why socket 0 with the same DIMM
temperature to the hand never throttles. Confirmation is a two-DIMM swap
(the Kingston for one socket-0 Micron, same slots) followed by the socket-1
and socket-0 12-minute runs: the trip should follow the Kingston.

## Closed: fan fixed in place, panel closed, the real workload

- Kingston removed (a spare Micron 2G6D1 in its slot), panel closed, no fan:
  trips at 6 min as before (r22, inlet 48.5 °C). **The Kingston was not the
  trigger.** What remains socket-1-specific is the E1 set or the air at that
  bank; a D1/E1 set swap between sockets would tell which, and does not
  change the fix.
- Fan fixed on the socket-1 side, panel closed: ik_llama 45 requests,
  16.05-16.17 tok/s flat, node 1 at 100.9 GB/s after, inlet 43 → 46.5 °C.
- **V4.1, default launcher, 701 × 14 back-to-back (`700x14-fan`): r1 38 s,
  r2-r14 31-32 s, no trip.** The morning's r8-r11 decay is gone in the
  production form. Inlet 44-48 °C during the run.

## Where that leaves it

Fix the air first — a fan across the socket-1 DIMM bank, or case intake that
keeps the inlet under ~40 °C — and the decay is gone at every layer above.
Then, when convenient, the Kingston swap to pin it to the one DIMM, and
retire that DIMM (15 × Micron, or a matching 36ASF4G72PZ) so the machine does
not depend on a fan. Everything in the stack — kt-kernel, SGLang, the V4.1
configuration, the BIOS P-state settings (left in place, harmless) — is
cleared.

## Tools added

- `tools/watch-dsv41-pools.sh` — per-subpool worker/manager CPU ticks over a
  window, beside a run.
- `tools/snap-dsv41-pool-sys.sh` — user/system split per pool.
- `tools/snap-dsv41-threads.sh` — every thread of TP0 with state, wchan, last
  CPU, ticks over a span.
- `tools/membw-probe.c` — per-node bandwidth, sequential or `random`.
- `tools/clock-probe.c` — effective core clock from a dependent-add chain.
- `tools/probe-dsv41-tripped.sh` — after a `KEEP=1` run: request with
  per-node clocks and pool states sampled, bandwidth on both sockets while
  idle, then idle in steps until a request comes back clean.
- `tools/watch-bmc-sensors.sh` — the BMC's sensors over Redfish, beside a run.
- `tools/memlat-probe.c` — unloaded DRAM latency by pointer chasing.
- `tools/hsmp-query.py` — HSMP telemetry; needs a driver that accepts Rome,
  which the in-tree one does not.
- `tools/probe-node1-ikllama.sh` — the stack-free reproduction: a small dense
  GGUF through ik_llama.cpp pinned to one socket for DURATION seconds, per-
  request decode tok/s, bandwidth probes after.
- `tools/measure-dsv41-prefill-order.sh` — `KEEP=1`, NUMA counters in the
  pre/post lines; `tools/watch-dsv41-thermal.sh` — NUMA hinting counters per
  sample; `tools/verify-dsv41-swa-long.sh` — `TEMP`/`TOP_P`/`REPS` and the
  comma gate, refuses to launch beside a GPU job.

Logs on the host: `logs/dsv41-prefill-700x14-{pool1-oom,swap,numa,keepalive1}.txt`,
`logs/dsv41-prefill-700x12-keep*.txt`, `logs/dsv41-pools-*.txt`,
`logs/dsv41-poolsys-*.txt`, `logs/dsv41-probe-tripped-*.txt`,
`logs/dsv41-heat-{nodes,gpu1-node1,random-node1}.txt`.
