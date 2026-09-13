# Per-socket read bandwidth is CCD-limited, not channel-limited — 2026-09-10

Machine idle: no server, no training, no other job. `tools/ccd-bandwidth-sweep.sh`,
node 0 only, `numactl --cpunodebind=0 --membind=0`, AVX2 read-only, 32 MiB per
thread first-touched locally, 64 passes, best of 3 after a discarded warmup.

Host: 2x EPYC 7452 (Zen 2), 8x 16 MB L3 domains per socket of 4 cores each,
16x Micron MTA36ASF4G72PZ-2G6 (32 GiB 2Rx4 DDR4-2666 RDIMM), 8 per socket.

## Result

| L3 domains | cores/domain | threads | total GB/s | per thread |
|---:|---:|---:|---:|---:|
| 1 | 1 | 1 | 19.9 | 19.89 |
| 1 | 2 | 2 | 31.9 | 15.96 |
| 1 | 4 | 4 | 36.0 | 9.00 |
| 2 | 1 | 2 | 33.5 | 16.74 |
| 2 | 2 | 4 | 35.9 | 8.98 |
| 2 | 4 | 8 | 36.9 | 4.62 |
| 4 | 1 | 4 | 63.1 | 15.78 |
| 4 | 2 | 8 | 69.5 | 8.68 |
| 4 | 4 | 16 | 72.4 | 4.53 |
| 8 | 1 | 8 | 101.6 | 12.71 |
| 8 | 2 | 16 | 109.7 | 6.86 |
| 8 | 4 | 32 | 116.0 | 3.62 |

Raw: `logs/ccd-sweep-20260910.txt`.

## Reading

One L3 domain saturates at 36.0 GB/s on 4 cores. Adding a second domain —
twice the cores, twice the L3 — buys 0.9 GB/s. Domains pair up behind a shared
~37 GB/s ceiling, which is one CCD's Infinity Fabric link carrying its two CCX
on Zen 2. Four domains give 2.0x that, eight give 3.2x.

So the 116 GB/s per-socket ceiling is 4 CCDs x ~37 GB/s, not 8 channels x 21.3
GB/s (170.6 GB/s on paper). We leave 32% of the installed DIMM bandwidth unused,
and no DIMM change can reach it.

Caveat: the die grouping is inferred, not read. Linux reports `die_id=0` for
every CPU here, so "two L3 domains share one link" comes from the 36.0 -> 36.9
step, not from topology data.

## Consequences

- Replacing DDR4-2666 with DDR4-3200 changes nothing on this CPU. The channels
  are already 32% idle.
- Cores past 4 per L3 domain are nearly free of bandwidth effect (36.0 at 4
  cores vs 31.9 at 2).
- The variable that matters is CCD count. On both Rome and Milan a CCD carries
  32 MB of L3, so CCD count = L3 MB / 32. The 7452 and the 7402 are 128 MB
  parts: 4 CCDs. A 7532 (Rome, 32c) or a 7543/7763 (Milan) is 256 MB: 8 CCDs.
- An 8-CCD part would lift the fabric ceiling to ~296 GB/s per socket, at which
  point the 8 DDR4-2666 channels bind instead. Expect ~145-155 GB/s per socket,
  roughly +30% over today, and only then would faster DIMMs start to pay.
- The kt-kernel MXFP4 expert path is memory-bound after
  kvcache-ai/ktransformers#2175, so decode should track that gain closely.

Posted to kvcache-ai/ktransformers#2175, where someone speccing a dual 7402
asked whether to spend on cores or on faster RAM. Neither, on a 4-CCD part.
