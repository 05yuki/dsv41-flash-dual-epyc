// Unloaded DRAM latency by dependent pointer chasing over a random cyclic
// permutation of a large buffer (madvise'd for 2 MB pages so TLB misses do
// not dominate). Distinguishes the two ways a socket's bandwidth can halve:
// a lower memory/fabric clock raises latency too; a command-rate throttle in
// the memory controller (DIMM thermal throttling) leaves it nearly unchanged.
//   numactl --cpunodebind=N --membind=N ./memlat-probe [MiB] [steps in millions]
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

int main(int argc, char** argv) {
  size_t mib = argc > 1 ? strtoull(argv[1], 0, 10) : 1024;
  uint64_t steps = (argc > 2 ? strtoull(argv[2], 0, 10) : 50) * 1000000ull;
  size_t n = mib << 20 >> 6;  // one 64-byte line per node
  uint64_t* buf = mmap(NULL, n * 64, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (buf == MAP_FAILED) { perror("mmap"); return 1; }
  madvise(buf, n * 64, MADV_HUGEPAGE);
  // Sattolo's algorithm: one cycle through every line, in random order
  uint64_t* idx = malloc(n * sizeof(uint64_t));
  for (size_t i = 0; i < n; i++) idx[i] = i;
  uint64_t x = 0x9E3779B97F4A7C15ull;
  for (size_t i = n - 1; i > 0; i--) {
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    size_t j = x % i;
    uint64_t t = idx[i]; idx[i] = idx[j]; idx[j] = t;
  }
  for (size_t i = 0; i < n; i++) buf[idx[i] * 8] = idx[(i + 1) % n] * 8;
  free(idx);
  uint64_t p = 0;
  for (uint64_t i = 0; i < 1000000; i++) p = buf[p];  // warm
  double t0 = now();
  for (uint64_t i = 0; i < steps; i++) p = buf[p];
  double t1 = now();
  printf("MiB=%zu steps=%llu latency=%.1f ns (p=%llu)\n", mib, (unsigned long long)steps,
         (t1 - t0) / steps * 1e9, (unsigned long long)p);
  return 0;
}
