// Per-node read bandwidth probe: NTHREADS threads each stream over their own
// buffer (first-touched by the thread, so it lands on the thread's node under
// numactl --membind) and report aggregate GB/s. Run as
//   numactl --cpunodebind=N --membind=N ./membw-probe [threads] [MiB per thread] [passes]
// to compare the two sockets while the model server sits in a given state.
// A fourth argument "random" reads one cache line at a random offset per
// access instead of streaming: the same bytes cost far more DRAM row
// activations, which is what heats DIMMs, so it is the pattern to use when
// trying to reproduce the node-1 bandwidth drop without the model.
#define _GNU_SOURCE
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int nthreads = 8, passes = 5, random_mode = 0;
static size_t bytes = 512u << 20;
static volatile uint64_t sink;

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static void* run(void* arg) {
  uint64_t* buf = malloc(bytes);
  size_t n = bytes / sizeof(uint64_t);
  for (size_t i = 0; i < n; i++) buf[i] = i;  // first touch here, on this thread
  double* out = arg;
  double t0 = now();
  uint64_t acc = 0;
  if (random_mode) {
    // one 64-byte line per access, 8 independent streams per thread so the
    // core keeps several misses in flight
    uint64_t x[8]; for (int j = 0; j < 8; j++) x[j] = 0x9E3779B97F4A7C15ull * (j + 1) + (uintptr_t)buf;
    size_t lines = n / 8, per_pass = lines / 8;
    for (int p = 0; p < passes; p++)
      for (size_t i = 0; i < per_pass; i++)
        for (int j = 0; j < 8; j++) {
          x[j] = x[j] * 6364136223846793005ull + 1442695040888963407ull;
          acc += buf[((x[j] >> 20) % lines) * 8];
        }
  } else
  for (int p = 0; p < passes; p++)
    for (size_t i = 0; i < n; i += 8)
      acc += buf[i] + buf[i + 1] + buf[i + 2] + buf[i + 3] + buf[i + 4] + buf[i + 5] + buf[i + 6] + buf[i + 7];
  double t1 = now();
  sink += acc;
  *out = (double)bytes * passes / (t1 - t0) / 1e9;
  free(buf);
  return NULL;
}

int main(int argc, char** argv) {
  if (argc > 1) nthreads = atoi(argv[1]);
  if (argc > 2) bytes = (size_t)atoi(argv[2]) << 20;
  if (argc > 3) passes = atoi(argv[3]);
  if (argc > 4 && argv[4][0] == 'r') random_mode = 1;
  pthread_t th[256]; double gbps[256];
  for (int i = 0; i < nthreads; i++) pthread_create(&th[i], NULL, run, &gbps[i]);
  double sum = 0, mn = 1e9, mx = 0;
  for (int i = 0; i < nthreads; i++) { pthread_join(th[i], NULL); sum += gbps[i]; if (gbps[i] < mn) mn = gbps[i]; if (gbps[i] > mx) mx = gbps[i]; }
  // in random mode the figure is bytes of whole lines touched per second
  printf("threads=%d MiB/thread=%zu passes=%d%s aggregate=%.1f GB/s per-thread min=%.1f max=%.1f\n",
         nthreads, bytes >> 20, passes, random_mode ? " random" : "", sum, mn, mx);
  return 0;
}
