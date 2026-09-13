// Effective core clock without perf: a chain of dependent 1-cycle adds runs
// at one add per cycle on Zen, so iterations / seconds is the effective GHz
// including any throttling that /proc/cpuinfo's P-state number hides.
// Run pinned: taskset -c <cpu> ./clock-probe [iterations in millions]
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
int main(int argc, char** argv) {
  uint64_t n = (argc > 1 ? strtoull(argv[1], 0, 10) : 2000) * 1000000ull;
  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  uint64_t x = 1;
  for (uint64_t i = 0; i < n; i++) __asm__ volatile("add $1, %0" : "+r"(x));
  clock_gettime(CLOCK_MONOTONIC, &t1);
  double s = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
  printf("%.3f GHz (%llu adds in %.3fs, x=%llu)\n", n / s / 1e9, (unsigned long long)n, s, (unsigned long long)x);
  return 0;
}
