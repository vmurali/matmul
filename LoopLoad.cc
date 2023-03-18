#include <cstdio>
#include <immintrin.h>
#include <x86intrin.h>

#ifndef num
#define num 50000
#endif

int main() {
  float __attribute__((__aligned__(64))) a[2];
  char* aptr = (char*)a;
  for (int i = 0; i < 20; i++) {
    asm volatile("vmovups (%0), %%zmm0;" : : "r"(aptr) : "zmm0");
    asm volatile("vmovups 64(%0), %%zmm0;" : : "r"(aptr) : "zmm0");
    long long start = __rdtsc();
    for (int i = 0; i < num; i++)
    asm volatile("vmovups 64(%0), %%zmm0;" : : "r"(aptr) : "zmm0");
    long long end = __rdtsc();
    double diff = end - start;
    printf("Cycles for each load: %lf cycles for START=%d\n", diff/num, 0);
    start = __rdtsc();
    for (int i = 0; i < num; i++)
    asm volatile("vmovups 2(%0), %%zmm0;" : : "r"(aptr) : "zmm0");
    end = __rdtsc();
    diff = end - start;
    printf("Cycles for each load: %lf cycles for START=%d\n", diff/num, 1);
  }
}
