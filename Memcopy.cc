#include <cstdio>
#include <cstdlib>
#include <immintrin.h>
#include <chrono>

int main() {
  float* a = new float[N];
  float res[16] = {0};
  register __m512 resR;
  resR = _mm512_loadu_ps(res);
  
  for (unsigned long long i = 0; i < N; i++) {
    a[i] = i;
  }
  register __m512 ld;
  for (int START = 0; START < 64; START++) {
    auto start = std::chrono::high_resolution_clock::now();
    for (char *i = (char *)a + START; i < (char*)a + (N-1ULL)*4; i += 64) {
      ld = _mm512_loadu_ps(i);
      resR = _mm512_add_ps(resR, ld);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = (end - start);
    printf("Time for each load+add: %lf ns for START=%d\n", diff.count()*(1e9)/N, START);
    _mm512_storeu_ps(res, resR);
    float fin = 0;
    for(int i = 0; i < 16; i++) {
      fin += res[i];
    }
    printf("%f\n", fin);
  }
}
