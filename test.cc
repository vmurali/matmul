#include "kernel.h"

#include <vector>
#include <thread>
#include <cstdio>
#include <cstdlib>

#ifdef DEBUG
void print_mm(float *X, int M, int N) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      printf("%f ", X[i*N+j]);
    }
    printf("\n");
  }
  printf("\n");
}
#endif

void initialize(float *X, int M, int N) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      X[i*N+j] = rand()%10;
    }
  }
}

void copy(float *X, float *Y, int M, int N) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      Y[i*N+j] = X[i*N+j];
    }
  }
}

void simple_mm(float *A, float *B, float *C, int M, int N, int K) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < K; k++) {
        C[i*N+j] += A[k*M+i]*B[k*N+j];
      }
    }
  }
}

void check(float *X, float *Y, int M, int N) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      if (X[i*N+j] != Y[i*N+j]) {
        printf("Error at (%d, %d), got: %f, expected: %f\n", i, j, X[i*N+j], Y[i*N+j]);
        exit(1);
      }
    }
  }
  printf("All equal\n");
}

int main(int argc, char**argv) {
  if (!(argc == 4 || argc == 5)) {
    printf("%d\n", argc);
    printf("Usage: %s M N K [numThreads]\n", argv[0]);
    return 1;
  }
  const int M = atoi(argv[1]);
  const int N = atoi(argv[2]);
  const int K = atoi(argv[3]);
  int numThreads = std::thread::hardware_concurrency();
  if (argc == 5)
    numThreads = atoi(argv[4]);
  std::vector<std::thread> threads;
  threads.reserve(numThreads);
  float *A = new float[M*K];
  float *B = new float[N*K];
  float *C = new float[M*N];
  float *D = new float[M*N];
  initialize(A, K, M);
  initialize(B, K, N);
  initialize(C, M, N);
  copy(C, D, M, N);
#ifdef DEBUG
  printf("A\n");
  print_mm(A, K, M);
  printf("B\n");
  print_mm(B, K, N);
  printf("C\n");
  print_mm(C, M, N);
#endif
  auto start = std::chrono::high_resolution_clock::now();
  mm_f32_full((char*)A, (char*)B, (char*)C, M, N, K, numThreads, threads);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;

  printf("%d %d %d: %f GFLOPS\n", M, N, K, (long long)M*N*K*2/diff.count()*(1e-9f));

  simple_mm(A, B, D, M, N, K);
  check(C, D, M, N);

  return 0;
}
