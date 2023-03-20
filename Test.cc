#include "Kernel.h"
#include "ThreadPool.h"

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <thread>

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
      X[i*N+j] = 0; //rand()%10;
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
  if (!(argc >= 5 && argc <= 7)) {
    printf("Usage: %s M N K check [numThreads] [seed]\n", argv[0]);
    return 1;
  }
  const int Morig = atoi(argv[1]);
  const int Norig = atoi(argv[2]);
  const int Korig = atoi(argv[3]);
  const int M = ((Morig + 15) >> 4) << 4;
  const int N = ((Norig + 15) >> 4) << 4;
  const int K = ((Korig + 15) >> 4) << 4;
  int isCheck = atoi(argv[4]);
  int numThreads = std::thread::hardware_concurrency()-1;
  if (argc >= 6 && atoi(argv[5]) != 0)
    numThreads = atoi(argv[5]);
  int seed = time(NULL);
  if (argc == 7 && atoi(argv[6]) != 0)
    seed = atoi(argv[6]);
  srand(seed);

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

  ThreadPool threadPool(numThreads);

  struct timespec start;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

  MMF32Full((char*)A, (char*)B, (char*)C, M, N, K, numThreads, threadPool);

  struct timespec end;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);

  long long diff = (end.tv_sec - start.tv_sec) * (long long)1e9 + (end.tv_nsec - start.tv_nsec);

  printf("%d\t%d\t%d\t%lld\t%d\n", Morig, Norig, Korig, diff, numThreads);

  if (isCheck) {
    simple_mm(A, B, D, M, N, K);
    check(C, D, M, N);
  }

  return 0;
}
