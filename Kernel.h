#ifndef KERNEL
#define KERNEL

#include "ThreadPool.h"

#include <vector>

struct MMF32Params {
  char *A;
  char *B;
  char *C;
  int M;
  int N;
  int K;
  int MBytes;
  int NBytes;
  char *AChunkEnd;
  char *BChunkEnd;
  int KRowsBytes;
  int CIncEndRow;
  MMF32Params(char *_A, char *_B, char *_C, int _M, int _N, int _K)
    : A(_A), B(_B), C(_C), M(_M), N(_N), K(_K),
      MBytes(M<<2),
      NBytes(N<<2),
      AChunkEnd(A + (M*K<<2)),
      BChunkEnd(B + (N*K<<2)),
      CIncEndRow(15*NBytes),
      KRowsBytes(K<<6)
	{}
};

void MMF32(char* A, char* B, char* C, int blockSize, MMF32Params *params);
void MMF32Full(char* A, char* B, char* C, int M, int N, int K, int numThreads, ThreadPool& threadPool);

#endif
