#include "Kernel.h"
#include "Pack.h"
#include "ThreadPool.h"

#include <immintrin.h>
#include <thread>
#include <vector>

#if defined(DEBUG) || defined(DEBUGMORE)
#include <cstdio>
#endif

#ifdef DEBUG
void print(char* A, char* B, char* C, int M, int N, int K, char* kA, char* kB, char* c[16], int mMask, int nMask) {
  int aBytes = kA - A;
  int bBytes = kB - B;
  printf("A(%4d,%4d):: (%4d, %4d)|"
         "B(%4d,%4d):: (%4d, %4d)|"
         "MMask: %4d|NMask: %4d\n"
         "C(%4d,%4d):: (%4ld, %4ld) (%4ld, %4ld) (%4ld, %4ld) (%4ld, %4ld) (%4ld, %4ld) (%4ld, %4ld) (%4ld, %4ld) (%4ld, %4ld) (%4ld, %4ld) (%4ld, %4ld) (%4ld, %4ld) (%4ld, %4ld) (%4ld, %4ld) (%4ld, %4ld) (%4ld, %4ld) (%4ld, %4ld)\n",
         M, K, aBytes/4/M, (aBytes/4)%M,
         N, K, bBytes/4/N, (bBytes/4)%N,
         _mm_popcnt_u32(mMask), _mm_popcnt_u32(nMask),
         M, N,
         (c[0]-C)/4/N, (c[0]-C)/4%N,
         (c[1]-C)/4/N, (c[1]-C)/4%N,
         (c[2]-C)/4/N, (c[2]-C)/4%N,
         (c[3]-C)/4/N, (c[3]-C)/4%N,
         (c[4]-C)/4/N, (c[4]-C)/4%N,
         (c[5]-C)/4/N, (c[5]-C)/4%N,
         (c[6]-C)/4/N, (c[6]-C)/4%N,
         (c[7]-C)/4/N, (c[7]-C)/4%N,
         (c[8]-C)/4/N, (c[8]-C)/4%N,
         (c[9]-C)/4/N, (c[9]-C)/4%N,
         (c[10]-C)/4/N, (c[10]-C)/4%N,
         (c[11]-C)/4/N, (c[11]-C)/4%N,
         (c[12]-C)/4/N, (c[12]-C)/4%N,
         (c[13]-C)/4/N, (c[13]-C)/4%N,
         (c[14]-C)/4/N, (c[14]-C)/4%N,
         (c[15]-C)/4/N, (c[15]-C)/4%N);
}
#endif

#ifdef DEBUGMORE
void printFloats(char *str, __m512 x) {
  float y[16];
  _mm512_storeu_ps(y, x);
  printf("%s: ", str);
  for (int i = 0; i < 16; i++) {
    printf(" %10.5f", y[i]);
  }
  printf("\n");
}

void printInts(char *str, __m512i x) {
  int y[16];
  _mm512_storeu_epi32(y, x);
  printf("%s: ", str);
  for (int i = 0; i < 16; i++) {
    printf(" %d", y[i]);
  }
  printf("\n");
}

#define broadcast_fma_incrementIdx_16_f32( \
  a, \
  b, \
  c, \
  offset \
) \
  printFloats("Cin", c); \
  asm volatile( \
    "vfmadd231ps " #offset "(%1), %2, %0;" \
    : "+v&"(c) \
    : "r"(a), "v"(b) \
    : \
  ); \
  printFloats("Bin", b); \
  printFloats("Cot", c)

#define broadcast_fma_incrementIdx_16_f32_mask( \
  a, \
  b, \
  c, \
  offset, \
  mask \
) \
  printFloats("Cin", c); \
  asm volatile( \
    "vfmadd231ps " #offset "(%1), %2, %0 %{%3%}%{z%};" \
    : "+v&"(c) \
    : "r"(a), "v"(b), "Yk"(mask) \
    : \
  ); \
  printFloats("Bin", b); \
  printFloats("Cot", c)

#else
#define broadcast_fma_incrementIdx_16_f32( \
  a, \
  b, \
  c, \
  offset \
) \
  asm volatile( \
    "vfmadd231ps " #offset "(%1), %2, %0;" \
    : "+v&"(c) \
    : "r"(a), "v"(b) \
    : \
  )

#define broadcast_fma_incrementIdx_16_f32_mask( \
  a, \
  b, \
  c, \
  offset, \
  mask \
) \
  asm volatile( \
    "vfmadd231ps " #offset "(%1), %2, %0 %{%3%}%{z%};" \
    : "+v&"(c) \
    : "r"(a), "v"(b), "Yk"(mask) \
    : \
  )
#endif

#define mm_16x16_f32( \
  a, \
  b, \
  c0, \
  c1, \
  c2, \
  c3, \
  c4, \
  c5, \
  c6, \
  c7, \
  c8, \
  c9, \
  cA, \
  cB, \
  cC, \
  cD, \
  cE, \
  cF \
) \
  broadcast_fma_incrementIdx_16_f32(a, b, c0, 0); \
  broadcast_fma_incrementIdx_16_f32(a, b, c1, 4); \
  broadcast_fma_incrementIdx_16_f32(a, b, c2, 8); \
  broadcast_fma_incrementIdx_16_f32(a, b, c3, 12); \
  broadcast_fma_incrementIdx_16_f32(a, b, c4, 16); \
  broadcast_fma_incrementIdx_16_f32(a, b, c5, 20); \
  broadcast_fma_incrementIdx_16_f32(a, b, c6, 24); \
  broadcast_fma_incrementIdx_16_f32(a, b, c7, 28); \
  broadcast_fma_incrementIdx_16_f32(a, b, c8, 32); \
  broadcast_fma_incrementIdx_16_f32(a, b, c9, 36); \
  broadcast_fma_incrementIdx_16_f32(a, b, cA, 40); \
  broadcast_fma_incrementIdx_16_f32(a, b, cB, 44); \
  broadcast_fma_incrementIdx_16_f32(a, b, cC, 48); \
  broadcast_fma_incrementIdx_16_f32(a, b, cD, 52); \
  broadcast_fma_incrementIdx_16_f32(a, b, cE, 56); \
  broadcast_fma_incrementIdx_16_f32(a, b, cF, 60)

#define mm_16x16_f32_mask( \
  a, \
  b, \
  c0, \
  c1, \
  c2, \
  c3, \
  c4, \
  c5, \
  c6, \
  c7, \
  c8, \
  c9, \
  cA, \
  cB, \
  cC, \
  cD, \
  cE, \
  cF, \
  mask \
) \
  broadcast_fma_incrementIdx_16_f32_mask(a, b, c0, 0, mask); \
  broadcast_fma_incrementIdx_16_f32_mask(a, b, c1, 4, mask); \
  broadcast_fma_incrementIdx_16_f32_mask(a, b, c2, 8, mask); \
  broadcast_fma_incrementIdx_16_f32_mask(a, b, c3, 12, mask); \
  broadcast_fma_incrementIdx_16_f32_mask(a, b, c4, 16, mask); \
  broadcast_fma_incrementIdx_16_f32_mask(a, b, c5, 20, mask); \
  broadcast_fma_incrementIdx_16_f32_mask(a, b, c6, 24, mask); \
  broadcast_fma_incrementIdx_16_f32_mask(a, b, c7, 28, mask); \
  broadcast_fma_incrementIdx_16_f32_mask(a, b, c8, 32, mask); \
  broadcast_fma_incrementIdx_16_f32_mask(a, b, c9, 36, mask); \
  broadcast_fma_incrementIdx_16_f32_mask(a, b, cA, 40, mask); \
  broadcast_fma_incrementIdx_16_f32_mask(a, b, cB, 44, mask); \
  broadcast_fma_incrementIdx_16_f32_mask(a, b, cC, 48, mask); \
  broadcast_fma_incrementIdx_16_f32_mask(a, b, cD, 52, mask); \
  broadcast_fma_incrementIdx_16_f32_mask(a, b, cE, 56, mask); \
  broadcast_fma_incrementIdx_16_f32_mask(a, b, cF, 60, mask)

void MMF32(char *aCurr, char *bCurr, char *cCurr, int blockSize,
           MMF32Params *params) {
  int NBytes = params->NBytes;

  char* c[16] = {cCurr, cCurr+NBytes, cCurr+2*NBytes, cCurr+3*NBytes,
    cCurr+4*NBytes, cCurr+5*NBytes, cCurr+6*NBytes, cCurr+7*NBytes,
    cCurr+8*NBytes, cCurr+9*NBytes, cCurr+10*NBytes, cCurr+11*NBytes,
    cCurr+12*NBytes, cCurr+13*NBytes, cCurr+14*NBytes, cCurr+15*NBytes};

  char *a = aCurr;
  char *b = bCurr;
  int KRowsBytes = params->KRowsBytes;
  for (; a < params->AChunkEnd; a += KRowsBytes) {
    for (; b < params->BChunkEnd; b += KRowsBytes) {
      char* kA = a;
      char* kB = b;
      register __m512 c0 = _mm512_loadu_ps(c[0]);
      register __m512 c1 = _mm512_loadu_ps(c[1]);
      register __m512 c2 = _mm512_loadu_ps(c[2]);
      register __m512 c3 = _mm512_loadu_ps(c[3]);
      register __m512 c4 = _mm512_loadu_ps(c[4]);
      register __m512 c5 = _mm512_loadu_ps(c[5]);
      register __m512 c6 = _mm512_loadu_ps(c[6]);
      register __m512 c7 = _mm512_loadu_ps(c[7]);
      register __m512 c8 = _mm512_loadu_ps(c[8]);
      register __m512 c9 = _mm512_loadu_ps(c[9]);
      register __m512 cA = _mm512_loadu_ps(c[10]);
      register __m512 cB = _mm512_loadu_ps(c[11]);
      register __m512 cC = _mm512_loadu_ps(c[12]);
      register __m512 cD = _mm512_loadu_ps(c[13]);
      register __m512 cE = _mm512_loadu_ps(c[14]);
      register __m512 cF = _mm512_loadu_ps(c[15]);

      for (; kA < a + KRowsBytes; kA += 64, kB += 64) {
#ifdef DEBUG
        printf("Both good: ");
	print(params->A, params->B, params->C, params->M, params->N, params->K,
              kA, kB, c, 0xFFFF, 0xFFFF);
#endif
        register __m512 kBVal;
        char *vA;
        vA = kA;
        kBVal = _mm512_loadu_ps(kB);
        mm_16x16_f32(
	  vA, kBVal,
          c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, cA, cB, cC, cD, cE, cF);
      }
      _mm512_storeu_ps(c[0], c0);
      _mm512_storeu_ps(c[1], c1);
      _mm512_storeu_ps(c[2], c2);
      _mm512_storeu_ps(c[3], c3);
      _mm512_storeu_ps(c[4], c4);
      _mm512_storeu_ps(c[5], c5);
      _mm512_storeu_ps(c[6], c6);
      _mm512_storeu_ps(c[7], c7);
      _mm512_storeu_ps(c[8], c8);
      _mm512_storeu_ps(c[9], c9);
      _mm512_storeu_ps(c[10], cA);
      _mm512_storeu_ps(c[11], cB);
      _mm512_storeu_ps(c[12], cC);
      _mm512_storeu_ps(c[13], cD);
      _mm512_storeu_ps(c[14], cE);
      _mm512_storeu_ps(c[15], cF);
      blockSize--;
      if (blockSize == 0) {
        return;
      }

      for (int i = 0; i < 16; i++) {
        c[i] += 64;
      }
    }
    for (int i = 0; i < 16; i++) {
      c[i] += params->CIncEndRow;
    }
    b = params->B;
  }
}

void MMF32Full(char* Aorig, char* Borig, char* C, int Morig, int Norig, int Korig, int numThreads, ThreadPool& threadPool) {
  char *A = (char*)PackLeft((float*)Aorig, Morig, Korig);
  char *B = (char*)PackRight((float*)Borig, Norig, Korig);
  int MTiles = Morig>>4;
  int NTiles = Norig>>4;
  int KTiles = Korig>>4;
  int M = MTiles << 4;
  int N = NTiles << 4;
  int K = KTiles << 4;
  std::vector<std::thread> threads;
  threads.reserve(numThreads);
  MMF32Params params(A, B, C, M, N, K);
  int MNTiles = MTiles*NTiles;
  int blockSize = (MNTiles+numThreads-1)/numThreads;
  int blockSizeBytes = blockSize<<6;
#ifdef DEBUG
  printf("blockSize: %d, MTiles: %d, NTiles: %d\n", blockSize, MTiles, NTiles);
#endif
  for (int j = 0; j < MNTiles; j += blockSize) {
    int aBlockBytes = (j/NTiles)<<6;
    int bBlockBytes = (j%NTiles)<<6;
    char *a = A + aBlockBytes*K;
    char *b = B + bBlockBytes*K;
    char *c = C + (aBlockBytes*N) + bBlockBytes;
#ifdef DEBUG
    printf("Call: %d %d A(%d %d) B(%d %d) C(%d %d)\n", aBlockBytes>>6, bBlockBytes>>6,
      (aBlockBytes>>2)/M, (aBlockBytes>>2)%M,
      (bBlockBytes>>2)/N, (bBlockBytes>>2)%N,
      ((aBlockBytes*N+bBlockBytes)>>2)/N, ((aBlockBytes*+bBlockBytes)>>2)%N
    );
#endif
    //threadPool.QueueJob(a, b, c, blockSize, &params, localT);
    threads.push_back(std::thread(MMF32, a, b, c, blockSize, &params));
  }
  //threadPool.WaitDone();
  for (auto &th: threads) {
    th.join();
  }
}
