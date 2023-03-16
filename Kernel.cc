#include "Kernel.h"

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
  broadcastIdx, \
  ones \
) \
  printInts("Idx", broadcastIdx); \
  printFloats("Ain", a); \
  printFloats("Cin", c); \
  { \
    __m512 temp; \
    asm volatile( \
      "vpermd %3, %1, %2;" \
      "vfmadd231ps  %4, %2, %0;" \
      "vpaddd %1, %5, %1;" \
      : "+v&"(c), "+v&"(broadcastIdx), "=v&"(temp) \
      : "v"(a), "v"(b), "v"(ones) \
      : \
    ); \
    printFloats("Tmp", temp); \
    printFloats("Bin", b); \
    printFloats("Cot", c); \
  } \
  printInts("One", ones)
#else
#define broadcast_fma_incrementIdx_16_f32( \
  a, \
  b, \
  c, \
  broadcastIdx, \
  ones \
) \
  asm volatile( \
    "vpermd %2, %1, %%zmm0;" \
    "vfmadd231ps  %3, %%zmm0, %0;" \
    "vpaddd %1, %4, %1;" \
    : "+v&"(c), "+v&"(broadcastIdx) \
    : "v"(a), "v"(b), "v"(ones) \
    : "zmm0")
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
  cF, \
  broadcastIdx, \
  ones \
) { \
  broadcast_fma_incrementIdx_16_f32(a, b, c0, broadcastIdx, ones); \
  broadcast_fma_incrementIdx_16_f32(a, b, c1, broadcastIdx, ones); \
  broadcast_fma_incrementIdx_16_f32(a, b, c2, broadcastIdx, ones); \
  broadcast_fma_incrementIdx_16_f32(a, b, c3, broadcastIdx, ones); \
  broadcast_fma_incrementIdx_16_f32(a, b, c4, broadcastIdx, ones); \
  broadcast_fma_incrementIdx_16_f32(a, b, c5, broadcastIdx, ones); \
  broadcast_fma_incrementIdx_16_f32(a, b, c6, broadcastIdx, ones); \
  broadcast_fma_incrementIdx_16_f32(a, b, c7, broadcastIdx, ones); \
  broadcast_fma_incrementIdx_16_f32(a, b, c8, broadcastIdx, ones); \
  broadcast_fma_incrementIdx_16_f32(a, b, c9, broadcastIdx, ones); \
  broadcast_fma_incrementIdx_16_f32(a, b, cA, broadcastIdx, ones); \
  broadcast_fma_incrementIdx_16_f32(a, b, cB, broadcastIdx, ones); \
  broadcast_fma_incrementIdx_16_f32(a, b, cC, broadcastIdx, ones); \
  broadcast_fma_incrementIdx_16_f32(a, b, cD, broadcastIdx, ones); \
  broadcast_fma_incrementIdx_16_f32(a, b, cE, broadcastIdx, ones); \
  broadcast_fma_incrementIdx_16_f32(a, b, cF, broadcastIdx, ones); \
} ((void)0)

void MMF32(char *aCurr, char *bCurr, char *cCurr, int blockSize,
           MMF32Params *params, char *T) {
  int NBytes = params->NBytes;

  char* c[16] = {cCurr, cCurr+NBytes, cCurr+2*NBytes, cCurr+3*NBytes,
    cCurr+4*NBytes, cCurr+5*NBytes, cCurr+6*NBytes, cCurr+7*NBytes,
    cCurr+8*NBytes, cCurr+9*NBytes, cCurr+10*NBytes, cCurr+11*NBytes,
    cCurr+12*NBytes, cCurr+13*NBytes, cCurr+14*NBytes, cCurr+15*NBytes};

  unsigned short MMask = ~(((unsigned short)0xFFFF) <<
                               ((unsigned short)(params->MRemainder)));
  unsigned short NMask = ~(((unsigned short)0xFFFF) <<
                               ((unsigned short)(params->NRemainder)));

  register __mmask16 mMask = _mm512_int2mask(MMask);
  register __mmask16 nMask = _mm512_int2mask(NMask);
  register __m512i broadcastIdx = _mm512_setzero_epi32();
  register const __m512i ones = _mm512_loadu_epi32(params->onesMem);

  char *a = aCurr;
  char *b = bCurr;
  bool useT;
  char *TAddr;
  for (; a < params->ARowChunkEnd; a += 64) {
    useT = false;
    for (; b < params->BRowChunkEnd; b += 64) {
      char* kA = a;
      char* kB = b;
      TAddr = T;
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

      for (; kA < params->AEnd; kA += params->MBytes, kB += params->NBytes) {
#ifdef DEBUG
        printf("Both good: ");
	print(params->A, params->B, params->C, params->M, params->N, params->K,
              kA, kB, c, 0xFFFF, 0xFFFF);
#endif
        register __m512 kAVal;
        if (useT) {
          kAVal = _mm512_loadu_ps(TAddr);
        } else {
          kAVal = _mm512_loadu_ps(kA);
          _mm512_storeu_ps(TAddr, kAVal);
        }
        TAddr += 64;
        mm_16x16_f32(
	  kAVal, _mm512_loadu_ps(kB),
          c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, cA, cB, cC, cD, cE, cF,
          broadcastIdx, ones);
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
      if (blockSize == 0)
        return;

      for (int i = 0; i < 16; i++) {
        c[i] += 64;
      }
      //useT = true;
    }
    {
      if (params->NRemainder != 0) {
        char* kA = a;
        char* kB = b;
        TAddr = T;
        register __m512 c0 = _mm512_maskz_loadu_ps(nMask, c[0]);
        register __m512 c1 = _mm512_maskz_loadu_ps(nMask, c[1]);
        register __m512 c2 = _mm512_maskz_loadu_ps(nMask, c[2]);
        register __m512 c3 = _mm512_maskz_loadu_ps(nMask, c[3]);
        register __m512 c4 = _mm512_maskz_loadu_ps(nMask, c[4]);
        register __m512 c5 = _mm512_maskz_loadu_ps(nMask, c[5]);
        register __m512 c6 = _mm512_maskz_loadu_ps(nMask, c[6]);
        register __m512 c7 = _mm512_maskz_loadu_ps(nMask, c[7]);
        register __m512 c8 = _mm512_maskz_loadu_ps(nMask, c[8]);
        register __m512 c9 = _mm512_maskz_loadu_ps(nMask, c[9]);
        register __m512 cA = _mm512_maskz_loadu_ps(nMask, c[10]);
        register __m512 cB = _mm512_maskz_loadu_ps(nMask, c[11]);
        register __m512 cC = _mm512_maskz_loadu_ps(nMask, c[12]);
        register __m512 cD = _mm512_maskz_loadu_ps(nMask, c[13]);
        register __m512 cE = _mm512_maskz_loadu_ps(nMask, c[14]);
        register __m512 cF = _mm512_maskz_loadu_ps(nMask, c[15]);

        for (; kA < params->AEnd; kA += params->MBytes, kB += params->NBytes) {
#ifdef DEBUG
          printf("AAAA good: ");
          print(params->A, params->B, params->C, params->M, params->N,
                params->K, kA, kB, c, 0xFFFF, nMask);
#endif  
          register __m512 kAVal;
          if (useT) {
            kAVal = _mm512_loadu_ps(TAddr);
          } else {
            kAVal = _mm512_loadu_ps(kA);
            _mm512_storeu_ps(TAddr, kAVal);
          }
          TAddr += 64;
          mm_16x16_f32(
            kAVal, _mm512_maskz_loadu_ps(nMask, kB),
            c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, cA, cB, cC, cD, cE, cF,
            broadcastIdx, ones);
        }

        _mm512_mask_storeu_ps(c[0], nMask, c0);
        _mm512_mask_storeu_ps(c[1], nMask, c1);
        _mm512_mask_storeu_ps(c[2], nMask, c2);
        _mm512_mask_storeu_ps(c[3], nMask, c3);
        _mm512_mask_storeu_ps(c[4], nMask, c4);
        _mm512_mask_storeu_ps(c[5], nMask, c5);
        _mm512_mask_storeu_ps(c[6], nMask, c6);
        _mm512_mask_storeu_ps(c[7], nMask, c7);
        _mm512_mask_storeu_ps(c[8], nMask, c8);
        _mm512_mask_storeu_ps(c[9], nMask, c9);
        _mm512_mask_storeu_ps(c[10], nMask, cA);
        _mm512_mask_storeu_ps(c[11], nMask, cB);
        _mm512_mask_storeu_ps(c[12], nMask, cC);
        _mm512_mask_storeu_ps(c[13], nMask, cD);
        _mm512_mask_storeu_ps(c[14], nMask, cE);
        _mm512_mask_storeu_ps(c[15], nMask, cF);

        blockSize--;
        if (blockSize == 0)
          return;
      }

      for (int i = 0; i < 16; i++) {
        c[i] += params->CIncEndRow;
      }
      b = params->B;
    }
  }
  int MRemainder = params->MRemainder;
  if (MRemainder != 0) {
    useT = false;  
    for (; b < params->BRowChunkEnd; b += 64) {
      char* kA = a;
      char* kB = b;
      TAddr = T;
      register __m512 c0 = _mm512_loadu_ps(c[0]);
      register __m512 c1;
      register __m512 c2;
      register __m512 c3;
      register __m512 c4;
      register __m512 c5;
      register __m512 c6;
      register __m512 c7;
      register __m512 c8;
      register __m512 c9;
      register __m512 cA;
      register __m512 cB;
      register __m512 cC;
      register __m512 cD;
      register __m512 cE;
      register __m512 cF;
      if (MRemainder > 1) c1 = _mm512_loadu_ps(c[1]);
      if (MRemainder > 2) c2 = _mm512_loadu_ps(c[2]);
      if (MRemainder > 3) c3 = _mm512_loadu_ps(c[3]);
      if (MRemainder > 4) c4 = _mm512_loadu_ps(c[4]);
      if (MRemainder > 5) c5 = _mm512_loadu_ps(c[5]);
      if (MRemainder > 6) c6 = _mm512_loadu_ps(c[6]);
      if (MRemainder > 7) c7 = _mm512_loadu_ps(c[7]);
      if (MRemainder > 8) c8 = _mm512_loadu_ps(c[8]);
      if (MRemainder > 9) c9 = _mm512_loadu_ps(c[9]);
      if (MRemainder > 10) cA = _mm512_loadu_ps(c[10]);
      if (MRemainder > 11) cB = _mm512_loadu_ps(c[11]);
      if (MRemainder > 12) cC = _mm512_loadu_ps(c[12]);
      if (MRemainder > 13) cD = _mm512_loadu_ps(c[13]);
      if (MRemainder > 14) cE = _mm512_loadu_ps(c[14]);

      for (; kA < params->AEnd; kA += params->MBytes, kB += params->NBytes) {
#ifdef DEBUG
        printf("BBBB good: ");
        print(params->A, params->B, params->C, params->M, params->N, params->K,
              kA, kB, c, mMask, 0xFFFF);
#endif
        register __m512 kAVal;
        if (useT) {
          kAVal = _mm512_maskz_loadu_ps(mMask, TAddr);
        } else {
          kAVal = _mm512_maskz_loadu_ps(mMask, kA);
          _mm512_mask_storeu_ps(TAddr, mMask, kAVal);
        }
        TAddr += 64;
        mm_16x16_f32(
	  kAVal, _mm512_loadu_ps(kB),
          c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, cA, cB, cC, cD, cE, cF,
          broadcastIdx, ones);
      }

      _mm512_storeu_ps(c[0], c0);
      if (MRemainder > 1) _mm512_storeu_ps(c[1], c1);
      if (MRemainder > 2) _mm512_storeu_ps(c[2], c2);
      if (MRemainder > 3) _mm512_storeu_ps(c[3], c3);
      if (MRemainder > 4) _mm512_storeu_ps(c[4], c4);
      if (MRemainder > 5) _mm512_storeu_ps(c[5], c5);
      if (MRemainder > 6) _mm512_storeu_ps(c[6], c6);
      if (MRemainder > 7) _mm512_storeu_ps(c[7], c7);
      if (MRemainder > 8) _mm512_storeu_ps(c[8], c8);
      if (MRemainder > 9) _mm512_storeu_ps(c[9], c9);
      if (MRemainder > 10) _mm512_storeu_ps(c[10], cA);
      if (MRemainder > 11) _mm512_storeu_ps(c[11], cB);
      if (MRemainder > 12) _mm512_storeu_ps(c[12], cC);
      if (MRemainder > 13) _mm512_storeu_ps(c[13], cD);
      if (MRemainder > 14) _mm512_storeu_ps(c[14], cE);

      blockSize--;
      if (blockSize == 0)
        return;

      for (int i = 0; i < 16; i++) {
        c[i] += 64;
      }
      //useT = true;
    }
    {
      if (NMask != 0) {
        char* kA = a;
        char* kB = b;
        TAddr = T;
        register __m512 c0 = _mm512_maskz_loadu_ps(nMask, c[0]);
        register __m512 c1;
        register __m512 c2;
        register __m512 c3;
        register __m512 c4;
        register __m512 c5;
        register __m512 c6;
        register __m512 c7;
        register __m512 c8;
        register __m512 c9;
        register __m512 cA;
        register __m512 cB;
        register __m512 cC;
        register __m512 cD;
        register __m512 cE;
        register __m512 cF;
        if (MRemainder > 1) c1 = _mm512_maskz_loadu_ps(nMask, c[1]);
        if (MRemainder > 2) c2 = _mm512_maskz_loadu_ps(nMask, c[2]);
        if (MRemainder > 3) c3 = _mm512_maskz_loadu_ps(nMask, c[3]);
        if (MRemainder > 4) c4 = _mm512_maskz_loadu_ps(nMask, c[4]);
        if (MRemainder > 5) c5 = _mm512_maskz_loadu_ps(nMask, c[5]);
        if (MRemainder > 6) c6 = _mm512_maskz_loadu_ps(nMask, c[6]);
        if (MRemainder > 7) c7 = _mm512_maskz_loadu_ps(nMask, c[7]);
        if (MRemainder > 8) c8 = _mm512_maskz_loadu_ps(nMask, c[8]);
        if (MRemainder > 9) c9 = _mm512_maskz_loadu_ps(nMask, c[9]);
        if (MRemainder > 10) cA = _mm512_maskz_loadu_ps(nMask, c[10]);
        if (MRemainder > 11) cB = _mm512_maskz_loadu_ps(nMask, c[11]);
        if (MRemainder > 12) cC = _mm512_maskz_loadu_ps(nMask, c[12]);
        if (MRemainder > 13) cD = _mm512_maskz_loadu_ps(nMask, c[13]);
        if (MRemainder > 14) cE = _mm512_maskz_loadu_ps(nMask, c[14]);

        for (; kA < params->AEnd; kA += params->MBytes, kB += params->NBytes) {
#ifdef DEBUG
          printf("None good: ");
          print(params->A, params->B, params->C, params->M, params->N,
                params->K, kA, kB, c, mMask, nMask);
#endif  
          register __m512 kAVal;
          if (useT) {
            kAVal = _mm512_maskz_loadu_ps(mMask, TAddr);
          } else {
            kAVal = _mm512_maskz_loadu_ps(mMask, kA);
            _mm512_mask_storeu_ps(TAddr, mMask, kAVal);
          }
          TAddr += 64;
          mm_16x16_f32(
            kAVal, _mm512_maskz_loadu_ps(nMask, kB),
            c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, cA, cB, cC, cD, cE, cF,
            broadcastIdx, ones);
        }

        _mm512_mask_storeu_ps(c[0], nMask, c0);
        if (MRemainder > 1) _mm512_mask_storeu_ps(c[1], nMask, c1);
        if (MRemainder > 2) _mm512_mask_storeu_ps(c[2], nMask, c2);
        if (MRemainder > 3) _mm512_mask_storeu_ps(c[3], nMask, c3);
        if (MRemainder > 4) _mm512_mask_storeu_ps(c[4], nMask, c4);
        if (MRemainder > 5) _mm512_mask_storeu_ps(c[5], nMask, c5);
        if (MRemainder > 6) _mm512_mask_storeu_ps(c[6], nMask, c6);
        if (MRemainder > 7) _mm512_mask_storeu_ps(c[7], nMask, c7);
        if (MRemainder > 8) _mm512_mask_storeu_ps(c[8], nMask, c8);
        if (MRemainder > 9) _mm512_mask_storeu_ps(c[9], nMask, c9);
        if (MRemainder > 10) _mm512_mask_storeu_ps(c[10], nMask, cA);
        if (MRemainder > 11) _mm512_mask_storeu_ps(c[11], nMask, cB);
        if (MRemainder > 12) _mm512_mask_storeu_ps(c[12], nMask, cC);
        if (MRemainder > 13) _mm512_mask_storeu_ps(c[13], nMask, cD);
        if (MRemainder > 14) _mm512_mask_storeu_ps(c[14], nMask, cE);
      }
    }
  }
}

void MMF32Full(char* A, char* B, char* C, int M, int N, int K, int numThreads, char *T) {
  std::vector<std::thread> threads;
  threads.reserve(numThreads);
  MMF32Params params(A, B, C, M, N, K);
  int MTiles = (M+15)>>4;
  int NTiles = (N+15)>>4;
  int MNTiles = MTiles*NTiles;
  int blockSize = (MNTiles+numThreads-1)/numThreads;
  char *a;
  char *b;
  char *c;
  int blockSizeBytes = blockSize<<6;
#ifdef DEBUG
  printf("blockSize: %d, MTiles: %d, NTiles: %d\n", blockSize, MTiles, NTiles);
#endif
  char *localT = T;
  for (int j = 0; j < MNTiles; j += blockSize, localT += K*64) {
    int aBlockBytes = (j/NTiles)<<6;
    int bBlockBytes = (j%NTiles)<<6;
    a = A + aBlockBytes;
    b = B + bBlockBytes;
    c = C + (aBlockBytes*N) + bBlockBytes;
#ifdef DEBUG
    printf("Call: %d %d A(%d %d) B(%d %d) C(%d %d)\n", aBlockBytes>>6, bBlockBytes>>6,
      (aBlockBytes>>2)/M, (aBlockBytes>>2)%M,
      (bBlockBytes>>2)/N, (bBlockBytes>>2)%N,
      ((aBlockBytes*N+bBlockBytes)>>2)/N, ((aBlockBytes*+bBlockBytes)>>2)%N
    );
#endif
    threads.push_back(std::thread(MMF32, a, b, c, blockSize, &params, localT));
  }
  for (auto &th: threads) {
    th.join();
  }
}

