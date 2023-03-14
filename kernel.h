#include <vector>
#include <thread>

struct MMF32Params {
  char *A;
  char *B;
  char *C;
  int M;
  int N;
  int K;
  int MRemainder;
  int NRemainder;
  int MBytes;
  int NBytes;
  char *ARowChunkEnd;
  char *BRowChunkEnd;
  int CIncEndRow;
  char *AEnd;
  const int onesMem[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  MMF32Params(char *_A, char *_B, char *_C, int _M, int _N, int _K)
    : A(_A), B(_B), C(_C), M(_M), N(_N), K(_K),
      MRemainder(M & 0xF),
      NRemainder(N & 0xF),
      MBytes(M << 2),
      NBytes(N << 2),
      ARowChunkEnd(A + ((M>>4)<<6)),
      BRowChunkEnd(B + ((N>>4)<<6)),
      CIncEndRow((NRemainder<<2) + 15*NBytes),
      AEnd(A + (MBytes*K)) {}
};


void mm_f32_full(char* A, char* B, char* C, int M, int N, int K,
                 int numThreads, std::vector<std::thread> &threads);
