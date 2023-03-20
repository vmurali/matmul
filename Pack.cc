#include "Pack.h"

#include <algorithm>

float * PackLeft(float* A, int Morig, int Korig) {
  int MTiles = (Morig+15)>>4;
  int KTiles = (Korig+15)>>4;
  int M = MTiles << 4;
  int K = KTiles << 4;
  float *c = new float[M*K<<2];
  int Mremain = M - Morig;
  int Kremain = K - Korig;

  int i = 0;
  for (int rowStep = 0; rowStep < Morig; rowStep += 16) {
    for (int colStep = 0; colStep < Korig; colStep += 16) {
      int colMin = std::min(colStep+16, Korig);
      int colCount = 0;
      for (int col = colStep; col < colMin; col++, colCount++) {
        int rowMin = std::min(rowStep+16, Morig);
        int rowCount = 0;
        for (int row = rowStep, rowCount = 0; row < rowMin; row++, rowCount++) {
          c[i] = *(A + row*Korig + col);
	  i++;
	}
	for (; rowCount < 16; rowCount++) {
          c[i] = 0;
	  i++;
	}
      }
      for (; colCount < 16; colCount++) {
        for (int row = 0; row < 16; row++) {
          c[i] = 0;
	  i++;
	}
      }
    }
  }
  return c;
}

float * PackRight(float* B, int Norig, int Korig) {
  int NTiles = (Norig+15)>>4;
  int KTiles = (Korig+15)>>4;
  int N = NTiles << 4;
  int K = KTiles << 4;
  float *c = new float[N*K<<2];
  int Nremain = N - Norig;
  int Kremain = K - Korig;

  int i = 0;
  for (int colStep = 0; colStep < Norig; colStep += 16) {
    for (int rowStep = 0; rowStep < Korig; rowStep += 16) {
      int rowMin = std::min(rowStep+16, Korig);
      int rowCount = 0;
      for (int row = rowStep; row < rowMin; row++, rowCount++) {
        int colMin = std::min(colStep+16, Norig);
        int colCount = 0;
        for (int col = colStep, colCount = 0; col < colMin; col++, colCount++) {
          c[i] = *(B + row*Norig + col);
	  i++;
	}
	for (; colCount < 16; colCount++) {
          c[i] = 0;
	  i++;
	}
      }
      for (; rowCount < 16; rowCount++) {
        for (int col = 0; col < 16; col++) {
          c[i] = 0;
	  i++;
	}
      }
    }
  }
  return c;
}

