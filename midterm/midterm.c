#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>


int main(int argc, char** argv) {
  int i, size = 100000000;
  float *A, *B, *C;
  double sum = 0.0;

  //Allocate aligned memory for A, B, and C and set to 0
  //32 byte alignment/256 bit register
  int alignment = 32;
  posix_memalign((void**)&A, alignment, 3 * size * sizeof(float));
  memset(A, 0, 3 * size * sizeof(float));

  B = &A[size];
  C = &A[2*size];
  
  if ((A == 0) || (B == 0) || (C == 0)) {
    fprintf(stderr, "Memory allocation failed in file %s, line %d\n", __FILE__,
            __LINE__);
    exit(1);
  }
  for (i = 0; i < size; i++) {
    B[i] = i * 0.00002;
    C[i] = -i * 0.00003;
  }


  int threadNum = 10;
  int chunkSize = 16;
  printf("Chunk Size: %i\n",chunkSize);
#pragma omp parallel for num_threads(threadNum) schedule(static, chunkSize) reduction(+:sum)
  for (int i = 0; i < size; i++){
    A[i] += C[i] * B[i] * B[i] * 1000.0;
    sum += A[i];
  }
  printf("Sum: %f\n", sum);
  return 0;
}
