#include <stdio.h>
#include <stdlib.h>
#include <omp.h>


int main(int argc, char** argv) {
  int i, size = 100000000;
  float *A, *B, *C;
  double sum = 0.0;
  //256 bit / 32 byte registers used by default on hmcse server
  alignment = 32;
  // TODO Allocated aligned memory for A, B, and C
  posix_memalign((void**)&A, alignment, size*sizeof(float))
  posix_memalign((void**)&B, alignment, size*sizeof(float))
  posix_memalign((void**)&C, alignment, size*sizeof(float))

  //begin intialization
  if ((A == 0) || (B == 0) || (C == 0)) {
    fprintf(stderr, "Memory allocation failed in file %s, line %d\n", __FILE__,
            __LINE__);
    exit(1);
  }
  for (i = 0; i < size; i++) {
    B[i] = i * 0.00002;
    C[i] = -i * 0.00003;
  }
  //end initialization
  //begin computation
  int numThreads = 10;
  int chunkSize = size/numThreads;
#pragma omp parallel for num_threads(numThreads) schedule(static,chunkSize)
  for (i = 0; i < size; i++) {
    A[i] += C[i] * B[i] * B[i] * 1000.0;
    sum += A[i];
  }
  printf("Sum: %f\n", sum);
  return 0;
}
