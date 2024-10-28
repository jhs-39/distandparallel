#include <microtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef float* Matrix;

Matrix createMatrix(int rows, int cols) {
  Matrix M;

  M = (Matrix)malloc(rows * cols * sizeof(M[0]));
  if (M == 0)
    fprintf(stderr, "Matrix allocation failed in file %s, line %d\n", __FILE__,
            __LINE__);

  return M;
}

void freeMatrix(Matrix M) {
  if (M) free(M);
}

void initMatrix(Matrix A, int rows, int cols) {
  int i, j;

  for (i = 0; i < rows; i++)
    for (j = 0; j < cols; j++) A[i * cols + j] = 1.0 / (i + j + 2);
}

// This can be improved!
void matVecMult(Matrix A, Matrix B, Matrix C, int rows, int cols) {
  int i, k;

  memset(C, 0, rows * sizeof(C[0]));

  for (k = 0; k < cols; k++)
    for (i = 0; i < rows; i++)
      C[i] += A[i * cols + k] * B[k];
}

void matVecMultParallelInstruction(Matrix A, Matrix B, Matrix C, int rows, int cols){
//instruction level parallelism by performing 4 operations in each loop instead of 1
  for(int k = 0; k < cols; k++){
    for (int i = 0; i < rows; i += 4) {
      C[i] += A[i * cols + k] * B[k];
      if (i + 1 < rows) C[i + 1] += A[(i + 1) * cols + k] * B[k];
      if (i + 2 < rows) C[i + 2] += A[(i + 2) * cols + k] * B[k];
      if (i + 3 < rows) C[i + 3] += A[(i + 3) * cols + k] * B[k];
    }
  }
}

int main(int argc, char** argv) {
  
  if (argc != 4){
    fprintf(stderr,"Usage: %s exp1size exp2size exp3size", argv[0]);
  }	  
    	
  int problemSizes[3];
  for (int i = 0; i < 3; i++) {
      problemSizes[i] = atoi(argv[i + 1]);
  }

  for(int i = 0; i < 3; i++){
    int n, m, p = 1;
    Matrix A, B, C;
    double t, time1, time2;
    double t_exp1, t_exp2;
    n = problemSizes[i];
    m = problemSizes[i];

    A = createMatrix(n,m);
    B = createMatrix(m,p);
    C = createMatrix(n,p);
    
    initMatrix(A,n,m);
    initMatrix(B,m,p);
    memset(C,0,n*p*sizeof(C[0]));
    
    //run experiment three times and take average
    for(int k = 0; k < 3; k++){
      time1 = microtime();
      matVecMult(A,B,C,n,m);
      time2 = microtime();
      t = t + (time2 - time1);

      time1 = microtime();
      matVecMultParallelInstruction(A,B,C,n,m);
      time2 = microtime();
      t_exp1 = t_exp1 + (time2 - time1);
    }
    t = t/3;
    t_exp1 = t_exp1/3;
    // Print results
    // Print Control
    printf("Unoptimized Control, Avg of 3 Trials\n");
    printf("\nTime = %g us\n", t);
    printf("Timer Resolution = %g us\n", getMicrotimeResolution());
    printf("Performance = %g Gflop/s\n", 2.0 * n * m * 1e-3 / t);
    printf("C[N/2] = %g\n\n", (double)C[n / 2]);
    
    printf("Instruction Parallelism; Unrolled for loop\n");
    printf("\nTime = %g us \n", t_exp1);
    printf("Timer Resolution = %g us \n", getMicrotimeResolution());
    printf("Performance = %g Gflop/s\n", 2.0 * n * m * 1e-3/t_exp1);
    printf("C[N/2] = %g\n\n", (double)C[n/2]);
    freeMatrix(A);
    freeMatrix(B);
    freeMatrix(C);
  }


  return 0;
}
