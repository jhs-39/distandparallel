//This File runs 2 different optimization experiments on matrix-vector multiplication by CPU
//The first attempts instruction-level parallelism with unrolling
//The second caches values of matrix B to prevent re-loading them

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

//The unimproved control
void matVecMult(Matrix A, Matrix B, Matrix C, int rows, int cols) {
  int i, k;

  memset(C, 0, rows * sizeof(C[0]));

  for (k = 0; k < cols; k++)
    for (i = 0; i < rows; i++)
      C[i] += A[i * cols + k] * B[k];
}

//instruction level parallelism by performing 2 operations in each loop instead of 1
//unsuccessful; merely adds time to compute. Likely due to OFast already implementing unrolling in 
void matVecMultParallelInstruction(Matrix A, Matrix B, Matrix C, int rows, int cols){
  for(int k = 0; k < cols; k++){
    double cache_b = B[k];
    
    //unroll 2 operations inside for loop
    for (int i = 0; i < rows - 1; i += 2) {
      C[i] += A[i * cols + k] * cache_b;
      C[i + 1] += A[(i + 1) * cols + k] * cache_b;
    }
    
    //Handle last col if odd
    if (rows % 2 != 0) {
      C[rows - 1] += A[(rows - 1) * cols + k] * cache_b;
    }
  }
}


//Experiment 1: save matrix b's value
//Successful speedup 
void matVecMultCache(Matrix A, Matrix B, Matrix C, int rows, int cols){
  for(int k = 0; k<cols; k++){
    double cache_b = B[k];
    for(int i = 0; i<rows; i++){
      C[i] += A[i*cols + k] * cache_b;
    }
  }
}

//Insight: we can do something similar to help cache A -- it references data that is non local (i*cols doesn't access sequentially)
//helper function for experiment below
//Return a row vector that can be scanned sequentially
double* getRow(Matrix A, int k, int cols){
  double* rtnPtr = malloc(cols*sizeof(double));
  
  for(int i = 0; i < cols; i++){
    rtnPtr[i] = A[i*cols + k];
  }
  
  return rtnPtr;
}

//caching the next step in memory as a form of instruction parallelism; reduced need to reload B[k]
//result: dramatic speedup
void matVecMultCacheLocality(Matrix A, Matrix B, Matrix C, int rows, int cols){
  for(int k=0;k<cols;k++){
    double cache_b = B[k];
    double* rowCopy = malloc(rows*sizeof(double));
    rowCopy = getRow(A,k,cols);
    for(int i = 0; i < rows; i++){
      C[i] += rowCopy[i] * cache_b;
    }
    free(rowCopy);
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
    double t, time1, time2 = 0;
    double t_exp1, t_exp2 = 0;
    
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
      matVecMultCache(A,B,C,n,m);
      time2 = microtime();
      t_exp1 = t_exp1 + (time2 - time1);

      time1 = microtime();
      matVecMultCacheLocality(A,B,C,n,m);
      time2 = microtime();
      t_exp2 = t_exp2 + (time2 - time1);
    }
    t = t/3;
    t_exp1 = t_exp1/3;
    t_exp2 = t_exp2/3;
    // Print results
    // Print Control
    printf("Unoptimized Control, Avg of 3 Trials\n");
    printf("\nTime = %g us\n", t);
    printf("Timer Resolution = %g us\n", getMicrotimeResolution());
    printf("Performance = %g Gflop/s\n", 2.0 * n * m * 1e-3 / t);
    printf("C[N/2] = %g\n\n", (double)C[n / 2]);
    
    printf("Caching B\n");
    printf("\nTime = %g us \n", t_exp1);
    printf("Timer Resolution = %g us \n", getMicrotimeResolution());
    printf("Performance = %g Gflop/s\n", 2.0 * n * m * 1e-3/t_exp1);
    printf("C[N/2] = %g\n\n", (double)C[n/2]);

    printf("Caching B and A\n");
    printf("\nTime = %g us \n", t_exp2);
    printf("Performance = %g Gflop/s\n", 2.0 * n * m * 1e-3/t_exp2);
    printf("C[N/2] = %g\n\n", (double)C[n/2]);
    freeMatrix(A);
    freeMatrix(B);
    freeMatrix(C);
  }


  return 0;
}
