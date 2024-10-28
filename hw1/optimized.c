//This File experiments with runtime speed up for matrix vector multiplication
//Major speedup is realized with applying locality rules -- since the 1d matrix representation is intialized row-major, it must be read row-major
//in order to satisfy locality
//Author: Jacob Sander, course 5522 
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
//unsuccessful; merely adds time to compute. Likely due to OFast already implementing unrolling automatically 
void matVecMultParallelInstruction(Matrix A, Matrix B, Matrix C, int rows, int cols){
  for(int i = 0; i < rows; i++){
    
    //unroll 2 operations inside for loop
    for (int k = 0; k < cols - 1; k += 2) {
      C[k] += A[i * cols + k] * B[k];
      C[k + 1] += A[i * cols + k + 1] * B[k+1];
    }
  }
}


//Experiment 1: save matrix b's value at i,k to reduce calls to load it
//Successful speedup, modest 
void matVecMultCache(Matrix A, Matrix B, Matrix C, int rows, int cols){
  for(int k = 0; k<cols; k++){
    double cache_b = B[k];
    for(int i = 0; i<rows; i++){
      C[i] += A[i*cols + k] * cache_b;
    }
  }
}

//Experiment 2: Locality. We need to call our matrix in row major order or else we'll miss the cache
void matVecMultCacheLocal(Matrix A, Matrix B, Matrix C, int rows, int cols){
  for(int i = 0; i<rows; i++){
    for(int k = 0; k<cols; k++){
      C[i] += A[i*cols + k] * B[k];
    }
  }
}

//Insight: we can do something similar to help cache A -- it references data that is non local (i*cols doesn't access sequentially)
//helper function for experiment below
//Return a col vector that can be scanned sequentially
double* getCol(Matrix A, int k, int rows, int cols){
  double* rtnPtr = malloc(rows*sizeof(double));
  
  for(int i = 0; i < rows; i++){
    rtnPtr[i] = A[i*cols + k];
  }
  
  return rtnPtr;
}

//Experiment 3: get column of matrix so that future references have sequential access
//dead end experiment - no speedup
void matVecMultCacheLocalityV2(Matrix A, Matrix B, Matrix C, int rows, int cols){
  
  for(int k = 0; k < rows; k++){
    double cache_b = B[k];
    double* colCopy = getCol(A,k,rows,cols);
    
    for(int i = 0; i < cols; i++){
      C[i] += colCopy[i] * cache_b;
    }
    free(colCopy);
  }
}

//take 3 arguments for three different experiment sizes of matrices
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
    double t_exp1, t_exp2, t_exp3 = 0;
    
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
      matVecMultCacheLocal(A,B,C,n,m);
      time2 = microtime();
      t_exp2 = t_exp2 + (time2 - time1);

      time1 = microtime();
      matVecMultParallelInstruction(A,B,C,n,m);
      time2 = microtime();
      t_exp3 = t_exp3 + (time2 - time1);
    }
    t = t/3;
    t_exp1 = t_exp1/3;
    t_exp2 = t_exp2/3;
    t_exp3 = t_exp3/3;
    // Print results
    // Print Control
 
    FILE* file = fopen("results.csv", "a");
    if (file == NULL) {
        fprintf(stderr, "Could not open file %s for writing\n", "results.csv");
        exit(EXIT_FAILURE);
    }
    fprintf(file,"Experiment,ProblemSize,Time (us)\n");
    fprintf(file,"Control,%d,%g\n",problemSizes[i],t);
    fprintf(file,"CacheB,%d,%g\n",problemSizes[i],t_exp1);
    fprintf(file,"RowMajor,%d,%g\n",problemSizes[i],t_exp2);
    fprintf(file,"ParallelInstruct,%d,%g\n",problemSizes[i],t_exp3);
    fclose(file);
    freeMatrix(A);
    freeMatrix(B);
    freeMatrix(C);
  }


  return 0;
}
