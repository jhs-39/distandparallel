//This File experiments with runtime speed up for matrix vector multiplication
//HW 2: OpenMP
//in order to satisfy locality
//Author: Jacob Sander, course 5522 
#include <microtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

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


//helper function for resetting matrices between experiments
void initExperiment(Matrix A, Matrix B, Matrix C, int n, int m, int p){
  A = createMatrix(n,m);
  B = createMatrix(m,p);
  C = createMatrix(n,p);

  initMatrix(A,n,m);
  initMatrix(B,m,p);
  memset(C,0,n*p*sizeof(C[0]));
}

//helper function for freeing components after experiments
void closeExperiment(Matrix A, Matrix B, Matrix C){
  freeMatrix(A);
  freeMatrix(B);
  freeMatrix(C);
}

//The unimproved control
void matVecMult(Matrix A, Matrix B, Matrix C, int rows, int cols) {
  int i, k;

  memset(C, 0, rows * sizeof(C[0]));

  for (k = 0; k < cols; k++)
    for (i = 0; i < rows; i++)
      C[i] += A[i * cols + k] * B[k];
}

//Experiment from HW1: Locality. We need to call our matrix in row major order or else we'll miss the cache
//Result: major speedup
void rowMajor(Matrix A, Matrix B, Matrix C, int rows, int cols){
  for(int i = 0; i<rows; i++){
    for(int k = 0; k<cols; k++){
      C[i] += A[i*cols + k] * B[k];
    }
  }
}

//Experiment HW2: Weak Optimized. I theorize that using OpenMP on the row level will constitute a weak speedup
//Results: 
void WeakOpenMP(Matrix A, Matrix B, Matrix C, int rows, int cols, int numThreads){
#pragma omp parallel num_threads(numThreads)
  for(int i = 0; i < rows; i++){
    for(int k = 0; k < cols; k++){
      C[i] += A[i*cols + k] * B[k];
    }
  }
}

//Experiment HW2: Strong Optimized. I theorize that using OpenMP on the column level will constitute a strong speedup
void StrongOpenMP(Matrix A, Matrix B, Matrix C, int rows, int cols, int numThreads){
  for(int i = 0; i < rows; i++){
#pragma omp parallel num_threads(numThreads)
    for(int k = 0; k< cols; k++){
      C[i] += A[i*cols + k] * B[k];
    }
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
  int numThreads[4];
  int threadNum = 1;
  for(int i = 0; i < 4; i++){
    numThreads[i] = threadNum;
    threadNum *= 2; 
  }

  //prepare results file
  FILE* file = fopen("results.csv", "w");
  if (file == NULL) {
    fprintf(stderr, "Could not open file %s for writing\n", "results.csv");
    exit(EXIT_FAILURE);
  }
  fprintf(file,"Experiment,ProblemSize,NumberThreads,Time (us),ErrorCheck\n");
  fclose(file);
  
  //Loop through job sizes
  for(int i = 0; i < 3; i++){
    //size of row
    int n = problemSizes[i];
    //size of col
    int m = problemSizes[i];

    //loop through number threads
    for(int j = 0; j < 4; j++){
      //vector width
      int p = 1;
      //matrix A by vector B to yield vector C
      Matrix A, B, C;
      double t=0, time1=0, time2=0, t_exp1=0, t_exp2=0, t_exp3=0;

      //run experiments three times and take average
      for(int k = 0; k < 3; k++){
        //control run    
	initExperiment(A,B,C,n,m,p);
        time1 = microtime();
        matVecMult(A,B,C,n,m);
        time2 = microtime();
        t = t + (time2 - time1);
	errorcheck_control = (double) C[n/2];
	closeExperiment(A,B,C);
        //row major order run
	initExperiment(A,B,C,n,m,p);
        time1 = microtime();
        rowMajor(A,B,C,n,m);
        time2 = microtime();
        t_exp1 = t_exp1 + (time2 - time1);
	errorcheck_rowmajor = (double) C[n/2];
	closeExperiment(A,B,C);
        //parallelize row ops
	initExperiment(A,B,C,n,m,p);
        time1 = microtime();
        WeakOpenMP(A,B,C,n,m,j);
        time2 = microtime();
        t_exp2 = t_exp2 + (time2 - time1);
	errorcheck_weakmp = (double) C[n/2];
	closeExperiment(A,B,C);
        //parallelize column ops
	initExperiment(A,B,C,n,m,p);
        time1 = microtime();
        StrongOpenMP(A,B,C,n,m,j);
        time2 = microtime();
        t_exp3 = t_exp3 + (time2 - time1);
	errorcheck_strongmp = (double) C[n/2];
	closeExperiment(A,B,C);
      }
      t = t/3;
      t_exp1 = t_exp1/3;
      t_exp2 = t_exp2/3;
      t_exp3 = t_exp3/3;
      // Print results for this problem size and thread number
      FILE* file = fopen("results.csv", "a");
      if (file == NULL) {
        fprintf(stderr, "Could not open file %s for appending\n", "results.csv");
        exit(EXIT_FAILURE);
      }
      fprintf(file,"Unoptimized,%d,%d,%g\n",problemSizes[i],numThreads[j],t,errorcheck_control);
      fprintf(file,"RowMajor,%d,%d,%g\n",problemSizes[i],numThreads[j],t_exp1,errorcheck_rowmajor);
      fprintf(file,"WeakOpenMP,%d,%d,%g\n",problemSizes[i],numThreads[j],t_exp2,errorcheck_weakmp);
      fprintf(file,"StrongOpenMP,%d,%d,%g\n",problemSizes[i],numThreads[j],t_exp3,errorcheck_strongmp);
      fprintf("C[N/2] = %g\n\n", (double)C[n / 2]);
      fclose(file);
   }
  }

  return 0;
}
