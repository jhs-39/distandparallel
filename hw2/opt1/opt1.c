//This File experiments with runtime speed up for matrix vector multiplication
//HW 2: OpenMP
//Opt1: Apply threading to rows
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
  //set to null to make dangling ptr more safe
  M = NULL;
}

void initMatrix(Matrix A, int rows, int cols) {
  int i, j;

  for (i = 0; i < rows; i++)
    for (j = 0; j < cols; j++) A[i * cols + j] = 1.0 / (i + j + 2);
}

//helper function for freeing components after experiments
void closeExperiment(Matrix A, Matrix B, Matrix C){
  freeMatrix(A);
  freeMatrix(B);
  freeMatrix(C);
}

//Experiment HW2: Weak Optimized. I theorize that using OpenMP on the row level will constitute a weak speedup as it is less parallel than calling it more often on the column loops
//Results: 
void WeakOpenMP(Matrix A, Matrix B, Matrix C, int rows, int cols, int numThreads){
  //initialize iterators outside threading for proper scope
  int i = 0, k = 0;
#pragma omp parallel for num_threads(numThreads) default(none) private(i,k) shared(A,B,C,rows,cols)
  for(i = 0; i < rows; i++){
    double localC = 0;
    for(k = 0; k < cols; k++){
      localC += A[i*cols + k] * B[k];
    }

    C[i] = localC;
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
  //try 4 different numbers of threads
  int numThreads[4];
  int threadNum = 1;
  for(int i = 0; i < 4; i++){
    numThreads[i] = threadNum;
    threadNum *= 2; 
  }

  //prepare results file
  FILE* file = fopen("results_opt1.csv", "w");
  if (file == NULL) {
    fprintf(stderr, "Could not open file %s for writing\n", "results.csv");
    exit(EXIT_FAILURE);
  }
  fprintf(file,"Experiment,ProblemSize,NumberThreads,Time (us),tSerial (us),ErrorCheck\n");
  fclose(file);
  
  //Loop through job sizes
  for(int i = 0; i < 3; i++){
    //size of row
    int n = problemSizes[i];
    //size of col
    int m = problemSizes[i];
    //width of vector
    int p = 1;
    double tSerial = 0;

    //Matrix init for this job size
    Matrix A,B,C;
    A = createMatrix(n,m);
    B = createMatrix(m,p);
    C = createMatrix(n,p);
    
    initMatrix(A,n,m);
    initMatrix(B,m,p);
    memset(C, 0, n * p * sizeof(C[0]));

    //loop through number threads
    for(int j = 0; j < 4; j++){
      //exectution time for experiment. Make sure to record tserial for later calculations
      double t=0, time1=0, time2=0;
      //check C[n/2] to make sure optimizations haven't affected what number we calculate
      double errorcheck=0;
      //run experiments three times and take average
      for(int k = 0; k < 3; k++){
	//parallelize row ops
        time1 = microtime();
        WeakOpenMP(A,B,C,n,m,numThreads[j]);
        time2 = microtime();
        t = t + (time2 - time1);
	errorcheck = (double) C[n/2];
      }
      t = t/3;

      //record tserial if threads == 1
      if(numThreads[j] == 1){
      	tSerial = t;
      }

      // Print results for this problem size and thread number
      FILE* file = fopen("results_opt1.csv", "a");
      if (file == NULL) {
        fprintf(stderr, "Could not open file %s for appending\n", "results.csv");
        exit(EXIT_FAILURE);
      }
      fprintf(file,"RowOpenMP,%d,%d,%g,%g,%g,\n",problemSizes[i],numThreads[j],t,tSerial,errorcheck);
      fclose(file);
   }

   closeExperiment(A,B,C);
  }

  return 0;
}
