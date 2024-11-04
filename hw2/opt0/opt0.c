//This File experiments with runtime speed up for matrix vector multiplication
//HW 2: OpenMP
//Control for HW2: row major and column major results without threading
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

//The unimproved control
void matVecMult(Matrix A, Matrix B, Matrix C, int rows, int cols) {
  int i, k;

  for (k = 0; k < cols; k++)
    for (i = 0; i < rows; i++)
      C[i] += A[i * cols + k] * B[k];
}

//Experiment from HW1: Locality. We need to call our matrix in row major order or else we'll miss the cache
//Result: major speedup
void rowMajor(Matrix A, Matrix B, Matrix C, int rows, int cols){

  for(int i = 0; i < rows; i++){
    for(int k = 0; k < cols; k++){
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
  FILE* file = fopen("results_control.csv", "w");
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
    //width of vector
    int p = 1;

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
      //exectution time for each experiment. time1 and time2 are start and end times for each test
      double t=0, time1=0, time2=0, t2=0;
      //check C[n/2] to make sure optimizations haven't affected what number we calculate
      double errorcheck=0,errorcheck2=0;
      //run experiments ten times and take average
      for(int k = 0; k < 3; k++){
        //control run 
        time1 = microtime();
        matVecMult(A,B,C,n,m);
        time2 = microtime();
        t = t + (time2 - time1);
	errorcheck = (double) C[n/2];
        memset(C,0,n*sizeof(C[0]));
	//row major order run
        time1 = microtime();
        rowMajor(A,B,C,n,m);
        time2 = microtime();
        t2 = t2 + (time2 - time1);
	errorcheck2 = (double) C[n/2];
        memset(C,0,n*sizeof(C[0]));
      }
      t = t/3;
      t2 = t2/3;

      // Print results for this problem size and thread number
      FILE* file = fopen("results_control.csv", "a");
      if (file == NULL) {
        fprintf(stderr, "Could not open file %s for appending\n", "results.csv");
        exit(EXIT_FAILURE);
      }
      fprintf(file,"Unoptimized,%d,%d,%g,%g\n",problemSizes[i],numThreads[j],t,errorcheck);
      fprintf(file,"RowMajor,%d,%d,%g,%g\n",problemSizes[i],numThreads[j],t2,errorcheck2);
      fclose(file);
   }

   closeExperiment(A,B,C);
  }

  return 0;
}
