#include <microtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

typedef float* Matrix;

Matrix createMatrix(int rows, int cols) {
  Matrix M;

  M = (Matrix)malloc(rows * cols * sizeof(double));
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
    for (j = 0; j < cols; j++)
      A[i * cols + j] = 1.0 / (i + j + 2);
}

//per 'customer' request -- use a defined data type here even though we only use it to scatter rows later
void Build_mpi_type(int cols, MPI_Datatype* row_type) {
    MPI_Type_contiguous(cols, MPI_DOUBLE, row_type); // Create contiguous datatype for a row
    MPI_Type_commit(row_type);
}

//construct the input for message passing
void Get_input(
      int      my_rank  /* in  */, 
      int      comm_sz  /* in  */, 
      int      n        /* in; num of rows in a*/,
      int      m        /* in; num of cols in a*/,
      Matrix  A        /* in; the input matrix*/,
      Matrix  B        /* in; the input vector*/,
      Matrix  local_a  /* out; local portion of matrix A */,
      Matrix  local_c   /* out; local portion of results vector C*/) {
   
   MPI_Datatype input_mpi_t;
   Build_mpi_type(m,&input_mpi_t);
   
   int local_rows = n / comm_sz; // Rows each process will handle

   //scatter matrix a among threads
   int err = MPI_Scatter(A, local_rows, input_mpi_t, local_a, local_rows, input_mpi_t, 0, MPI_COMM_WORLD);
   if (err != MPI_SUCCESS) {
    fprintf(stderr, "MPI_Scatter failed\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
   }

   //broadcast vector to all threads. Dont use row custom type
   int err_bcast = MPI_Bcast(B, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
   if (err_bcast != MPI_SUCCESS) {
     fprintf(stderr, "MPI_Bcast failed\n");
     MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
   }

   MPI_Type_free(&input_mpi_t);
}

void mpiVecMult(Matrix A, Matrix B, Matrix C, int rows, int cols){
  //memset C results to 0
  memset(C,0,rows*sizeof(C[0]));
  //threading vars
  int my_rank;
  int comm_sz;

  MPI_Init(NULL,NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

  if (comm_sz <= 0) {
    fprintf(stderr, "Error: Invalid number of MPI processes.\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
}

  int localRows = rows/comm_sz;

  Matrix local_A = createMatrix(localRows,cols);
  Matrix local_C = createMatrix(localRows,1);
  memset(local_A, 0, localRows * cols * sizeof(local_A[0]));
  memset(local_C, 0, localRows * sizeof(local_C[0]));

  if (local_A == NULL || local_C == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }  

  //set up thread communication & distribute data
  Get_input(my_rank, comm_sz, rows, cols, A, B, local_A, local_C);

  for(int i = 0; i < localRows; i++){
    for(int j = 0; j < cols; j++){
      local_C[i] += local_A[i*cols + j] * B[j]; 
    }
  }
  
  MPI_Gather(local_C, localRows, MPI_DOUBLE, C, localRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  free(local_A);
  free(local_C);

  MPI_Finalize();
}

int main(int argc, char** argv) {
  FILE *file = freopen("error_log.txt", "w", stderr);
  if (file == NULL) {
    perror("Failed to redirect stderr");
    return EXIT_FAILURE;
  }		
	
  int n, m, p = 1;
  Matrix A, B, C;
  double t, time1, time2;

  if (argc != 3) {
    fprintf(stderr, "USAGE: %s rows cols\n", argv[0]);
    exit(1);
  }

  n = atoi(argv[1]);
  m = atoi(argv[2]);

  A = createMatrix(n, m);
  B = createMatrix(m, p);
  C = createMatrix(n, p);

  initMatrix(A, n, m);
  initMatrix(B, m, p);
  memset(C, 0, n * p * sizeof(C[0]));

  // measure time
  time1 = microtime();
  mpiVecMult(A, B, C, n, m);
  time2 = microtime();

  t = time2 - time1;

  // Print results
  printf("\nTime = %g us\n", t);
  printf("Timer Resolution = %g us\n", getMicrotimeResolution());
  printf("Performance = %g Gflop/s\n", 2.0 * n * n * 1e-3 / t);
  printf("C[N/2] = %g\n\n", (double)C[n / 2]);

  freeMatrix(A);
  freeMatrix(B);
  freeMatrix(C);

  return 0;
}
