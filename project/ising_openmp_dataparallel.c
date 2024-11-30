#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include "microtime.h"
#include "ising_model.h"

//parallelize the calculation by dividing matrix into submatrices; then, locking and unlocking only occur on boundary
//Will have interesting topology on problem size -- 'surface to volume' ratio
void ising_openmp_dataparallel(int **lattice, int L, double T, int steps, int num_threads){

    // Create a 2D array of locks, one lock for each element in the array
    omp_lock_t **locks = (omp_lock_t **)malloc(L * sizeof(omp_lock_t *));
    for (int i = 0; i < L; i++) {
        locks[i] = (omp_lock_t *)malloc(L * sizeof(omp_lock_t));
        for (int j = 0; j < L; j++) {
            omp_init_lock(&locks[i][j]);  // Initialize each lock
        }
    }

  //subdivide into row-major strips for locality
  int blockdim_i, blockdim_j;
  //all columns in strip for locality 
  blockdim_j = L;
  //subdivide along rows
  blockdim_i = L/num_threads;
  
  #pragma omp parallel shared(locks, lattice) num_threads(num_threads)
  {
    int thread_id = omp_get_thread_num();
    
    //the lower boundary of i in the sublattice. highbound will then be blockdim + xbound
    //wrap around L
    int j_bound = 0;
    int i_bound = (thread_id * blockdim_i);
    
    //evenly divide work
    for (int i = 0; i < steps/num_threads; i++){
      unsigned int seed = thread_id;
     
      //pull random x and y from strip 
      seed += microtime();
      int x = (rand_r(&seed) % blockdim_i) + i_bound;
      seed += microtime();
      int y = (rand_r(&seed) % blockdim_j);
      //assess if pull is on vertical boundary of strip
      int boundary = (x - i_bound) % (blockdim_i - 1);

      //if(thread_id == 1){
        //printf("Debug thread: X: %d, Y: %d\n", x, y);
      //}

      //debug
      if(x < i_bound || x > (blockdim_i + i_bound)){
	    printf("Out of bounds err in thread %d\n", thread_id);
      }  

      if(boundary == 0){
        locking_metropolis(lattice,L,T,x,y,locks);
      }else{
        metropolis(lattice, L, T, x, y);
      }
      
      if (i % 10 == 0) {  // Synchronize every 10 iterations
        #pragma omp barrier
      }
    }
  }

  //Clean up locks and memory
  #pragma omp parallel for num_threads(num_threads)
  for (int i = 0; i < L; i++) {
    for (int j = 0; j < L; j++) {
      omp_destroy_lock(&locks[i][j]);
    }
    free(locks[i]);
  }
  free(locks);
}


