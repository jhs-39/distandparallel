#include <omp.h>
#include <stdlib.h>
#include "ising_model.h"

void ising_openmp_taskparallel(int **lattice, int L, double T, int steps, int num_threads){
  
	//parallelize the for-loop task with 5 mutex locks acquired each round; the lattice site and 4 neighbors. If threads collide, mutual lock
	//will occur. No native openmp support for time release locks so improvising one. Implementing count on task completion

    // Create a 2D array of locks, one lock for each element in the array
    omp_lock_t **locks = (omp_lock_t **)malloc(L * sizeof(omp_lock_t *));
    for (int i = 0; i < L; i++) {
        locks[i] = (omp_lock_t *)malloc(L * sizeof(omp_lock_t));
        for (int j = 0; j < L; j++) {
            omp_init_lock(&locks[i][j]);  // Initialize each lock
        }
    }

  //run metropolis for specified number of steps
  #pragma omp parallel for num_threads(num_threads)
  for(int i = 0; i < steps; i++){
    int x = random_int(0,(L-1));
    int y = random_int(0,(L-1));
    locking_metropolis(lattice,L,T,x,y,locks);
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


