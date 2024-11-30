#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "microtime.h"
#include "ising_model.h"

// Function to initialize the lattice with random spins
void initialize_lattice(int **lattice, int L) {
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            lattice[i][j] = (random_int(0, 1) == 0) ? 1 : -1;  // Random spin initialization
        }
    }
}

// Function to print the lattice configuration
void print_lattice(int **lattice, int L) {
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            if (lattice[i][j] == 1) {
                printf("+");
            } else {
                printf("-");
            }
        }
        printf("\n");
    }
}

// Function to generate a random integer between min and max (inclusive)
//NOT thread safe
int random_int(int min, int max) {
    return min + rand() % (max - min + 1);
}

// Function to generate a random double between 0.0 and 1.0
//Thread safe
double random_double() {
    unsigned int seed = 42 + omp_get_thread_num(); // Unique seed for each thread (can also use time or thread_id)
    return (double)rand_r(&seed) / RAND_MAX;
}

// Metropolis algorithm for the Ising model update
void metropolis(int **lattice, int L, double T, int x, int y) {
    //add neighbors together to create local 'field'
    int sum_neighbors = lattice[(x + 1) % L][y] + lattice[(x - 1 + L) % L][y] +
                        lattice[x][(y + 1) % L] + lattice[x][(y - 1 + L) % L];
    
    //if the current lattice site is flipped, what is the increase in system energy?
    //example: positive lattice site * positive neighbors -> increase in system energy if we flip to -
    //interaction strength J is assumed to be 1
    int deltaE = 2 * lattice[x][y] * sum_neighbors;

    //accept flip with probability scaled by T, E
    double threshold = random_double();

    //partition has two terms in the sum
    double partition = exp(-deltaE/T) + exp(deltaE/T);
    double probabilityOfFlip = exp(-deltaE/T)/partition;
    if (threshold < probabilityOfFlip) {  // T is temperature parameter. Boltzmann constant assumed to be 1
        lattice[x][y] = -lattice[x][y]; // Flip the spin
    }
}

//serial update; control condition for comparison
void serial_metropolis(int **lattice, int L, double T, int steps){
  
  for(int i = 0; i < steps; i++){
    int x = random_int(0,(L-1));
    int y = random_int(0,(L-1));
    metropolis(lattice,L,T,x,y);
  }
}

//we expect this to cause false sharing and cache misses as threads may hit same row
void naive_metropolis(int **lattice, int L, double T, int steps, int num_threads){
  int x,y;
  unsigned int seed;
  #pragma omp parallel for private(x,y,seed) shared(lattice) schedule(static) num_threads(num_threads)
  for(int i = 0; i < steps; i++){
    seed = i + omp_get_thread_num();
    x = rand_r(&seed) % L;
    y = rand_r(&seed) % L;
    metropolis(lattice,L,T,x,y);
  }
}

//acquire locks on lattice site and all neighbors
//return 1 if successful
int getLocks(omp_lock_t **locks, int L, int x, int y){
   //integer indicating if all locks are acquired
   int lock_acquired = 1;

    // Try to acquire all locks
    int locks_acquired[5] = {0};  // Array to track the lock status
    locks_acquired[0] = omp_test_lock(&locks[x][y]);  // lock at (x, y)
    locks_acquired[1] = omp_test_lock(&locks[(x - 1 + L) % L][y]);  // lock at (x - 1, y)
    locks_acquired[2] = omp_test_lock(&locks[(x + 1 + L) % L][y]);  // lock at (x + 1, y)
    locks_acquired[3] = omp_test_lock(&locks[x][(y - 1 + L) % L]);  // lock at (x, y - 1)
    locks_acquired[4] = omp_test_lock(&locks[x][(y + 1 + L) % L]);  // lock at (x, y + 1)

    // Check if all locks were acquired
    for (int i = 0; i < 5; i++) {
        if (!locks_acquired[i]) {
            lock_acquired = 0;  // One of the locks wasn't acquired
            break;
        }
    }

    // If all locks were acquired, return 1
    if (lock_acquired) {
        return 1;  // All locks successfully acquired
    }

    // If not all locks were acquired, release any acquired locks and return 0
    if (!lock_acquired){
      if (locks_acquired[0]) omp_unset_lock(&locks[x][y]);
      if (locks_acquired[1]) omp_unset_lock(&locks[(x - 1 + L) % L][y]);
      if (locks_acquired[2]) omp_unset_lock(&locks[(x + 1 + L) % L][y]);
      if (locks_acquired[3]) omp_unset_lock(&locks[x][(y - 1 + L) % L]);
      if (locks_acquired[4]) omp_unset_lock(&locks[x][(y + 1 + L) % L]);
    }
    
    return 0;  // Not all locks were acquired
}

//Metropolis that locks neighbors for race condition; used for multithreading
//includes time release locks
int locking_metropolis(int **lattice, int L, double T, int x, int y, omp_lock_t **locks){
    int success = 0;    

    //lock lattice site and neighbors. lattice edges are treated 'wrap around' to opposite edge (top to bottom, left to right)
    double start_time = omp_get_wtime();  // Record start time
    double timeout = .000001; //1 us timeout time to release lock and try somewhere else
    int locked = 0;

    // Try acquiring the lock until timeout
    while (!locked) {
      if (omp_get_wtime() - start_time > timeout) {
	break;  // Timeout reached, exit loop and continue to next iteration
      }

      // Attempt to acquire the locks
      if (getLocks(locks,L,x,y)) {
        locked = 1;  // Lock acquired successfully
      }
    }
    
    //if locks successful, do metropolis and unlock sites
    if(locked){
      metropolis(lattice, L, T, x, y);

      //unlock lattice site
      omp_unset_lock(&locks[x][y]);

      //unlock neighbor locks
      omp_unset_lock(&locks[(x - 1 + L)%L][y]);
      omp_unset_lock(&locks[(x + 1 + L)%L][y]);
      omp_unset_lock(&locks[x][(y - 1 + L)%L]);
      omp_unset_lock(&locks[x][(y + 1 + L)%L]);
      success = 1;
    }
    return success;
}

//for data parallelism, only lock on boundaries of sublattice to save time
int boundary_metropolis(int **lattice, int L, double T, int i_bound, int i_block_size, int j_bound, int j_block_size, unsigned int seed, omp_lock_t **locks){
  //int i = random_int(i_bound, i_bound + i_block_size - 1);
  //int j = random_int(j_bound, j_bound + j_block_size - 1);
  //seed unique based on region -- unique to each thread
  int i = (rand_r(&seed) % (i_block_size)) + i_bound;
  seed += microtime();
  int j = (rand_r(&seed) % j_block_size) + j_bound;

  int iBoundTest = (i - i_bound) % (i_block_size-1);

  //debug
  if(i < i_bound){
	  printf("Error: i out of bounds\n");
  }else if(i > (i_bound + i_block_size)){
    printf("Error: i above bounds\n");
  }
  
  
  //if on boundary, lock all neighbors and self due to risk of collision
  if(iBoundTest == 0){
    int success = 0;
    do{
      success = locking_metropolis(lattice, L, T, i, j, locks);
    }while(success == 0);
  }else{
    metropolis(lattice, L, T, i, j);
    //debug
    //if(omp_get_thread_num() == 3){
      //printf("Thread 3 updated successfully\n");
    //}
  }

  return 1;
}

//look in the working array for adjacency. Since we are in strips for data parallel, only consider vertical adjacency 
int collisionTest(int **workingSites, int L, int i, int j){
  int collision;
  {
  if((workingSites[(i+1+L)%L][j] == 1) || (workingSites[(i-1+L)%L][j] == 1)){
    collision = 0;
  }else{
    collision = 1;
  }
  }
  return collision;
}

//ignore locks altogether -- lets use a signaling array to avoid collisions and not wait for locks
//data parallel approach without locks
//assumption; lattice has been split into row major strips using i_bound and j_bound
int signal_metropolis(int **lattice, int L, double T, int i_bound, int i_block_size, int j_bound, int j_block_size, unsigned int seed, int **workingSites){
  //printf("Debug: Signal Metropolis\n\n");
  seed += microtime();
  int i = (rand_r(&seed) % i_block_size) + i_bound;
  seed += microtime();
  int j = (rand_r(&seed) % j_block_size);
  //printf("I: %d J: %d\n\n",i,j);
  int iBoundTest = (i - i_bound) % (i_block_size - 1);

  int locked = 0;

  //if on vertical boundary of strip, make sure no other threads are looking at same data
  if(iBoundTest == 0){
//critically, check if any threads are working vertically adjacent
#pragma omp critical
{
    //look in the working array for adjacency. Since we are in strips for data parallel, only consider vertical adjacency
    if(collisionTest(workingSites, L, i, j) == 1){
      workingSites[i][j] = 1;
      locked = 1;
    }
}  

  //now 'locked' we can complete metropolis step without fear of interference
  if(locked){
    metropolis(lattice,L,T,i,j);
  }
  
  //release work site after completion of metropolis step
#pragma omp critical
{
  workingSites[i][j] = 0;
}

  }else{
    //if not on boundary, not in danger of collision -- freely apply
    metropolis(lattice,L,T,i,j);
  }

  return 1;
}

void ising_openmp_signalparallel(int **lattice, int L, double T, int steps, int num_threads){
  int** workingSites = (int **)malloc(L * sizeof(int *));
  for (int i = 0; i < L; i++) {
    workingSites[i] = (int *)malloc(L * sizeof(int));
  }
  for(int i = 0; i < L; i++){
    for(int j = 0; j < L; j++){
      workingSites[i][j] = 0;
    }
  }

  //subdivide into row-major strips for locality
  int blockdim_i, blockdim_j;
  //all columns in strip for locality
  blockdim_j = L;
  //subdivide along rows; each thread gets equal work
  blockdim_i = L/num_threads;
  //printf("Debug: beginning of parallel block\n\n");

  #pragma omp parallel shared(lattice) num_threads(num_threads)
  {
    int thread_id = omp_get_thread_num();
    
    //starting index for columns; is all cols
    int j_bound = 0;
    //starting index for rows for this thread
    int i_bound = (thread_id * blockdim_i);
    //printf("Thread %d: i_bound = %d, blockdim_i = %d\n", thread_id, i_bound, blockdim_i);
    //printf("Thread %d: processing %d steps\n", thread_id, steps/num_threads);
    for (int i = 0; i < steps/num_threads; i++){
      unsigned int seed = i + thread_id + microtime();
      signal_metropolis(lattice,L,T,i_bound,blockdim_i,j_bound,blockdim_j,seed,workingSites);
    }
  }
  //printf("Debug: end of parallel block\n");

  for (int i = 0; i < L; i++) {
    free(workingSites[i]);
  }
  free(workingSites);

}

