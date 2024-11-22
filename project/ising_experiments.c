//This file runs a 2d Ising Model in parallel and serial implementations
//3 tasks include
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "microtime.h"
#include "ising_model.h"
#include "ising_openmp_taskparallel.h"
#include "ising_openmp_dataparallel.h"

int main() {
    srand(time(NULL));
    
    //size of square 2d lattice
    int L = 128;
    //printf("Please enter the size L of a desired LxL Lattice: \n");
    //scanf("%d", &L);
    //temperature parameter
    double T = 2;
    //printf("Please enter temperature parameter. Theoretic critical temp of 2d Ising model is ~2.2\n");
    //scanf("%lf", &T);
    //how many MonteCarlo steps are used in a discrete step
    int STEPS = 8000000;
    //printf("Enter number of steps to use in Monte Carlo simulation of lattice: \n");
    //scanf("%d", &STEPS);
    
    //number threads to test
    int num_threads[] = {1, 2, 4, 8};
    //measured runtimes
    double time = 0;

    //variables for storing run time
    double start, end;

    //allocate lattice memory. random init to -1 or +1
    int** lattice = (int **)malloc(L * sizeof(int *));
    for (int i = 0; i < L; i++) {
      lattice[i] = (int *)malloc(L * sizeof(int));
    }
    
    initialize_lattice(lattice,L);

    //print initial configuration
    //printf("Initial Lattice:\n");
    //print_lattice(lattice,L);

    //simulate the Ising model using Metropolis algorithm
    //serial metropolis

    start = microtime();
    serial_metropolis(lattice, L, T, STEPS);
    end = microtime();
    
    printf("Serial time: %f\n", end - start);
    print_lattice(lattice,L);
    
    //parallelism without locks
    //we expect this to perform poorly due to false sharing and cache locality and race condition of no locks. when threads access same line it will false share
    printf("Naive Parallelism\n");
    
    for(int i = 0; i < 4; i++){
      start = microtime();
      naive_metropolis(lattice,L,T,STEPS,num_threads[i]);
      end = microtime();
      time = end - start;
      printf("Thread count: %d\n",num_threads[i]);
      printf("Run Time: %f\n", time);
      //print_lattice(lattice,L);
      initialize_lattice(lattice,L);
    }

    //openmp multithread; task parallelism with time release of lock
    //this may also perform poorly due to overhead of all the locks
    printf("Task Parallelism Test\n");
    for(int i = 0; i < 4; i++){
      start = microtime();
      ising_openmp_taskparallel(lattice, L, T, STEPS, num_threads[i]);
      end = microtime();
      time = end - start;

      printf("Thread count: %d\n", num_threads[i]);
      printf("Run Time: %f\n", time);
      //print_lattice(lattice,L);
      initialize_lattice(lattice,L);
    }
    
    printf("Data Parallelism Test\n");

    //openmp mulithread; data parallelism to minimize collision and locking of threads
    for(int i = 0; i < 4; i++){
      start = microtime();
      ising_openmp_dataparallel(lattice,L,T,STEPS,num_threads[i]);
      end = microtime();
      time = end - start;
      printf("Thread count: %d\n", num_threads[i]);
      printf("Run Time: %f\n", time);
      //print_lattice(lattice,L);
      initialize_lattice(lattice,L);
    }
    
    printf("Begin Final\n");
    //openmp multithread; dataparallelism without locks; maintains an array to make sure boundaries arent't colliding
    for(int i = 0; i < 4; i++){
      start = microtime();
      ising_openmp_signalparallel(lattice,L,T,STEPS,num_threads[i]);
      end = microtime();
      time = end - start;
      printf("Thread count: %d\n",num_threads[i]);
      printf("Run Time: %f\n",time);
      //print_lattice(lattice,L);
      initialize_lattice(lattice,L);
    }
    
    //print results of run to .csv
    
    
    return 0;
}
