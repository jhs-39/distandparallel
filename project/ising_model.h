#ifndef ISING_MODEL_H
#define ISING_MODEL_H

#include <omp.h>

void initialize_lattice(int **lattice, int L);
void print_lattice(int **lattice, int L);
int random_int(int min, int max);
double random_double();
void metropolis(int **lattice, int L, double T, int x, int y);
void serial_metropolis(int **lattice, int L, double T, int steps);
void naive_metropolis(int **lattice, int L, double T, int steps, int num_threads);
void locking_metropolis(int **lattice, int L, double T, int x, int y, omp_lock_t **locks);
int boundary_metropolis(int **lattice, int L, double T, int i_bound, int i_blocksize, int j_bound, int j_blocksize, unsigned int seed, omp_lock_t **locks);
int signal_metropolis(int **lattice, int L, double T, int i_bound, int i_blocksize, int j_bound, int j_blocksize, unsigned int seed, int **workSites);
void ising_openmp_signalparallel(int **lattice, int L, double T, int steps, int num_threads);

#endif
