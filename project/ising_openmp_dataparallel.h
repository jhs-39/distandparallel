#ifndef ISING_OPENMP_DATAPARALLEL_H
#define ISING_OPENMP_DATAPARALLEL_H

void ising_openmp_dataparallel(int **lattice, int L, double T, int steps, int num_threads);
#endif
