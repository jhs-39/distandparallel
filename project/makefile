CC=gcc
CFLAGS= -g -Wall -fopenmp -fno-unroll-loops -I. -O0 -march=native -lm

TARGETS=ising_experiments # add your target here

all: $(TARGETS)

ising_experiments: ising_model.o microtime.o ising_experiments.o ising_openmp_taskparallel.o ising_openmp_dataparallel.o
	$(CC) $(CFLAGS) -o ising_experiments ising_model.o microtime.o ising_experiments.o ising_openmp_taskparallel.o ising_openmp_dataparallel.o $(LDFLAGS)

ising_openmp_taskparallel.o: ising_openmp_taskparallel.c ising_openmp_taskparallel.h
	$(CC) $(CFLAGS) -c $<

ising_openmp_dataparallel.o: ising_openmp_dataparallel.c ising_openmp_dataparallel.h
	$(CC) $(CFLAGS) -c $<

ising_model.o: ising_model.c ising_model.h
	$(CC) $(CFLAGS) -c $<

microtime.o: microtime.c microtime.h
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f *.o *~ core $(TARGETS)
