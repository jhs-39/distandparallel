CC=gcc
CFLAGS= -g -Wall -fopenmp -I. -O3 -march=native

TARGETS=optimized # add your target here

all: $(TARGETS)

optimized: optimized.o microtime.o
	$(CC) -fopenmp -o $@ $^

optimized.o: optimized.c microtime.h
	$(CC) $(CFLAGS) -c $<

microtime.o: microtime.c microtime.h
	$(CC) $(CFLAGS) -c $<

# Add your rules here.
# Example: code in opt1.c

# opt1: opt1.o microtime.o
# 	$(CC) -o $@ $^

# opt1.o: opt1.c microtime.h
# 	$(CC) $(CFLAG) -c $<


clean:
	rm -f *.o *~ core $(TARGETS)
