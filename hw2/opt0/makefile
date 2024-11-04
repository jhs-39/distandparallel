CC=gcc
CFLAGS= -g -Wall -fopenmp -I. -O3 -march=native

TARGETS=opt0 # add your target here

all: $(TARGETS)

opt0: opt0.o microtime.o
	$(CC) -fopenmp -o $@ $^

opt0.o: opt0.c microtime.h
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
