#include "microtime.h"
#include <sys/time.h>

double getMicrotimeResolution(void) {
  double time1, time2;

  time1 = microtime();
  do {
    time2 = microtime();
  } while (time1 == time2);

  return time2 - time1;
}

double microtime(void) {
  struct timeval t;

  gettimeofday(&t, 0);

  return 1.0e6 * t.tv_sec + (double)t.tv_usec;
}
