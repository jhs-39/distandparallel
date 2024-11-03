#ifndef MICROTIME_H
#define MICROTIME_H

#ifdef __cplusplus
extern "C" {
#endif

double microtime(void);                /* Time in micro-seconds */
double getMicrotimeResolution(void); /* Timer resolution in micro-seconds */

#ifdef __cplusplus
}
#endif

#endif // MICROTIME_H
