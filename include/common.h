#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <string.h>
#include <cmath>

//#include <sys/time.h>

#ifndef VALUE_TYPE
#define VALUE_TYPE double
#endif

#ifndef BENCH_REPEAT
#define BENCH_REPEAT 1
#endif

#ifndef WARP_SIZE
#define WARP_SIZE   64
#endif

//#ifndef WARP_PER_BLOCK
//#define WARP_PER_BLOCK   5
//#endif

#define SUBSTITUTION_FORWARD  0
#define SUBSTITUTION_BACKWARD 1

#define OPT_WARP_NNZ   1
#define OPT_WARP_RHS   2
#define OPT_WARP_AUTO  3

#define MAX_SOURCE_SIZE (0x100000)
