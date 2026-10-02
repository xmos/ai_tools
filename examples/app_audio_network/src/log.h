#ifndef __log_h__
#define __log_h__

#include <stdint.h>

#define LOG2_8_Q_VALUE                    3
#define LOG2_16_Q_VALUE                  10

int log2_8(uint32_t x);
int log2_16(uint32_t x);

int log2_8_64(uint64_t x);
int log2_16_64(uint64_t x);

#endif
