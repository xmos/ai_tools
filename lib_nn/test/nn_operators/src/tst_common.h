

#ifndef TST_COMMON_H_
#define TST_COMMON_H_

#include <stdint.h>

#define TEST_C_GLOBAL (1)
#define DO_PRINT_EXTRA_GLOBAL (1)


int16_t  pseudo_rand_int16(unsigned *r);
uint16_t pseudo_rand_uint16(unsigned *r);
int32_t  pseudo_rand_int32(unsigned *r);
uint32_t pseudo_rand_uint32(unsigned *r);
int64_t  pseudo_rand_int64(unsigned *r);
uint64_t pseudo_rand_uint64(unsigned *r);


void pseudo_rand_bytes(unsigned *r, char* buffer, unsigned size);



#endif //TST_COMMON_H_