

#ifndef TST_COMMON_H_
#define TST_COMMON_H_

#include <stdint.h>

#define TEST_C_GLOBAL         (1)
#define DO_PRINT_EXTRA_GLOBAL (1)

#define UNITY_SET_FILE()        Unity.TestFile = __FILE__

#ifdef __xcore__
#define WORD_ALIGNED __attribute__((aligned (4)))
#else
#define WORD_ALIGNED
#endif

#ifndef QUICK_TEST
#define QUICK_TEST  0
#endif

#define USE_ASM(FUNC)   (defined(__XS3A__) && USE_ASM_ ## FUNC)

#define IF_QUICK_TEST(X, Y)  ((QUICK_TEST)? X : Y)

#define PRINTF(...)     do{if (DO_PRINT_EXTRA) {printf(__VA_ARGS__);}} while(0)

#ifdef __XC__
extern "C" {
#endif

void test_crc32(unsigned *r);

int16_t  pseudo_rand_int16(unsigned *r);
uint16_t pseudo_rand_uint16(unsigned *r);
int32_t  pseudo_rand_int32(unsigned *r);
uint32_t pseudo_rand_uint32(unsigned *r);
int64_t  pseudo_rand_int64(unsigned *r);
uint64_t pseudo_rand_uint64(unsigned *r);


void pseudo_rand_bytes(unsigned *r, char* buffer, unsigned size);

void print_warns(
    int start_case, 
    unsigned test_c, 
    unsigned test_asm);

#ifdef __XC__
} // extern "C"
#endif

#endif //TST_COMMON_H_