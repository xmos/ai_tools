

#ifndef TST_COMMON_H_
#define TST_COMMON_H_

#include <stdint.h>

#define TEST_C_GLOBAL         (0)
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


int8_t  pseudo_rand_int8();
int16_t  pseudo_rand_int16();
uint16_t pseudo_rand_uint16();
int32_t  pseudo_rand_int32();
uint32_t pseudo_rand_uint32();
int64_t  pseudo_rand_int64();
uint64_t pseudo_rand_uint64();


void pseudo_rand_bytes(char* buffer, unsigned size);

void print_warns(
    int start_case);

#ifdef __XC__
} // extern "C"
#endif

#endif //TST_COMMON_H_