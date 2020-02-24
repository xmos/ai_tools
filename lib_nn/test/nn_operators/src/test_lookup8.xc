
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <syscall.h>

#include "tst_common.h"

#include "nn_operator.h"
#include "xs3_vpu.h"

// #include "dsp_xs3_vector.h"
#include "Unity.h"

#ifdef __XC__
#define WORD_ALIGNED [[aligned(4)]]
#else
#define WORD_ALIGNED
#endif

#if (defined(__XS3A__) && USE_ASM_avgpool2d_global)
 #define HAS_ASM (1)
#else
 #define HAS_ASM (1)
#endif

#define TEST_ASM ((HAS_ASM)     && 1)
#define TEST_C ((TEST_C_GLOBAL) && 1)

#define DO_PRINT_EXTRA ((DO_PRINT_EXTRA_GLOBAL) && 1)

#define PRINTF(...)     do{if (DO_PRINT_EXTRA) {printf(__VA_ARGS__);}} while(0)

static unsigned seed = 4412311;








#define DEBUG_ON        (0 || TEST_DEBUG_ON)
#define MAX_LEN         128
#define REPS            50
void test_lookup8_case0()
{

    PRINTF("test_lookup8_case0()...\n");

    uint8_t WORD_ALIGNED src[MAX_LEN] = { 0 };

    uint8_t WORD_ALIGNED lut[256];

#if TEST_C
    uint8_t WORD_ALIGNED dst_c[MAX_LEN] = { 0 };
#endif

#if TEST_ASM
    uint8_t WORD_ALIGNED dst_asm[MAX_LEN] = { 0 };
#endif

    pseudo_rand_bytes(&seed, src, MAX_LEN);
    pseudo_rand_bytes(&seed, lut, 256);


    for(int k = 0; k < REPS; k++){

        PRINTF("\trep %d...\n", k); 

        uint32_t size = pseudo_rand_uint32(&seed) % MAX_LEN;

#if TEST_C
        memset(dst_c, 0xCC, sizeof(dst_c));
        PRINTF("\t\tC...\n");
        lookup8_c((uint8_t*)dst_c, (uint8_t*) src, (uint8_t*) lut, size);
        for(int i = 0; i < MAX_LEN; i++)
            TEST_ASSERT_EQUAL_MESSAGE( (i < size)? lut[src[i]] : 0xCC, dst_c[i], "C failed.");
#endif

#if TEST_ASM
        memset(dst_asm, 0xCC, sizeof(dst_asm));
        PRINTF("\t\tASM...\n");
        lookup8_asm((uint8_t*)dst_asm, (uint8_t*) src, (uint8_t*) lut, size);

        for(int i = 0; i < MAX_LEN; i++){
            TEST_ASSERT_EQUAL_MESSAGE( (i < size)? lut[src[i]] : 0xCC, dst_asm[i], "ASM failed.");
        }
#endif

        

    }

}
#undef REPS
#undef MAX_LEN
#undef DEBUG_ON



