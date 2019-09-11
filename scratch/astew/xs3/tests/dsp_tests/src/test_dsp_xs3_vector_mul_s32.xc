
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>

#include "dsp_xs3_vector.h"
#include "Unity.h"
#include "xs3_vpu.h"
#include "tst_common.h"

#ifdef __XC__
#define WORD_ALIGNED [[aligned(4)]]
#else
#define WORD_ALIGNED
#endif

unsigned seed = 5523;

static inline unsigned cls(const int32_t num)
{  
    unsigned res;
    asm("cls %0, %1" : "=r"(res) : "r"(num) );
    return res-1;
}


#define DEBUG_ON (0)
static void _test_dsp_xs3_vector_mul_s32_case1()
{

    typedef struct {
        int32_t b;
        int b_exp;
        int32_t c;
        int c_exp;
        int32_t a;
        int a_exp;
    } test_case;


    test_case cases[] = {
        {   0x20000000, -29,     0x20000000, -29,          1<<28, -28,              },
        {       1,   0,             1,   0,                1<<28, -28,              },
        {       1,   1,             1,   0,                2<<27, -27,              },
        {      -1,   0,             1,   0,               -1<<29, -29,              },
        {      -1,   0,            -1,   0,                1<<30, -30,              },
        {       4,   0,             3,   0,               12<<25, -25,              },
        {   1<<30, -30,         1<<30, -30,                1<<28, -28,              },
        {  -1<<30, -30,        -1<<30, -30,                1<<30, -30,              },
        {   1<<30, -30,        -1<<30, -30,               -1<<29, -29,              },
    };

    const unsigned N = sizeof(cases)/sizeof(test_case);

    for(int u = 0; u < N; u++){
#if DEBUG_ON
        printf("Starting case %u\n", u);
#endif

        int32_t WORD_ALIGNED A_c[1]   = { 654321 };
        int32_t WORD_ALIGNED A_asm[1] = { 123456 };
        int32_t WORD_ALIGNED B[1] = { 0 };
        int32_t WORD_ALIGNED C[1] = { 0 };

        B[0] = cases[u].b;
        C[0] = cases[u].c;

        int B_exp = cases[u].b_exp;
        int C_exp = cases[u].c_exp;

        unsigned B_hr = cls(B[0]);
        unsigned C_hr = cls(C[0]);

        int A_exp, B_shr, C_shr;
        unsigned A_hr;

        dsp_xs3_vector_mul_s32_prepare(&A_exp, &B_shr, &C_shr, B_exp, B_hr, C_exp, C_hr);

#if DEBUG_ON
        printf("B_exp = %d\n", B_exp);
        printf("C_exp = %d\n", C_exp);
        printf("B_hr  = %u\n", B_hr );
        printf("C_hr  = %u\n", C_hr );
        printf("B_shr  = %d\n", B_shr );
        printf("C_shr  = %d\n", C_shr );
#endif

        TEST_ASSERT_EQUAL_MESSAGE(cases[u].a_exp, A_exp, "A_exp was calculated incorrectly.");

        dsp_xs3_vector_mul_s32_c(A_c, &A_hr, B, C, B_shr, C_shr, 1);
        dsp_xs3_vector_mul_s32_asm(A_asm, &A_hr, B, C, B_shr, C_shr, 1);

        TEST_ASSERT_EQUAL_MESSAGE(cases[u].a, A_c[0], "A[0] was calculated incorrectly. (C)");
        TEST_ASSERT_EQUAL_MESSAGE(cases[u].a, A_asm[0], "A[0] was calculated incorrectly. (ASM)");

        TEST_ASSERT_LESS_OR_EQUAL_MESSAGE(2, A_hr, "A had too much headroom.");
        
    }
}
#undef DEBUG_ON



//Random values. Confirm C and ASM match. Length=1
#define DEBUG_ON (0)
#define N_CASES  (1024)
static void _test_dsp_xs3_vector_mul_s32_case2()
{


    for(int u = 0; u < N_CASES; u++){
#if DEBUG_ON
        printf("Starting case %u\n", u);
#endif

        int32_t WORD_ALIGNED A_c[1] = { 654321 };
        int32_t WORD_ALIGNED A_asm[1] = { 123456 };
        int32_t WORD_ALIGNED B[1] = { 0 };
        int32_t WORD_ALIGNED C[1] = { 0 };

        const int exp_diff = (pseudo_rand_uint32(&seed) % 20) - 10;
        const int B_exp = (pseudo_rand_uint32(&seed) % 30) - 20;
        const int C_exp = B_exp + exp_diff;

        B[0] = pseudo_rand_int32(&seed);
        C[0] = pseudo_rand_int32(&seed);

        B[0] >>= pseudo_rand_uint32(&seed) % 30;
        C[0] >>= pseudo_rand_uint32(&seed) % 30;

        unsigned B_hr = cls(B[0]);
        unsigned C_hr = cls(C[0]);

        int A_exp, B_shr, C_shr;
        unsigned A_hr;

        dsp_xs3_vector_mul_s32_prepare(&A_exp, &B_shr, &C_shr, B_exp, B_hr, C_exp, C_hr);

#if DEBUG_ON
        printf("B = %d * 2^(%d)\t\t(B_hr: %u)\n", B[0], B_exp, B_hr);
        printf("C = %d * 2^(%d)\t\t(C_hr: %u)\n", C[0], C_exp, C_hr);

        printf("A_exp = %d\n", A_exp);
        printf("B_shr = %d\n", B_shr);
        printf("C_shr = %d\n", C_shr);
        printf("\n");
#endif

        dsp_xs3_vector_mul_s32_c  (  A_c, &A_hr, B, C, B_shr, C_shr, 1);
        dsp_xs3_vector_mul_s32_asm(A_asm, &A_hr, B, C, B_shr, C_shr, 1);

        TEST_ASSERT_EQUAL_MESSAGE(A_c[0], A_asm[0], "ASM result did not match C.");

        if(B[0]*C[0] != 0)
            TEST_ASSERT_LESS_OR_EQUAL_MESSAGE(2, A_hr, "A had too much headroom.");
        else
            TEST_ASSERT_EQUAL_MESSAGE(31, A_hr, "A_hr should be 31 when product is zero.");
    }
}
#undef N_CASES
#undef DEBUG_ON




//Random vectors. Confirm C and ASM match. Length=7
#define DEBUG_ON (0)
#define N_CASES  (16)
#define VLEN     (7)
static void _test_dsp_xs3_vector_mul_s32_case3()
{


    for(int u = 0; u < N_CASES; u++){
#if DEBUG_ON
        printf("Starting case %u\n", u);
#endif

        int32_t WORD_ALIGNED A_c[VLEN] = { 0 };
        int32_t WORD_ALIGNED A_asm[VLEN] = { 0 };
        int32_t WORD_ALIGNED B[VLEN] = { 0 };
        int32_t WORD_ALIGNED C[VLEN] = { 0 };

        memset(A_c, 0xFF, sizeof(A_c));
        memset(A_asm, 0xFF, sizeof(A_asm));

        const int exp_diff = (pseudo_rand_uint32(&seed) % 20) - 10;
        const int B_exp = (pseudo_rand_uint32(&seed) % 30) - 20;
        const int C_exp = B_exp + exp_diff;

        for(int i = 0; i < VLEN; i++){
            B[i] = pseudo_rand_int32(&seed);
            B[i] >>= pseudo_rand_uint32(&seed) % 30;

            C[i] = pseudo_rand_int32(&seed);
            C[i] >>= pseudo_rand_uint32(&seed) % 30;
        }

        
        unsigned B_hr = dsp_xs3_calc_headroom(B, VLEN);
        unsigned C_hr = dsp_xs3_calc_headroom(C, VLEN);

        int A_exp, B_shr, C_shr;

        dsp_xs3_vector_mul_s32_prepare(&A_exp, &B_shr, &C_shr, B_exp, B_hr, C_exp, C_hr);

#if DEBUG_ON
        printf("B = [");
        for(int i = 0; i < VLEN; i++){
            printf("%d, ", B[i]);
        }
        printf("] * 2^(%d)\t\t(B_hr: %u)\n", B_exp, B_hr);
        
        printf("C = [");
        for(int i = 0; i < VLEN; i++){
            printf("%d, ", C[i]);
        }
        printf("] * 2^(%d)\t\t(C_hr: %u)\n", C_exp, C_hr);

        printf("A_exp = %d\n", A_exp);
        printf("B_shr = %d\n", B_shr);
        printf("C_shr = %d\n", C_shr);
        printf("\n");
#endif

        unsigned A_hr_c, A_hr_asm;
        dsp_xs3_vector_mul_s32_c  (  A_c, &A_hr_c, B, C, B_shr, C_shr, VLEN);
        dsp_xs3_vector_mul_s32_asm(A_asm, &A_hr_asm, B, C, B_shr, C_shr, VLEN);

        
#if DEBUG_ON
        printf("A_c = [");
        for(int i = 0; i < VLEN; i++){
            printf("%d, ", A_c[i]);
        }
        printf("] * 2^(%d)\t\t(A_hr_c: %u)\n", A_exp, A_hr_c);
        
        printf("A_asm = [");
        for(int i = 0; i < VLEN; i++){
            printf("%d, ", A_asm[i]);
        }
        printf("] * 2^(%d)\t\t(A_hr_c: %u)\n", A_exp, A_hr_asm);
        printf("\n");
#endif

        for(int i = 0; i < VLEN; i++){
            TEST_ASSERT_EQUAL_MESSAGE(A_c[i], A_asm[i], "ASM result did not match C.");
        }

        TEST_ASSERT_EQUAL_MESSAGE(dsp_xs3_calc_headroom(A_c, VLEN), A_hr_c, "A_hr not correct. (C)");
        TEST_ASSERT_EQUAL_MESSAGE(dsp_xs3_calc_headroom(A_asm, VLEN), A_hr_asm, "A_hr not correct. (ASM)");
    }
}
#undef VLEN
#undef N_CASES
#undef DEBUG_ON




//Random vectors. Confirm C and ASM match. Length is randomish
#define DEBUG_ON    (0)
#define N_CASES     (64)
#define VLEN_MAX    (256)
static void _test_dsp_xs3_vector_mul_s32_case4()
{


    for(int u = 0; u < N_CASES; u++){
#if DEBUG_ON
        printf("Starting case %u\n", u);
#endif

        int32_t WORD_ALIGNED A_c[VLEN_MAX]      = { 0 };
        int32_t WORD_ALIGNED A_asm[VLEN_MAX]    = { 0 };
        int32_t WORD_ALIGNED B[VLEN_MAX]        = { 0 };
        int32_t WORD_ALIGNED C[VLEN_MAX]        = { 0 };

        memset(A_c, 0xFF, sizeof(A_c));
        memset(A_asm, 0xFF, sizeof(A_asm));

        unsigned length = pseudo_rand_uint16(&seed) % VLEN_MAX;
        
        const int exp_diff = (pseudo_rand_uint32(&seed) % 20) - 10;
        const int B_exp = (pseudo_rand_uint32(&seed) % 30) - 20;
        const int C_exp = B_exp + exp_diff;


        for(int i = 0; i < length; i++){
            B[i] = pseudo_rand_int32(&seed);
            B[i] >>= pseudo_rand_uint32(&seed) % 30;

            C[i] = pseudo_rand_int32(&seed);
            C[i] >>= pseudo_rand_uint32(&seed) % 30;
        }
        
        unsigned B_hr = dsp_xs3_calc_headroom(B, length);
        unsigned C_hr = dsp_xs3_calc_headroom(C, length);

        int A_exp, B_shr, C_shr;

        dsp_xs3_vector_mul_s32_prepare(&A_exp, &B_shr, &C_shr, B_exp, B_hr, C_exp, C_hr);

#if DEBUG_ON
        printf("B = [");
        for(int i = 0; i < length; i++){
            printf("%d, ", B[i]);
        }
        printf("] * 2^(%d)\t\t(B_hr: %u)\n", B_exp, B_hr);
        
        printf("C = [");
        for(int i = 0; i < length; i++){
            printf("%d, ", C[i]);
        }
        printf("] * 2^(%d)\t\t(C_hr: %u)\n", C_exp, C_hr);

        printf("A_exp = %d\n", A_exp);
        printf("B_shr = %d\n", B_shr);
        printf("C_shr = %d\n", C_shr);
        printf("\n");
#endif

        unsigned A_hr_c, A_hr_asm;
        dsp_xs3_vector_mul_s32_c  (  A_c, &A_hr_c,   B, C, B_shr, C_shr, length);
        dsp_xs3_vector_mul_s32_asm(A_asm, &A_hr_asm, B, C, B_shr, C_shr, length);

        
#if DEBUG_ON
        printf("A_c = [");
        for(int i = 0; i < length; i++){
            printf("%d, ", A_c[i]);
        }
        printf("] * 2^(%d)\t\t(A_hr_c: %u)\n", A_exp, A_hr_c);
        
        printf("A_asm = [");
        for(int i = 0; i < length; i++){
            printf("%d, ", A_asm[i]);
        }
        printf("] * 2^(%d)\t\t(A_hr_c: %u)\n", A_exp, A_hr_asm);
        printf("\n");
#endif

        for(int i = 0; i < length; i++){
            TEST_ASSERT_EQUAL_MESSAGE(A_c[i], A_asm[i], "ASM result did not match C.");
        }

        TEST_ASSERT_EQUAL_MESSAGE(dsp_xs3_calc_headroom(A_c,length), A_hr_c, "A_hr not correct. (C)");
        TEST_ASSERT_EQUAL_MESSAGE(dsp_xs3_calc_headroom(A_asm, length), A_hr_asm, "A_hr not correct. (ASM)");
    }
}
#undef VLEN_MAX
#undef N_CASES
#undef DEBUG_ON

void test_dsp_xs3_vector_mul_s32()
{
    // _test_dsp_xs3_vector_mul_s32_case1();
    // _test_dsp_xs3_vector_mul_s32_case2();
    // _test_dsp_xs3_vector_mul_s32_case3();
    _test_dsp_xs3_vector_mul_s32_case4();
}