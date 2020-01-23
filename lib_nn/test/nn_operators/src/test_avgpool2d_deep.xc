
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <syscall.h>

#include "tst_common.h"

#include "nn_operator.h"
#include "xs3_vpu.h"

#include "Unity.h"

#ifdef __XC__
#define WORD_ALIGNED [[aligned(4)]]
#else
#define WORD_ALIGNED
#endif

#if (defined(__XS3A__) && USE_ASM_avgpool2d_deep)
 #define HAS_ASM (1)
#else
 #define HAS_ASM (0)
#endif

#define TEST_ASM ((HAS_ASM) && 1)
#define TEST_C ((TEST_C_GLOBAL) && 1)

#define DO_PRINT_EXTRA ((DO_PRINT_EXTRA_GLOBAL) && 1)

#define PRINTF(...)     do{if (DO_PRINT_EXTRA) {printf(__VA_ARGS__);}} while(0)



static unsigned seed = 4321234;















#define DEBUG_ON    (0 || TEST_DEBUG_ON)
#define C_in        (VPU_INT8_EPV)
#define C_out       (C_in)
#define height      (8)
#define width       (8)
void test_avgpool2d_deep_case1()
{
    int8_t WORD_ALIGNED  X[height][width][C_in];

    int8_t WORD_ALIGNED Y_expected[height/2][width/2][C_out] = {{{ 0 }}};

    PRINTF("test_avgpool2d_deep_case1()...\n");

    memset(X, 0x00, sizeof(X));

#if TEST_C
    PRINTF("\tC...\n");
    int8_t WORD_ALIGNED  Y_c[height/2][width/2][C_out];
    memset(Y_c, 0xCC, sizeof(Y_c));
    avgpool2d_deep_c((int8_t*) X, (int8_t*) Y_c, height, width, C_in);
#endif
#if TEST_ASM
    PRINTF("\tASM...\n");
    int8_t WORD_ALIGNED  Y_asm[height/2][width/2][C_out];
    memset(Y_asm, 0xCC, sizeof(Y_asm));
    avgpool2d_deep_asm((int8_t*) X, (int8_t*) Y_asm, height, width, C_in);
#endif

    PRINTF("\tChecking...\n");
    for(unsigned h = 0; h < height/2; h++){
        for(unsigned w = 0; w < width/2; w++){
            for(unsigned c = 0; c < C_out; c++){
                char str_buff[100];
                //Annoying, but only doing sprintf if necessary saves a ton of time in xsim
                int do_sprintf = 0;
#if TEST_C
                do_sprintf = do_sprintf || (Y_expected[h][w][c] != Y_c[h][w][c]);
#endif
#if TEST_ASM
                do_sprintf = do_sprintf || (Y_expected[h][w][c] != Y_asm[h][w][c]);
#endif
                if(do_sprintf)  sprintf(str_buff, "(h,w,c) = (%u,%u,%u)", h,w,c);
#if TEST_C
                TEST_ASSERT_EQUAL_MESSAGE(Y_expected[h][w][c], Y_c[h][w][c], str_buff);
#endif
#if TEST_ASM
                TEST_ASSERT_EQUAL_MESSAGE(Y_expected[h][w][c], Y_asm[h][w][c], str_buff);
#endif
            }
        }
    }
}
#undef height
#undef width
#undef C_out
#undef C_in
#undef DEBUG_ON









#define DEBUG_ON    (1 || TEST_DEBUG_ON)
#define C_in        (2*VPU_INT8_EPV)
#define C_out       (C_in)
#define height      (10)
#define width       (8)
#define REPS        (50)
#define CHECKS      (500)
void test_avgpool2d_deep_case2()
{
    int8_t WORD_ALIGNED X[height][width][C_in];

    PRINTF("test_avgpool2d_deep_case2()...\n");

    for(int rep = 0; rep < REPS; rep++){

        PRINTF("\ttest vector %d...\n", rep);

        //Randomize input image content
        pseudo_rand_bytes(&seed, (char*)X, sizeof(X));

        //Process it
#if TEST_C
        PRINTF("\t\tC...\n");
        int8_t WORD_ALIGNED Y_c[height/2][width/2][C_out];
        avgpool2d_deep_c((int8_t*) X, (int8_t*) Y_c, height, width, C_in);
#endif
#if TEST_ASM
        PRINTF("\t\tASM...\n");
        int8_t WORD_ALIGNED Y_asm[height/2][width/2][C_out];\
        avgpool2d_deep_asm((int8_t*) X, (int8_t*) Y_asm, height, width, C_in);
#endif

        PRINTF("\t\tChecking...\n");
        //Spot-check the result
        for(unsigned check = 0; check < CHECKS; check++){

            //Pick an output pixel and channel
            unsigned row = pseudo_rand_uint16(&seed) % (height/2);
            unsigned col = pseudo_rand_uint16(&seed) % (width/2);
            unsigned ch  = pseudo_rand_uint16(&seed) % C_in;
            
            if(check == 0)
                row = col = ch = 0;


            unsigned in_row = 2*row;
            unsigned in_col = 2*col;

            int32_t acc = 0;
            acc += X[in_row + 0][in_col + 0][ch];
            acc += X[in_row + 0][in_col + 1][ch];
            acc += X[in_row + 1][in_col + 0][ch];
            acc += X[in_row + 1][in_col + 1][ch];

            acc = (acc + 0x02) >> 2;
            if (acc == INT8_MIN)
                acc = VPU_INT8_MIN;

            int8_t exp_out     = acc;

#if (DEBUG_ON)
            unsigned dbg = 0;
  #if (TEST_C)
            dbg = dbg || exp_out != Y_c[row][col][ch];
  #endif
  #if (TEST_ASM)
            dbg = dbg || exp_out != Y_asm[row][col][ch];
  #endif
            if(dbg) {
                    PRINTF("rep = %d; check = %d\n", rep, check);
                    PRINTF("row, col, ch = (%d, %d, %d)\n", row, col, ch);
                    PRINTF("in vals = (%d, %d, %d, %d)\n",
                        X[in_row + 0][in_col + 0][ch], X[in_row + 0][in_col + 1][ch], 
                        X[in_row + 1][in_col + 0][ch], X[in_row + 1][in_col + 1][ch]);
                    PRINTF("exp_out: %d\n", exp_out);
  #if (TEST_C)
                        PRINTF("actual out (C): %d\n", Y_c[row][col][ch]);
  #endif
  #if (TEST_ASM)
                        PRINTF("actual out (ASM): %d\n", Y_asm[row][col][ch]);
  #endif
              }
#endif

#if TEST_C
            TEST_ASSERT_EQUAL(exp_out, Y_c[row][col][ch]);
#endif
#if TEST_ASM
            TEST_ASSERT_EQUAL(exp_out, Y_asm[row][col][ch]);
#endif

        }

#if DEBUG_ON
        printf("\n");
#endif

    }

}
#undef CHECKS
#undef REPS
#undef width
#undef height
#undef C_out
#undef C_in
#undef DEBUG_ON





void test_avgpool2d_deep()
{
    test_avgpool2d_deep_case1();
    test_avgpool2d_deep_case2();
}
