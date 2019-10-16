
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <syscall.h>

#include "tst_common.h"

#include "nn_operator.h"
#include "nn_types.h"
#include "xs3_vpu.h"

// #include "dsp_xs3_vector.h"
#include "Unity.h"

#ifdef __XC__
#define WORD_ALIGNED [[aligned(4)]]
#else
#define WORD_ALIGNED
#endif

#if (defined(__XS3A__) && USE_ASM_conv2d_deepin_deepout_relu)
 #define HAS_ASM (1)
#else
 #define HAS_ASM (0)
#endif

#define TEST_C (1)

#define DO_PRINT_EXTRA (1)

#define PRINTF(...)     do{if (DO_PRINT_EXTRA) {printf(__VA_ARGS__);}} while(0)


#define ADD_VAL (-((1<<14)-1))










#define DEBUG_ON    (0 || TEST_DEBUG_ON)
#define C_out       (VPU_INT8_ACC_PERIOD)
#define C_in        (VPU_INT8_EPV)
#define K_h         (1)
#define K_w         (1)
#define height      (8)
#define width       (8)
void test_conv2d_deepin_deepout_relu_case1()
{

    int8_t   WORD_ALIGNED  K[C_out][K_h][K_w][C_in]      = {{{{ 0 }}}};
    data16_t  WORD_ALIGNED  B[2][C_out]                  = {{ 0 }};
    int8_t   WORD_ALIGNED  X[height][width][C_in];
    int16_t  WORD_ALIGNED  shifts[C_out]                 = { 0 };
    int16_t  WORD_ALIGNED  scales[C_out]                 = { 0 };

    int8_t WORD_ALIGNED Y_expected[height][width][C_out] = {{{ 0 }}};
    int8_t   WORD_ALIGNED  Y_c[height][width][C_out];

    PRINTF("test_conv2d_deepin_deepout_relu_case1()...\n");

#if TEST_C
    PRINTF("\tC...\n");
    memset(Y_c, 0xCC, sizeof(Y_c));
    conv2d_deepin_deepout_relu_c((int8_t*) K, (data16_t*) B, (int8_t*) X, (int8_t*) Y_c, 
                                height, width, K_h, K_w, C_out, C_in, shifts, scales);
#endif
#if HAS_ASM
    PRINTF("\tASM...\n");
    int8_t   WORD_ALIGNED  Y_asm[height][width][C_out];
    memset(Y_asm, 0xCC, sizeof(Y_asm));
    conv2d_deepin_deepout_relu_asm((int8_t*) K, (data16_t*) B, (int8_t*) X, (int8_t*) Y_asm, 
                                height, width, K_h, K_w, C_out, C_in, shifts, scales);
#endif

    PRINTF("\tChecking...\n");

    for(unsigned h = 0; h < height; h++){
        for(unsigned w = 0; w < width; w++){
            for(unsigned c = 0; c < C_out; c++){
                    char str_buff[100];
                    //Annoying, but only doing sprintf if necessary saves a ton of time in xsim
                    int do_sprintf = 0;
#if TEST_C
                    do_sprintf = do_sprintf || (Y_expected[h][w][c] != Y_c[h][w][c]);
#endif
#if HAS_ASM
                    do_sprintf = do_sprintf || (Y_expected[h][w][c] != Y_asm[h][w][c]);
#endif
                    if(do_sprintf)  sprintf(str_buff, "(h,w,c) = (%u,%u,%u)", h,w,c);
#if TEST_C
                    TEST_ASSERT_EQUAL_MESSAGE(Y_expected[h][w][c], Y_c[h][w][c], str_buff);
#endif
#if HAS_ASM
                    TEST_ASSERT_EQUAL_MESSAGE(Y_expected[h][w][c], Y_asm[h][w][c], str_buff);
#endif
            }
        }
    }
}
#undef width
#undef height
#undef K_w
#undef K_h
#undef C_in
#undef C_out
#undef DEBUG_ON





















#define DEBUG_ON        (0 || TEST_DEBUG_ON)
#define C_out           (VPU_INT8_ACC_PERIOD)
#define C_in            (VPU_INT8_EPV)
#define K_h             (1)
#define K_w             (1)
#define height          (4)
#define width           (4)
#define VECTOR_FMT      ("test_data/conv2d_deepin_deepout_relu_case2.%u.dat")
#include "../test_data/conv2d_deepin_deepout_relu_case2.h"
void test_conv2d_deepin_deepout_relu_case2()
{

    int8_t   WORD_ALIGNED  K[C_out][K_h][K_w][C_in];
    data16_t WORD_ALIGNED  B[2][C_out];
    int8_t   WORD_ALIGNED  X[height][width][C_in];
    int16_t  WORD_ALIGNED  shifts[C_out];
    int16_t  WORD_ALIGNED  scales[C_out];
    int8_t   WORD_ALIGNED  Y_expected[height][width][C_out]; 
    
    int16_t Y_check[] = { Y_CHECK };

    PRINTF("test_conv2d_deepin_deepout_relu_case2()...\n");

    for(int v = 0; v < TEST_VECTOR_COUNT; v++){
        PRINTF("\ttest vector %d...\n", v);
    
        int8_t   WORD_ALIGNED  Y_c[height][width][C_out];


        char filename[100];
        sprintf(filename, VECTOR_FMT, v);

        int input_file = _open(filename, O_RDONLY|O_BINARY, S_IREAD);
        assert(input_file != -1);
        _read(input_file, (char*) K,          sizeof(K));
        _read(input_file, (char*) B,          sizeof(B));
        _read(input_file, (char*) X,          sizeof(X));
        _read(input_file, (char*) shifts,     sizeof(shifts));
        _read(input_file, (char*) scales,     sizeof(scales));
        _read(input_file, (char*) Y_expected, sizeof(Y_expected));
        _close(input_file);

        assert(Y_check[v] == Y_expected[0][0][0]);
#if TEST_C
        PRINTF("\t\tC...\n");
        memset(Y_c, 0xCC, sizeof(Y_c));
        conv2d_deepin_deepout_relu_c((int8_t*) K, (data16_t*) B, (int8_t*) X, (int8_t*) Y_c, 
                                    height, width, K_h, K_w, C_out, C_in, shifts, scales);
#endif
#if HAS_ASM
        PRINTF("\t\tASM...\n");
        int8_t   WORD_ALIGNED  Y_asm[height][width][C_out];
        memset(Y_asm, 0xCC, sizeof(Y_asm));
        conv2d_deepin_deepout_relu_asm((int8_t*) K, (data16_t*) B, (int8_t*) X, (int8_t*) Y_asm, 
                                    height, width, K_h, K_w, C_out, C_in, shifts, scales);
#endif

        PRINTF("\t\tChecking...\n");

        for(unsigned h = 0; h < height; h++){
            for(unsigned w = 0; w < width; w++){
                for(unsigned c = 0; c < C_out; c++){
                    char str_buff[100];
                    //Annoying, but only doing sprintf if necessary saves a ton of time in xsim
                    int do_sprintf = 0;
#if TEST_C
                    do_sprintf = do_sprintf || (Y_expected[h][w][c] != Y_c[h][w][c]);
#endif
#if HAS_ASM
                    do_sprintf = do_sprintf || (Y_expected[h][w][c] != Y_asm[h][w][c]);
#endif
                    if(do_sprintf)  sprintf(str_buff, "(v,h,w,c) = (%u,%u,%u,%u)", v,h,w,c);
#if TEST_C
                    TEST_ASSERT_EQUAL_MESSAGE(Y_expected[h][w][c], Y_c[h][w][c], str_buff);
#endif
#if HAS_ASM
                    TEST_ASSERT_EQUAL_MESSAGE(Y_expected[h][w][c], Y_asm[h][w][c], str_buff);
#endif
                }
            }
        }
    }

}
#undef VECTOR_FMT
#undef TEST_VECTORS
#undef width
#undef height
#undef K_w
#undef K_h
#undef C_in
#undef C_out
#undef DEBUG_ON
























#define DEBUG_ON        (0 || TEST_DEBUG_ON)
#define C_out           (VPU_INT8_ACC_PERIOD)
#define C_in            (VPU_INT8_EPV)
#define K_h             (3)
#define K_w             (3)
#define height          (4)
#define width           (4)
#define VECTOR_FMT      ("test_data/conv2d_deepin_deepout_relu_case3.%u.dat")
#include "../test_data/conv2d_deepin_deepout_relu_case3.h"
void test_conv2d_deepin_deepout_relu_case3()
{

    int8_t   WORD_ALIGNED  K[C_out][K_h][K_w][C_in];
    data16_t WORD_ALIGNED  B[2][C_out];
    int8_t   WORD_ALIGNED  X[height][width][C_in];
    int16_t  WORD_ALIGNED  shifts[C_out];
    int16_t  WORD_ALIGNED  scales[C_out];
    int8_t   WORD_ALIGNED  Y_expected[height][width][C_out]; 
    
    int16_t Y_check[] = { Y_CHECK };

    PRINTF("test_conv2d_deepin_deepout_relu_case3()...\n");

    for(int v = 0; v < TEST_VECTOR_COUNT; v++){
        PRINTF("\ttest vector %d...\n", v);
    
        int8_t   WORD_ALIGNED  Y_c[height][width][C_out];


        char filename[100];
        sprintf(filename, VECTOR_FMT, v);

        int input_file = _open(filename, O_RDONLY|O_BINARY, S_IREAD);
        assert(input_file != -1);
        _read(input_file, (char*) K, sizeof(K));
        _read(input_file, (char*) B, sizeof(B));
        _read(input_file, (char*) X, sizeof(X));
        _read(input_file, (char*) shifts, sizeof(shifts));
        _read(input_file, (char*) scales, sizeof(scales));
        _read(input_file, (char*) Y_expected, sizeof(Y_expected));
        _close(input_file);

        assert(Y_check[v] == Y_expected[0][0][0]);
#if TEST_C
        PRINTF("\t\tC...\n");
        memset(Y_c, 0xCC, sizeof(Y_c));
        conv2d_deepin_deepout_relu_c((int8_t*) K, (data16_t*) B, (int8_t*) X, (int8_t*) Y_c, 
                                    height, width, K_h, K_w, C_out, C_in, shifts, scales);
#endif
#if HAS_ASM
        PRINTF("\t\tASM...\n");
        int8_t   WORD_ALIGNED  Y_asm[height][width][C_out];
        memset(Y_asm, 0xCC, sizeof(Y_asm));
        conv2d_deepin_deepout_relu_asm((int8_t*) K, (data16_t*) B, (int8_t*) X, (int8_t*) Y_asm, 
                                    height, width, K_h, K_w, C_out, C_in, shifts, scales);
#endif

        PRINTF("\t\tChecking...\n");

        for(unsigned c = 0; c < C_out; c++){
            for(unsigned h = 0; h < height; h++){
                for(unsigned w = 0; w < width; w++){
                    char str_buff[100];
                    //Annoying, but only doing sprintf if necessary saves a ton of time in xsim
                    int do_sprintf = 0;
#if TEST_C
                    do_sprintf = do_sprintf || (Y_expected[h][w][c] != Y_c[h][w][c]);
#endif
#if HAS_ASM
                    do_sprintf = do_sprintf || (Y_expected[h][w][c] != Y_asm[h][w][c]);
#endif
                    if(do_sprintf)  sprintf(str_buff, "(v,h,w,c) = (%u,%u,%u,%u)", v,h,w,c);
#if TEST_C
                    TEST_ASSERT_EQUAL_MESSAGE(Y_expected[h][w][c], Y_c[h][w][c], str_buff);
#endif
#if HAS_ASM
                    TEST_ASSERT_EQUAL_MESSAGE(Y_expected[h][w][c], Y_asm[h][w][c], str_buff);
#endif
                }
            }
        }
    }

}
#undef VECTOR_FMT
#undef TEST_VECTORS
#undef width
#undef height
#undef K_w
#undef K_h
#undef C_in
#undef C_out
#undef DEBUG_ON
























#define DEBUG_ON        (0 || TEST_DEBUG_ON)
#define C_out           (2*VPU_INT8_ACC_PERIOD)
#define C_in            (2*VPU_INT8_EPV)
#define K_h             (3)
#define K_w             (3)
#define height          (8)
#define width           (8)
#define VECTOR_FMT      ("test_data/conv2d_deepin_deepout_relu_case4.%u.dat")
#include "../test_data/conv2d_deepin_deepout_relu_case4.h"
void test_conv2d_deepin_deepout_relu_case4()
{

    int8_t   WORD_ALIGNED  K[C_out][K_h][K_w][C_in];
    data16_t WORD_ALIGNED  B[2][C_out];
    int8_t   WORD_ALIGNED  X[height][width][C_in];
    int16_t  WORD_ALIGNED  shifts[C_out];
    int16_t  WORD_ALIGNED  scales[C_out];
    int8_t   WORD_ALIGNED  Y_expected[height][width][C_out]; 
    
    int16_t Y_check[] = { Y_CHECK };

    PRINTF("test_conv2d_deepin_deepout_relu_case4()...\n");

    for(int v = 0; v < TEST_VECTOR_COUNT; v++){
        PRINTF("\ttest vector %d...\n", v);
    
        int8_t   WORD_ALIGNED  Y_c[height][width][C_out];

        memset(Y_c, 0xCC, sizeof(Y_c));

        char filename[100];
        sprintf(filename, VECTOR_FMT, v);

        int input_file = _open(filename, O_RDONLY|O_BINARY, S_IREAD);
        assert(input_file != -1);
        _read(input_file, (char*) K, sizeof(K));
        _read(input_file, (char*) B, sizeof(B));
        _read(input_file, (char*) X, sizeof(X));
        _read(input_file, (char*) shifts, sizeof(shifts));
        _read(input_file, (char*) scales, sizeof(scales));
        _read(input_file, (char*) Y_expected, sizeof(Y_expected));
        _close(input_file);

        assert(Y_check[v] == Y_expected[0][0][0]);

        // PRINTF("\t!!0x%x\t0x%x\n", K, &K[C_out-1][K_h-1][K_w-1][C_in-1]);

#if TEST_C
        PRINTF("\t\tC...\n");
        conv2d_deepin_deepout_relu_c((int8_t*) K, (data16_t*) B, (int8_t*) X, (int8_t*) Y_c, 
                                    height, width, K_h, K_w, C_out, C_in, shifts, scales);
#endif
#if HAS_ASM
        PRINTF("\t\tASM...\n");
        int8_t   WORD_ALIGNED  Y_asm[height][width][C_out];
        memset(Y_asm, 0xCC, sizeof(Y_asm));
        conv2d_deepin_deepout_relu_asm((int8_t*) K, (data16_t*) B, (int8_t*) X, (int8_t*) Y_asm, 
                                    height, width, K_h, K_w, C_out, C_in, shifts, scales);
#endif

        PRINTF("\t\tChecking...\n");

        for(unsigned c = 0; c < C_out; c++){
            for(unsigned h = 0; h < height; h++){
                for(unsigned w = 0; w < width; w++){
                    char str_buff[100];
                    //Annoying, but only doing sprintf if necessary saves a ton of time in xsim
                    int do_sprintf = 0;
#if TEST_C
                    do_sprintf = do_sprintf || (Y_expected[h][w][c] != Y_c[h][w][c]);
#endif
#if HAS_ASM
                    do_sprintf = do_sprintf || (Y_expected[h][w][c] != Y_asm[h][w][c]);
#endif
                    if(do_sprintf)  sprintf(str_buff, "(v,h,w,c) = (%u,%u,%u,%u)", v,h,w,c);
#if TEST_C
                    TEST_ASSERT_EQUAL_MESSAGE(Y_expected[h][w][c], Y_c[h][w][c], str_buff);
#endif
#if HAS_ASM
                    TEST_ASSERT_EQUAL_MESSAGE(Y_expected[h][w][c], Y_asm[h][w][c], str_buff);
#endif
                }
            }
        }
        
    }

}
#undef VECTOR_FMT
#undef TEST_VECTORS
#undef width
#undef height
#undef K_w
#undef K_h
#undef C_in
#undef C_out
#undef DEBUG_ON


void test_conv2d_deepin_deepout_relu()
{
    test_conv2d_deepin_deepout_relu_case1();
    test_conv2d_deepin_deepout_relu_case2();
    test_conv2d_deepin_deepout_relu_case3();
    test_conv2d_deepin_deepout_relu_case4();
}

