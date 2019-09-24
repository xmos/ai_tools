
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

#if (defined(__XS3A__) && USE_ASM_conv2d_deepin_deepout_relu)
 #define HAS_ASM (1)
#else
 #define HAS_ASM (0)
#endif

// static unsigned seed = 44334;


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

    memset(Y_c, 0xCC, sizeof(Y_c));

    conv2d_deepin_deepout_relu_c((int8_t*) K, (data16_t*) B, (int8_t*) X, (int8_t*) Y_c, 
                                height, width, K_h, K_w, C_out, C_in, shifts, scales);
#if HAS_ASM
    int8_t   WORD_ALIGNED  Y_asm[height][width][C_out];
    memset(Y_asm, 0xCC, sizeof(Y_asm));
    conv2d_deepin_deepout_relu_asm((int8_t*) K, (data16_t*) B, (int8_t*) X, (int8_t*) Y_asm, 
                                height, width, K_h, K_w, C_out, C_in, shifts, scales);
#endif

    for(unsigned h = 0; h < height; h++){
        for(unsigned w = 0; w < width; w++){
            for(unsigned c = 0; c < C_out; c++){
                char str_buff[100];
                sprintf(str_buff, "(h,w,c) = (%u,%u,%u)", h,w,c);
                TEST_ASSERT_EQUAL_MESSAGE(Y_expected[h][w][c], Y_c[h][w][c], str_buff);
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
#define TEST_VECTORS    (10)
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

    for(int v = 0; v < TEST_VECTORS; v++){
        printf("test vector %d...\n", v);
    
        int8_t   WORD_ALIGNED  Y_c[height][width][C_out];

        memset(Y_c, 0xCC, sizeof(Y_c));

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

#if DEBUG_ON
        printf("B:\t");
        for(int x = 0; x < C_out; x++){
            printf("%d, ", B[x]);
        }
        printf("\n\n");

        printf("X:");
        for(int x = 0; x < height; x++){
            printf("\t");
            for(int y = 0; y < width; y++){
                printf("%d, ", X[x][y][0]);
            }
            printf("\n");
        }
        printf("\n");

        // printf("%d\t%d\t%d\n", Y_check[v], Y_expected[0][0][0], v);
        // break;
#endif
        assert(Y_check[v] == Y_expected[0][0][0]);

        conv2d_deepin_deepout_relu_c((int8_t*) K, (data16_t*) B, (int8_t*) X, (int8_t*) Y_c, 
                                    height, width, K_h, K_w, C_out, C_in, shifts, scales);
#if HAS_ASM
        int8_t   WORD_ALIGNED  Y_asm[height][width][C_out];
        memset(Y_asm, 0xCC, sizeof(Y_asm));
        conv2d_deepin_deepout_relu_asm((int8_t*) K, (data16_t*) B, (int8_t*) X, (int8_t*) Y_asm, 
                                    height, width, K_h, K_w, C_out, C_in, shifts, scales);
#endif


        for(unsigned h = 0; h < height; h++){
            for(unsigned w = 0; w < width; w++){
                for(unsigned c = 0; c < C_out; c++){
                    char str_buff[100];
                    sprintf(str_buff, "(v,h,w,c) = (%u,%u,%u,%u)", v,h,w,c);
                    TEST_ASSERT_EQUAL_MESSAGE(Y_expected[h][w][c], Y_c[h][w][c], str_buff);
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
#define TEST_VECTORS    (10)
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

    for(int v = 0; v < TEST_VECTORS; v++){
        printf("test vector %d...\n", v);
    
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

#if DEBUG_ON
        printf("B:\t");
        for(int x = 0; x < C_out; x++){
            printf("%d, ", B[x]);
        }
        printf("\n\n");

        printf("X:");
        for(int x = 0; x < height; x++){
            printf("\t");
            for(int y = 0; y < width; y++){
                printf("%d, ", X[x][y][0]);
            }
            printf("\n");
        }
        printf("\n");

        // printf("%d\t%d\t%d\n", Y_check[v], Y_expected[0][0][0], v);
        // break;
#endif

        assert(Y_check[v] == Y_expected[0][0][0]);

        conv2d_deepin_deepout_relu_c((int8_t*) K, (data16_t*) B, (int8_t*) X, (int8_t*) Y_c, 
                                    height, width, K_h, K_w, C_out, C_in, shifts, scales);
#if HAS_ASM
        int8_t   WORD_ALIGNED  Y_asm[height][width][C_out];
        memset(Y_asm, 0xCC, sizeof(Y_asm));
        conv2d_deepin_deepout_relu_asm((int8_t*) K, (data16_t*) B, (int8_t*) X, (int8_t*) Y_asm, 
                                    height, width, K_h, K_w, C_out, C_in, shifts, scales);
#endif


        for(unsigned c = 0; c < C_out; c++){
            for(unsigned h = 0; h < height; h++){
                for(unsigned w = 0; w < width; w++){
                    char str_buff[100];
                    sprintf(str_buff, "(v,h,w,c) = (%u,%u,%u,%u)", v,h,w,c);
                    TEST_ASSERT_EQUAL_MESSAGE(Y_expected[h][w][c], Y_c[h][w][c], str_buff);
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
#define TEST_VECTORS    (4)
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

    for(int v = 0; v < TEST_VECTORS; v++){
        printf("test vector %d...\n", v);
    
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

#if DEBUG_ON
        printf("B:\t");
        for(int x = 0; x < C_out; x++){
            printf("%d, ", B[x]);
        }
        printf("\n\n");

        printf("X:");
        for(int x = 0; x < height; x++){
            printf("\t");
            for(int y = 0; y < width; y++){
                printf("%d, ", X[x][y][0]);
            }
            printf("\n");
        }
        printf("\n");

        // printf("%d\t%d\t%d\n", Y_check[v], Y_expected[0][0][0], v);
        // break;
#endif

        assert(Y_check[v] == Y_expected[0][0][0]);

        conv2d_deepin_deepout_relu_c((int8_t*) K, (data16_t*) B, (int8_t*) X, (int8_t*) Y_c, 
                                    height, width, K_h, K_w, C_out, C_in, shifts, scales);
#if HAS_ASM
        int8_t   WORD_ALIGNED  Y_asm[height][width][C_out];
        memset(Y_asm, 0xCC, sizeof(Y_asm));
        conv2d_deepin_deepout_relu_asm((int8_t*) K, (data16_t*) B, (int8_t*) X, (int8_t*) Y_asm, 
                                    height, width, K_h, K_w, C_out, C_in, shifts, scales);
#endif


        for(unsigned c = 0; c < C_out; c++){
            for(unsigned h = 0; h < height; h++){
                for(unsigned w = 0; w < width; w++){
                    char str_buff[100];
                    sprintf(str_buff, "(v,h,w,c) = (%u,%u,%u,%u)", v,h,w,c);
                    TEST_ASSERT_EQUAL_MESSAGE(Y_expected[h][w][c], Y_c[h][w][c], str_buff);
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

