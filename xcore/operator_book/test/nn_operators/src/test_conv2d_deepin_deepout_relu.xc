
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


#define ADD_VAL (-((1<<14)-1))

// static unsigned seed = 44334;
unsafe{
void conv2d_deepin_deepout_relu_asm(
    const int8_t* K, 
    const data16_t* B,
    const int8_t* X, 
    int8_t* Y,
    const int32_t height, 
    const int32_t width,
    const int32_t K_h, 
    const int32_t K_w,
    const int32_t C_out, 
    const int32_t C_in,
    const int16_t* shifts, 
    const int16_t* scales)
{
    const int16_t add_vector[16] = {ADD_VAL,ADD_VAL,ADD_VAL,ADD_VAL,ADD_VAL,ADD_VAL,ADD_VAL,ADD_VAL,ADD_VAL,ADD_VAL,ADD_VAL,ADD_VAL,ADD_VAL,ADD_VAL,ADD_VAL,ADD_VAL,};
    const data16_t* B_lo = B;
    const data16_t* B_hi = &B[C_out];

    const int K_h_half = K_h >> 1;
    const int K_w_half = K_w >> 1;

    // const unsigned kernel_chunk_size = 16 * C_in * K_w * K_h;

    int8_t* unsafe y = (int8_t* unsafe) Y;
    // printf("0x%08x\n", K);
    // printf("0x%08x\n", &K[15*C_in*K_h*K_w]);

    // for(int i = 0; i < 16; i++){
    //     printf("K start for channel %d: 0x%08x\n", 15-i, &K[i*C_in*K_h*K_w + C_in*K_w*1 + C_in*1]);
    // }

#if 0
    printf("Running\n");

    int pxl_r = 0;
    int pxl_c = 0;

    printf("X = np.array([\n");
    for(int tmp_r = -K_h_half; tmp_r <= K_h_half; tmp_r++){
        printf("[");
        for(int tmp_c = -K_w_half; tmp_c <= K_w_half; tmp_c++){
            printf("[");
            for(int c_in = 0; c_in < C_in; c_in++){
                int yy = pxl_r + tmp_r;
                int xx = pxl_c + tmp_c;
                int offset = yy*(width*C_in) + xx*C_in + c_in;
                
                if(xx < 0 || yy < 0)  printf("0,");
                else                  printf("%d,", X[offset]);
            }
            printf("],\n");
        }
        printf("],");
    } 
    printf("],dtype=np.int8)\n\n");

    

    printf("K = np.array([\n");
    for(int tmp_r = 0; tmp_r < K_h; tmp_r++){
        printf("[");
        for(int tmp_c = 0; tmp_c < K_w; tmp_c++){
            printf("[");
            for(int c_in = 0; c_in < C_in; c_in++){

                int offset = (15*K_h*K_w*C_in) + tmp_r*(K_w*C_in) + tmp_c*C_in + c_in;
                printf("%d,", K[offset]);
                // printf("0x%08x\n", &K[offset]);

            }
            printf("],\n");
        }
        printf("],");
    } 
    printf("],dtype=np.int8)\n\n");


    printf("B[0] = %ld\n", (((int32_t)B_hi[0]) << 16) | ((uint32_t)B_lo[0]));

    printf("shifts[0] = %d\n", shifts[0]);
    printf("scales[0] = %d\n", scales[0]);
#endif

    for(int row = 0; row < height; row++){
        for(int col = 0; col < width; col++){

            int patch_start_row = row - K_h_half;
            int patch_end_row   = row + K_h_half;
            int patch_start_col = col - K_w_half;
            int patch_end_col   = col + K_w_half;

            unsigned pad_l = 0;
            unsigned pad_t = 0;
            unsigned pad_r = 0;
            unsigned pad_b = 0;

            if(patch_start_row < 0){
                pad_t = -patch_start_row;
                patch_start_row = 0;
            }
            if(patch_start_col < 0){
                pad_l = -patch_start_col;
                patch_start_col = 0;
            }
            if(patch_end_row >= height){
                pad_b = patch_end_row - (height-1);
                patch_end_row = height-1;
            }
            if(patch_end_col >= width){
                pad_r = patch_end_col - (width-1);
                patch_end_col = width-1;
            }

            const int8_t* patch_x = X + C_in * (patch_start_row * width + patch_start_col);

            const unsigned K_offset = C_in * (pad_t * K_w + pad_l);
            const int8_t* patch_k = K + K_offset;
            
            const unsigned patch_cols = patch_end_col - patch_start_col + 1;
            const unsigned patch_rows = patch_end_row - patch_start_row + 1;

            const unsigned patch_row_incr = C_in * (width - patch_cols);
            const unsigned kernel_row_incr = C_in * (K_w - patch_cols);

            const unsigned patch_row_maccs = (C_in >> 5) * patch_cols;

            //Needs to be the number of bytes to get to the start of the next kernel chunk plus the offset from the start
            //  of a kernel chunk to the start of that patch_k
            const unsigned kernel_advance = C_in * (pad_b * K_w + pad_r);

            // printf("height\t\t= %u\n", height);
            // printf("width\t\t= %u\n", width);
            // printf("K_h\t\t= %u\n", K_h);
            // printf("K_w\t\t= %u\n", K_w);
            // printf("row\t\t= %u\n", row);
            // printf("col\t\t= %u\n", col);
            // printf("y\t\t= 0x%08X\t(0x%08X)\n", y, Y);
            // printf("patch_k\t\t= 0x%08X\t(0x%08X)\n", patch_k, K);
            // printf("B_lo\t\t= 0x%08X\t(0x%08X)\n", B_lo, B);
            // printf("B_hi\t\t= 0x%08X\t(0x%08X)\n", B_hi, B);
            // printf("patch_row_incr\t= %u\n", patch_row_incr);
            // printf("kernel_row_incr\t= %u\n", kernel_row_incr);
            // printf("patch_x\t\t= 0x%08X\t(0x%08X)\n", patch_x, X);
            // printf("patch_rows\t= %u\n", patch_rows);
            // printf("patch_row_maccs\t= %u\n", patch_row_maccs);
            // printf("(C_out>>4)\t= %u\n", (C_out>>4));
            // printf("kernel_advance\t= %u\n", kernel_advance);            
            // printf("shifts\t\t= 0x%08X\n", shifts);
            // printf("scales\t\t= 0x%08X\n", scales);

            // printf("\n\n");


            y = (int8_t*unsafe) conv2d_deepin_deepout_relu_asm_patch((int8_t*)y, patch_k, B_lo, B_hi, patch_row_incr, kernel_row_incr, patch_x, 
                                patch_rows, patch_row_maccs, (C_out>>4), kernel_advance, shifts, scales, add_vector);

            // return;
        }
    }
}
}


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

