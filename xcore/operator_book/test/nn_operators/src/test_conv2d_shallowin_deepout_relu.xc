
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

#if (defined(__XS3A__) && USE_ASM_conv2d_shallowin_deepout_relu)
 #define HAS_ASM (1)
#else
 #define HAS_ASM (0)
#endif

#define TEST_C (1)

#define DO_PRINT_EXTRA (1)

#define PRINTF(...)     do{if (DO_PRINT_EXTRA) {printf(__VA_ARGS__);}} while(0)

// static unsigned seed = 4412311;


unsafe {
void conv2d_shallowin_deepout_relu_asm2(
    const int8_t* K, 
    const data16_t* B,
    const int8_t* X, 
    int8_t* Y,
    const int32_t height, 
    const int32_t width,
    const int32_t K_h, 
    const int32_t K_w,
    const int32_t C_out,
    const int16_t* shifts, 
    const int16_t* scales)
{
    const unsigned C_in = 4;
    const unsigned K_w_dim = 8;

    const unsigned cout_groups = C_out >> VPU_INT8_ACC_PERIOD_LOG2;
    
    const data16_t* B_lo = &B[0];
    const data16_t* B_hi = &B[C_out];

    const int32_t K_h_half = K_h >> 1;
    const int32_t K_w_half = K_w >> 1;

    int8_t* unsafe y_ptr = (int8_t* unsafe) Y;

    const unsigned X_pxl_bytes = C_in;
    const unsigned X_row_bytes = X_pxl_bytes * width;

    const unsigned K_row_bytes = VPU_INT8_ACC_PERIOD * K_w_dim * C_in;

    for(int row = 0; row < height; row++){
        unsigned pad_t = 0;
        unsigned pad_b = 0;

        int patch_start_row = row - K_h_half;
        int patch_end_row = row + K_h_half;

        if(patch_start_row < 0){
            pad_t = -patch_start_row;
            patch_start_row = 0;
        }

        if((height-1) < patch_end_row){
            pad_b = patch_end_row - (height-1);
            patch_end_row = (height-1);
        }

        const unsigned patch_rows = patch_end_row - patch_start_row + 1;
        
        const unsigned x_row_offset = patch_start_row * X_row_bytes;

        const unsigned k_offset = pad_t * K_row_bytes;
        const int8_t* K_start = K + k_offset;
        const unsigned kernel_advance = (pad_t+pad_b)*K_row_bytes;



        for(int col = 0; col < width; col++){

            unsigned pad_l = 0;
            unsigned pad_r = K_w_dim - K_w;

            int patch_start_col = col - K_w_half;
            int patch_end_col = col + K_w_half;

            if(patch_start_col < 0){
                pad_l = -patch_start_col;
            }

            if((width-1) < patch_end_col){
                pad_r = pad_r + (patch_end_col - (width-1));
            }

            uint32_t padding_mask = (pad_r == 0)? 0xFFFFFFFF : ((((uint32_t)1)<<(32-(X_pxl_bytes*pad_r)))-1);
            padding_mask = padding_mask ^ ((1<<(X_pxl_bytes*pad_l))-1);
            
            const int x_offset = x_row_offset + patch_start_col * X_pxl_bytes;

            const int8_t* X_start = X + x_offset;

            
            if(row == 2 && col == 0 && 1){
                printf("\n\n\n");
                printf("row:\t\t%d\n", row);
                printf("col:\t\t%d\n", col);
                printf("K:\t\t0x%x\n", K);
                printf("K_start:\t0x%x\n", K_start);
                printf("K_h:\t\t%d\n", K_h);
                printf("K_w:\t\t%d\n", K_w);
                printf("K_row_bytes:\t%d\n", K_row_bytes);
                printf("kernel_advance:\t%d\n", kernel_advance);
                
                printf("X:\t\t0x%x\n", X);
                printf("X_start:\t0x%x\n", X_start);
                printf("height:\t\t%d\n", height);
                printf("width:\t\t%d\n", width);
                printf("X_row_bytes:\t%d\n", X_row_bytes);
                printf("Y:\t\t0x%x\n", Y);
                printf("y_ptr:\t\t0x%x\n", y_ptr);

                printf("C_out:\t\t%d\n", C_out);
                printf("cout_groups:\t%d\n", cout_groups);
                printf("padding_mask:\t0x%x\n", padding_mask);
                printf("patch_rows:\t%d\n", patch_rows);

                printf("pad_t:\t\t%d\n", pad_t);
                printf("pad_b:\t\t%d\n", pad_b);
                printf("pad_l:\t\t%d\n", pad_l);
                printf("pad_r:\t\t%d\n", pad_r);

                printf("patch_start_row: %d\n", patch_start_row);
                printf("patch_end_row:   %d\n", patch_end_row);
                printf("patch_start_col: %d\n", patch_start_col);
                printf("patch_end_col:   %d\n", patch_end_col);
                
                printf("B_lo:\t\t0x%x\n", B_lo);
                printf("B_hi:\t\t0x%x\n", B_hi);
                printf("shifts:\t\t0x%x\n", shifts);
                printf("scales:\t\t0x%x\n", scales);
            }



            y_ptr = conv2d_shallowin_deepout_relu_asm_patch(
                    y_ptr, 
                    K_start,
                    cout_groups,
                    padding_mask,
                    X_start,
                    patch_rows,
                    B_lo,
                    B_hi,
                    shifts,
                    scales,
                    X_row_bytes,
                    kernel_advance);
            

        }
    }
}
}





#define DEBUG_ON    (0 || TEST_DEBUG_ON)
#define C_out       (VPU_INT8_ACC_PERIOD)
#define C_in        (4)
#define K_h         (1)
#define K_w         (1)
#define height      (8)
#define width       (8)
void test_conv2d_shallowin_deepout_relu_case1()
{
    PRINTF("test_conv2d_shallowin_deepout_relu_case1()...\n");

    int8_t   WORD_ALIGNED  K[C_out][K_h][8][C_in]           = {{{{ 0 }}}};
    data16_t WORD_ALIGNED  B[2][C_out]                      = {{ 0 }};
    int8_t   WORD_ALIGNED  X[height][width][C_in];
    int16_t  WORD_ALIGNED  shifts[C_out]                    = { 0 };
    int16_t  WORD_ALIGNED  scales[C_out]                    = { 0 };
    int8_t   WORD_ALIGNED  Y_expected[height][width][C_out] = {{{ 0 }}};

#if TEST_C
    PRINTF("\t\tC...\n");
    int8_t   WORD_ALIGNED  Y_c[height][width][C_out];
    memset(Y_c, 0xCC, sizeof(Y_c));
    conv2d_shallowin_deepout_relu_c((int8_t*) K, (data16_t*)B, (int8_t*) X, (int8_t*) Y_c, 
                                    height, width, K_h, K_w, C_out, shifts, scales);
#endif
#if HAS_ASM
    PRINTF("\t\tASM...\n");
    int8_t   WORD_ALIGNED  Y_asm[height][width][C_out];
    memset(Y_asm, 0xCC, sizeof(Y_asm));
    conv2d_shallowin_deepout_relu_asm((int8_t*) K, (data16_t*)B, (int8_t*) X, (int8_t*) Y_asm, 
                                height, width, K_h, K_w, C_out, shifts, scales);
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
#define C_out           (16)
#define C_in            (4)
#define K_h             (1)
#define K_w             (1)
#define height          (2)
#define width           (2)
#define TEST_VECTORS    (10)
#define VECTOR_FMT      ("test_data/conv2d_shallowin_deepout_relu_case2.%u.dat")
#include "../test_data/conv2d_shallowin_deepout_relu_case2.h"
void test_conv2d_shallowin_deepout_relu_case2()
{
    
    PRINTF("test_conv2d_shallowin_deepout_relu_case2()...\n");

    int8_t   WORD_ALIGNED  K[C_out][K_h][8][C_in];
    data16_t WORD_ALIGNED  B[2][C_out];
    int8_t   WORD_ALIGNED  X[height][width][C_in];
    int16_t  WORD_ALIGNED  shifts[C_out];
    int16_t  WORD_ALIGNED  scales[C_out];
    int8_t   WORD_ALIGNED  Y_expected[height][width][C_out]; 
    
    int16_t Y_check[] = { Y_CHECK };

    for(int v = 0; v < TEST_VECTORS; v++){
        PRINTF("\ttest vector %d...\n", v);

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
        int8_t   WORD_ALIGNED  Y_c[height][width][C_out];
        memset(Y_c, 0xCC, sizeof(Y_c));
        conv2d_shallowin_deepout_relu_c((int8_t*) K, (data16_t*)B, (int8_t*) X, (int8_t*) Y_c, 
                                    height, width, K_h, K_w, C_out, shifts, scales);
#endif
#if HAS_ASM
        PRINTF("\t\tASM...\n");
        int8_t   WORD_ALIGNED  Y_asm[height][width][C_out];
        memset(Y_asm, 0xCC, sizeof(Y_asm));
        conv2d_shallowin_deepout_relu_asm((int8_t*) K, (data16_t*)B, (int8_t*) X, (int8_t*) Y_asm, 
                                    height, width, K_h, K_w, C_out, shifts, scales);
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
#define C_out           (16)
#define C_in            (4)
#define K_h             (3)
#define K_w             (3)
#define height          (4)
#define width           (4)
#define TEST_VECTORS    (10)
#define VECTOR_FMT      ("test_data/conv2d_shallowin_deepout_relu_case3.%u.dat")
#include "../test_data/conv2d_shallowin_deepout_relu_case3.h"
void test_conv2d_shallowin_deepout_relu_case3()
{
    PRINTF("test_conv2d_shallowin_deepout_relu_case3()...\n");

    int8_t   WORD_ALIGNED  K[C_out][K_h][8][C_in];
    data16_t WORD_ALIGNED  B[2][C_out];
    int8_t   WORD_ALIGNED  X[height][width][C_in];
    int16_t  WORD_ALIGNED  shifts[C_out];
    int16_t  WORD_ALIGNED  scales[C_out];
    int8_t   WORD_ALIGNED  Y_expected[height][width][C_out]; 
    
    int16_t Y_check[] = { Y_CHECK };

    for(int v = 0; v < TEST_VECTORS; v++){
        PRINTF("\ttest vector %d...\n", v);

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
        int8_t   WORD_ALIGNED  Y_c[height][width][C_out];
        memset(Y_c, 0xCC, sizeof(Y_c));
        conv2d_shallowin_deepout_relu_c((int8_t*) K, (data16_t*)B, (int8_t*) X, (int8_t*) Y_c, 
                                    height, width, K_h, K_w, C_out, shifts, scales);
#endif
#if HAS_ASM
        PRINTF("\t\tASM...\n");
        int8_t   WORD_ALIGNED  Y_asm[height][width][C_out];
        memset(Y_asm, 0xCC, sizeof(Y_asm));
        conv2d_shallowin_deepout_relu_asm((int8_t*) K, (data16_t*)B, (int8_t*) X, (int8_t*) Y_asm, 
                                    height, width, K_h, K_w, C_out, shifts, scales);
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
#define C_out           (32)
#define C_in            (4)
#define K_h             (3)
#define K_w             (3)
#define height          (8)
#define width           (8)
#define TEST_VECTORS    (4)
#define VECTOR_FMT      ("test_data/conv2d_shallowin_deepout_relu_case4.%u.dat")
#include "../test_data/conv2d_shallowin_deepout_relu_case4.h"
void test_conv2d_shallowin_deepout_relu_case4()
{
    PRINTF("test_conv2d_shallowin_deepout_relu_case4()...\n");

    int8_t   WORD_ALIGNED  K[C_out][K_h][8][C_in];
    data16_t WORD_ALIGNED  B[2][C_out];
    int8_t   WORD_ALIGNED  X[height][width][C_in];
    int16_t  WORD_ALIGNED  shifts[C_out];
    int16_t  WORD_ALIGNED  scales[C_out];
    int8_t   WORD_ALIGNED  Y_expected[height][width][C_out]; 
    
    int16_t Y_check[] = { Y_CHECK };

    for(int v = 0; v < TEST_VECTORS; v++){
        PRINTF("\ttest vector %d...\n", v);
    

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
        int8_t   WORD_ALIGNED  Y_c[height][width][C_out];
        memset(Y_c, 0xCC, sizeof(Y_c));
        conv2d_shallowin_deepout_relu_c((int8_t*) K, (data16_t*)B, (int8_t*) X, (int8_t*) Y_c, 
                                    height, width, K_h, K_w, C_out, shifts, scales);
#endif
#if HAS_ASM
        PRINTF("\t\tASM...\n");
        int8_t   WORD_ALIGNED  Y_asm[height][width][C_out];
        memset(Y_asm, 0xCC, sizeof(Y_asm));
        conv2d_shallowin_deepout_relu_asm((int8_t*) K, (data16_t*)B, (int8_t*) X, (int8_t*) Y_asm, 
                                    height, width, K_h, K_w, C_out, shifts, scales);
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




void test_conv2d_shallowin_deepout_relu()
{
    test_conv2d_shallowin_deepout_relu_case1();
    test_conv2d_shallowin_deepout_relu_case2();
    test_conv2d_shallowin_deepout_relu_case3();
    test_conv2d_shallowin_deepout_relu_case4();
}
