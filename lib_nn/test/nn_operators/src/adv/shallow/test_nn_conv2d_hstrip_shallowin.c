
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <syscall.h>

#include "../../tst_common.h"

#include "nn_operator.h"
#include "nn_op_helper.h"
#include "xs3_vpu.h"

#include "unity.h"


#if USE_ASM(nn_conv2d_hstrip_shallowin)
 #define HAS_ASM (1)
#else
 #define HAS_ASM (0)
#endif

#define TEST_ASM ((HAS_ASM)     && 1)
#define TEST_C ((TEST_C_GLOBAL) && 1)

#if TEST_C && TEST_ASM
  #define Y_C_ASM  (int8_t*)Y_c, (int8_t*)Y_asm
#elif TEST_C && !TEST_ASM
  #define Y_C_ASM (int8_t*)Y_c
#elif !TEST_C && TEST_ASM
  #define Y_C_ASM (int8_t*)Y_asm
#else
  #error Neither TEST_C nor TEST_ASM is specified.
#endif

#define DO_PRINT_EXTRA ((DO_PRINT_EXTRA_GLOBAL) && 0)





static void check_Y(
    const int8_t y_exp, 
    const unsigned row,
    const unsigned col,
    const unsigned chn,
    const unsigned line,
#if TEST_C
    const nn_image_t* Y_c,
#endif
#if TEST_ASM
    const nn_image_t* Y_asm,
#endif
    const nn_image_params_t* y_params)
{
    char str_buff[200];

    unsigned y_offset = IMG_ADDRESS_VECT(y_params, row, col, chn);

    int flg = 0;

    //Only sprintf-ing if the test will fail saves a ton of time.
#if TEST_C
    int8_t y_c = Y_c[y_offset];
    flg |= (y_c == y_exp)? 0x00 : 0x01;
#endif
#if TEST_ASM
    int8_t y_asm = Y_asm[y_offset];
    flg |= (y_asm == y_exp)? 0x00 : 0x02;
#endif

    if(flg){
        sprintf(str_buff, "%s%s%s failed. (row, col, chn) = (%u, %u, %u)  [test vector @ %u]", 
                (flg&0x01)? "C" : "", (flg==0x03)? " and " : "", (flg&0x02)? "ASM" : "",
                row, col, chn, line);
    }

#if TEST_C
    TEST_ASSERT_EQUAL_MESSAGE(y_exp, y_c, str_buff);
#endif
#if TEST_ASM
    TEST_ASSERT_EQUAL_MESSAGE(y_exp, y_asm, str_buff);
#endif
}




#define CHANS_OUT       (VPU_INT8_ACC_PERIOD)
#define Y_HEIGHT        (1)
#define K_w_array       (32/CHANS_IN)







#define CHANS_IN        (4)
#define K_h             (1)
#define K_w             (1)
#define K_hstride       (1)
#define X_HEIGHT        (1)
#define X_WIDTH         (1)
#define Y_WIDTH         (1)
void test_nn_conv2d_hstrip_shallowin_case0()
{
    PRINTF("%s...\n", __func__);

    nn_image_t WORD_ALIGNED  X[X_HEIGHT][X_WIDTH][CHANS_IN];

    nn_tensor_t WORD_ALIGNED  K[CHANS_OUT][K_h][K_w_array][CHANS_IN];

    struct {
        int32_t bias[CHANS_OUT];
        int16_t shift1[CHANS_OUT];
        int16_t scale[CHANS_OUT];
        int16_t shift2[CHANS_OUT];
    } BSS;

    nn_bss_block_t bss[BSS_BLOCK_COUNT(CHANS_OUT)];

    nn_image_t WORD_ALIGNED  Y_c[Y_HEIGHT][Y_WIDTH][CHANS_OUT];
    nn_image_t WORD_ALIGNED  Y_asm[Y_HEIGHT][Y_WIDTH][CHANS_OUT];
    
    typedef struct {
        int8_t x;
        int8_t k;
        int8_t expected;
        unsigned line;
    } test_case_t;

    test_case_t casses[] = {
        //  x,  k,        expected,   line
        {   0,  0,               0,   __LINE__ },
        {   1,  0,               0,   __LINE__ },
        {   0,  1,               0,   __LINE__ },
        {   1,  1,        CHANS_IN,   __LINE__ },
        {   2,  1,      2*CHANS_IN,   __LINE__ },
        {   1,  2,      2*CHANS_IN,   __LINE__ },
    };

    const unsigned N_casses = sizeof(casses)/sizeof(test_case_t);

    const unsigned start_case = 0;
    const unsigned stop_case = -1;

    print_warns(start_case, TEST_C, TEST_ASM);

    for(int v = start_case; v < N_casses && v <= stop_case; v++){
        PRINTF("\tvector %d..\n", v);

        test_case_t* casse = &casses[v];

        nn_image_params_t x_params = { X_HEIGHT, X_WIDTH, CHANS_IN  };
        nn_image_params_t y_params = { Y_HEIGHT, Y_WIDTH, CHANS_OUT };


        memset(X, casse->x, x_params.height * x_params.width * x_params.channels * sizeof(int8_t));
                        
        for(int cout = 0; cout < y_params.channels; cout++)
            for(int row = 0; row < K_h; row++)
                for(int col = 0; col < K_w_array; col++)
                    for(int cin = 0; cin < x_params.channels; cin++)
                        K[cout][row][col][cin] = (col < K_w)? casse->k : 0;

        for(int k = 0; k < y_params.channels; k++){
            BSS.bias[k]     = k;
            BSS.shift1[k]   = 0;
            BSS.scale[k]    = 1;
            BSS.shift2[k]   = 0;
        }

        nn_standard_BSS_layout((data16_t*) &bss, (int32_t*) &BSS.bias, (int16_t*) &BSS.shift1, 
                                (int16_t*) &BSS.scale, (int16_t*) &BSS.shift2, NULL, y_params.channels);

        const mem_stride_t x_v_stride = x_params.width * x_params.channels;
        const nn_tensor_t* K_init = &K[y_params.channels-1][0][0][0];

#if TEST_C
        PRINTF("\t\t\tC...\n");
        memset(Y_c, 0xCC, sizeof(Y_c));
        nn_conv2d_hstrip_shallowin_c((nn_image_t*) Y_c, (nn_image_t*) X, K_init, (nn_bss_block_t*) &bss, 
                                        K_h, K_hstride, x_params.channels,
                                        x_v_stride, y_params.channels, y_params.width);
#endif
#if TEST_ASM
        PRINTF("\t\t\tASM...\n");
        memset(Y_asm, 0xCC, sizeof(Y_asm));
        nn_conv2d_hstrip_shallowin_asm((nn_image_t*) Y_asm, (nn_image_t*) X, K_init, (nn_bss_block_t*) &bss, 
                                        K_h, K_hstride, x_params.channels,
                                        x_v_stride, y_params.channels, y_params.width);
#endif

    
        PRINTF("\t\t\tChecking...\n");
        for(unsigned row = 0; row < y_params.height; row++){
            for(unsigned col = 0; col < y_params.width; col++){
                for(unsigned chn = 0; chn < y_params.channels; chn++){
                    int8_t y_exp = casse->expected + chn;
                    check_Y(y_exp, row, col, chn, casse->line, Y_C_ASM, &y_params);
                }
            }
        }
    }
}
#undef CHANS_IN  
#undef K_h       
#undef K_w       
#undef K_hstride 
#undef X_HEIGHT  
#undef X_WIDTH   
#undef Y_WIDTH   
#undef ZERO_POINT










#define CHANS_IN        (4)
#define X_HEIGHT        (1)
#define X_WIDTH         (3)
#define Y_HEIGHT        (1)
#define Y_WIDTH         (3)
#define K_h             (1)
#define K_w             (1)
#define K_hstride       (1)
void test_nn_conv2d_hstrip_shallowin_case1()
{
    PRINTF("%s...\n", __func__);

    nn_image_t WORD_ALIGNED  X[X_HEIGHT][X_WIDTH][CHANS_IN];

    nn_tensor_t WORD_ALIGNED  K[CHANS_OUT][K_h][K_w_array][CHANS_IN];

    struct {
        int32_t bias[CHANS_OUT];
        int16_t shift1[CHANS_OUT];
        int16_t scale[CHANS_OUT];
        int16_t shift2[CHANS_OUT];
    } BSS;

    nn_bss_block_t bss[BSS_BLOCK_COUNT(CHANS_OUT)];

    nn_image_t WORD_ALIGNED  Y_c[Y_HEIGHT][Y_WIDTH][CHANS_OUT];
    nn_image_t WORD_ALIGNED  Y_asm[Y_HEIGHT][Y_WIDTH][CHANS_OUT];
    
    typedef struct {
        int8_t x;
        int8_t k;
        int8_t expected;
        unsigned line;
    } test_case_t;

    test_case_t casses[] = {
        //  x,  k,        expected,   line
        {   0,  0,               0,   __LINE__ },
        {   1,  0,               0,   __LINE__ },
        {   0,  1,               0,   __LINE__ },
        {   1,  1,        CHANS_IN,   __LINE__ },
        {   2,  1,      2*CHANS_IN,   __LINE__ },
        {   1,  2,      2*CHANS_IN,   __LINE__ },
    };

    const unsigned N_casses = sizeof(casses)/sizeof(test_case_t);

    const unsigned start_case = 0;
    const unsigned stop_case = -1;

    print_warns(start_case, TEST_C, TEST_ASM);

    for(int v = start_case; v < N_casses && v <= stop_case; v++){
        PRINTF("\tvector %d..\n", v);

        test_case_t* casse = &casses[v];

        nn_image_params_t x_params = { X_HEIGHT, X_WIDTH, CHANS_IN };
        nn_image_params_t y_params = { Y_HEIGHT, Y_WIDTH, CHANS_OUT };

        for(int row = 0; row < X_HEIGHT; row++)
            for(int col = 0; col < X_WIDTH; col++)
                for(int cin = 0; cin < x_params.channels; cin++)
                    X[row][col][cin] = casse->x * (col+1);

        for(int cout = 0; cout < y_params.channels; cout++)
            for(int row = 0; row < K_h; row++)
                for(int col = 0; col < K_w_array; col++)
                    for(int cin = 0; cin < x_params.channels; cin++)
                        K[cout][row][col][cin] = (col < K_w)? casse->k : 0;

        for(int k = 0; k < y_params.channels; k++){
            BSS.bias[k]     =-k;
            BSS.shift1[k]   = 0;
            BSS.scale[k]    = 1;
            BSS.shift2[k]   = 0;
        }

        nn_standard_BSS_layout((data16_t*) &bss, (int32_t*) &BSS.bias, (int16_t*) &BSS.shift1, 
                                (int16_t*) &BSS.scale, (int16_t*) &BSS.shift2, NULL, CHANS_OUT);
            
        const mem_stride_t x_v_stride = x_params.width*x_params.channels;
        const nn_tensor_t* K_init = &K[y_params.channels-1][0][0][0];
        nn_image_t* X_patch_start = &X[0][0][0];

#if TEST_C
        PRINTF("\t\t\tC...\n");
        memset(Y_c, 0xCC, sizeof(Y_c));
        nn_conv2d_hstrip_shallowin_c((nn_image_t*) Y_c, X_patch_start, K_init, (nn_bss_block_t*) &bss, 
                                        K_h, K_hstride, x_params.channels, 
                                        x_v_stride, y_params.channels, y_params.width);
#endif
#if TEST_ASM
        PRINTF("\t\t\tASM...\n");
        memset(Y_asm, 0xCC, sizeof(Y_asm));
        nn_conv2d_hstrip_shallowin_asm((nn_image_t*) Y_asm, X_patch_start, K_init, (nn_bss_block_t*) &bss, 
                                        K_h, K_hstride, x_params.channels,
                                        x_v_stride, y_params.channels, y_params.width);
#endif

    
        PRINTF("\t\t\tChecking...\n");
        for(unsigned row = 0; row < y_params.height; row++){
            for(unsigned col = 0; col < y_params.width; col++){
                for(unsigned chn = 0; chn < y_params.channels; chn++){
                    
                    int8_t y_exp = (col+1) * casse->expected - chn;

                    check_Y(y_exp, row, col, chn, casse->line, Y_C_ASM, &y_params);
                }
            }
        }
    }
}
#undef CHANS_IN  
#undef X_HEIGHT  
#undef X_WIDTH   
#undef Y_HEIGHT  
#undef Y_WIDTH   
#undef K_h       
#undef K_w       
#undef K_hstride










#define CHANS_IN        (4)
#define X_HEIGHT        (3)
#define X_WIDTH         (3)
#define Y_HEIGHT        (1)
#define Y_WIDTH         (1)
#define K_h             (3)
#define K_w             (3)
#define K_hstride       (1)
void test_nn_conv2d_hstrip_shallowin_case2()
{
    PRINTF("%s...\n", __func__);

    nn_image_t WORD_ALIGNED  X[X_HEIGHT][X_WIDTH][CHANS_IN];

    nn_tensor_t WORD_ALIGNED  K[CHANS_OUT][K_h][K_w_array][CHANS_IN];

    struct {
        int32_t bias[CHANS_OUT];
        int16_t shift1[CHANS_OUT];
        int16_t scale[CHANS_OUT];
        int16_t shift2[CHANS_OUT];
    } BSS;

    nn_bss_block_t bss[BSS_BLOCK_COUNT(CHANS_OUT)];

    nn_image_t WORD_ALIGNED  Y_c[Y_HEIGHT][Y_WIDTH][CHANS_OUT];
    nn_image_t WORD_ALIGNED  Y_asm[Y_HEIGHT][Y_WIDTH][CHANS_OUT];
    
    print_warns(0, TEST_C, TEST_ASM);


    nn_image_params_t x_params = { X_HEIGHT, X_WIDTH, CHANS_IN };
    nn_image_params_t y_params = { Y_HEIGHT, Y_WIDTH, CHANS_OUT };

    int8_t tmp = -10;

    for(int row = 0; row < X_HEIGHT; row++)
        for(int col = 0; col < X_WIDTH; col++)
            for(int cin = 0; cin < x_params.channels; cin++)
                X[row][col][cin] = tmp++;

    for(int cout = 0; cout < y_params.channels; cout++)
        for(int row = 0; row < K_h; row++)
            for(int col = 0; col < K_w_array; col++)
                for(int cin = 0; cin < x_params.channels; cin++)
                    K[cout][row][col][cin] = (col < K_w)? tmp++ : 0;

    int16_t shift1s[] = { 8, 8, 2, 8, 8, 6, 6, 7, 8, 8, 9, 9, 7, 5, 7, 8 };
    for(int k = 0; k < y_params.channels; k++){
        BSS.bias[k]     = - 100 * k;
        BSS.shift1[k]   = shift1s[k];
        BSS.scale[k]    = 8;
        BSS.shift2[k]   = 3;
    }

    nn_standard_BSS_layout((data16_t*) &bss, (int32_t*) &BSS.bias, (int16_t*) &BSS.shift1, 
                            (int16_t*) &BSS.scale, (int16_t*) &BSS.shift2, NULL, CHANS_OUT);
        
    const mem_stride_t x_v_stride = x_params.width*x_params.channels;
    const nn_tensor_t* K_init = &K[y_params.channels-1][0][0][0];
    nn_image_t* X_patch_start = &X[0][0][0];

#if TEST_C
    PRINTF("\t\t\tC...\n");
    memset(Y_c, 0xCC, sizeof(Y_c));
    nn_conv2d_hstrip_shallowin_c((nn_image_t*) Y_c, X_patch_start, K_init, (nn_bss_block_t*) &bss, 
                                    K_h, K_hstride, x_params.channels, 
                                    x_v_stride, y_params.channels, y_params.width);
#endif
#if TEST_ASM
    PRINTF("\t\t\tASM...\n");
    memset(Y_asm, 0xCC, sizeof(Y_asm));
    nn_conv2d_hstrip_shallowin_asm((nn_image_t*) Y_asm, X_patch_start, K_init, (nn_bss_block_t*) &bss, 
                                    K_h, K_hstride, x_params.channels,
                                    x_v_stride, y_params.channels, y_params.width);
#endif


    PRINTF("\t\t\tChecking...\n");
    for(unsigned row = 0; row < y_params.height; row++){
        for(unsigned col = 0; col < y_params.width; col++){
            for(unsigned chn = 0; chn < y_params.channels; chn++){
                
                int32_t acc = BSS.bias[chn];

                for(int xr = 0; xr < x_params.height; xr++)
                    for(int xc = 0; xc < x_params.width; xc++)
                        for(int xchn = 0; xchn < x_params.channels; xchn++)
                            acc += ((int32_t)X[xr][xc][xchn]) * K[chn][xr][xc][xchn];

                acc = (acc + (1<<(BSS.shift1[chn]-1))) >> BSS.shift1[chn];
                acc = acc * BSS.scale[chn];
                acc = (acc + (1<<(BSS.shift2[chn]-1))) >> BSS.shift2[chn];

                int8_t y_exp = acc;

                check_Y(y_exp, row, col, chn, __LINE__, Y_C_ASM, &y_params);
            }
        }
    }
}
#undef CHANS_IN  
#undef X_HEIGHT  
#undef X_WIDTH   
#undef Y_HEIGHT  
#undef Y_WIDTH   
#undef K_h       
#undef K_w       
#undef K_hstride










#define CHANS_IN        (4)
#define X_HEIGHT        (3)
#define X_WIDTH         (5)
#define Y_HEIGHT        (1)
#define Y_WIDTH         (3)
#define K_h             (3)
#define K_w             (3)
#define K_hstride       (1)
void test_nn_conv2d_hstrip_shallowin_case3()
{
    PRINTF("%s...\n", __func__);

    nn_image_t WORD_ALIGNED  X[X_HEIGHT][X_WIDTH][CHANS_IN];

    nn_tensor_t WORD_ALIGNED  K[CHANS_OUT][K_h][K_w_array][CHANS_IN];

    struct {
        int32_t bias[CHANS_OUT];
        int16_t shift1[CHANS_OUT];
        int16_t scale[CHANS_OUT];
        int16_t shift2[CHANS_OUT];
    } BSS;

    nn_bss_block_t bss[BSS_BLOCK_COUNT(CHANS_OUT)];

    nn_image_t WORD_ALIGNED  Y_c[Y_HEIGHT][Y_WIDTH][CHANS_OUT];
    nn_image_t WORD_ALIGNED  Y_asm[Y_HEIGHT][Y_WIDTH][CHANS_OUT];
    
    print_warns(0, TEST_C, TEST_ASM);


    nn_image_params_t x_params = { X_HEIGHT, X_WIDTH, CHANS_IN };
    nn_image_params_t y_params = { Y_HEIGHT, Y_WIDTH, CHANS_OUT };

    int8_t tmp = -10;

    for(int row = 0; row < X_HEIGHT; row++)
        for(int col = 0; col < X_WIDTH; col++)
            for(int cin = 0; cin < x_params.channels; cin++)
                X[row][col][cin] = tmp++;

    for(int cout = 0; cout < y_params.channels; cout++)
        for(int row = 0; row < K_h; row++)
            for(int col = 0; col < K_w_array; col++)
                for(int cin = 0; cin < x_params.channels; cin++)
                    K[cout][row][col][cin] = (col < K_w)? tmp++ : 0;

    int16_t shift1s[] = { 10, 11, 26, 26, 26, 26, 9, 10, 11, 26, 26, 26, 26, 9, 10, 11 };
    for(int k = 0; k < y_params.channels; k++){
        BSS.bias[k]     = - 100 * k;
        BSS.shift1[k]   = shift1s[k];
        BSS.scale[k]    = 8;
        BSS.shift2[k]   = 3;
    }

    nn_standard_BSS_layout((data16_t*) &bss, (int32_t*) &BSS.bias, (int16_t*) &BSS.shift1, 
                            (int16_t*) &BSS.scale, (int16_t*) &BSS.shift2, NULL, CHANS_OUT);
        
    const mem_stride_t x_v_stride = x_params.width*x_params.channels;
    const nn_tensor_t* K_init = &K[y_params.channels-1][0][0][0];
    nn_image_t* X_patch_start = &X[0][0][0];

#if TEST_C
    PRINTF("\t\t\tC...\n");
    memset(Y_c, 0xCC, sizeof(Y_c));
    nn_conv2d_hstrip_shallowin_c((nn_image_t*) Y_c, X_patch_start, K_init, (nn_bss_block_t*) &bss, 
                                    K_h, K_hstride, x_params.channels, 
                                    x_v_stride, y_params.channels, y_params.width);
#endif
#if TEST_ASM
    PRINTF("\t\t\tASM...\n");
    memset(Y_asm, 0xCC, sizeof(Y_asm));
    nn_conv2d_hstrip_shallowin_asm((nn_image_t*) Y_asm, X_patch_start, K_init, (nn_bss_block_t*) &bss, 
                                    K_h, K_hstride, x_params.channels,
                                    x_v_stride, y_params.channels, y_params.width);
#endif


    PRINTF("\t\t\tChecking...\n");
    for(unsigned row = 0; row < y_params.height; row++){
        for(unsigned col = 0; col < y_params.width; col++){
            for(unsigned chn = 0; chn < y_params.channels; chn++){
                
                int32_t acc = BSS.bias[chn];

                for(int xr = 0; xr < x_params.height; xr++)
                    for(int xc = 0; xc < x_params.width; xc++)
                        for(int xchn = 0; xchn < x_params.channels; xchn++)
                            acc += ((int32_t)X[xr][col+xc][xchn]) * K[chn][xr][xc][xchn];

                acc = (acc + (1<<(BSS.shift1[chn]-1))) >> BSS.shift1[chn];
                acc = acc * BSS.scale[chn];
                acc = (acc + (1<<(BSS.shift2[chn]-1))) >> BSS.shift2[chn];

                int8_t y_exp = acc;

                check_Y(y_exp, row, col, chn, __LINE__, Y_C_ASM, &y_params);
            }
        }
    }
}
#undef CHANS_IN  
#undef X_HEIGHT  
#undef X_WIDTH   
#undef Y_HEIGHT  
#undef Y_WIDTH   
#undef K_h       
#undef K_w       
#undef K_hstride





void test_nn_conv2d_hstrip_shallowin()
{
    UNITY_SET_FILE();

    RUN_TEST(test_nn_conv2d_hstrip_shallowin_case0);
    RUN_TEST(test_nn_conv2d_hstrip_shallowin_case1);
    RUN_TEST(test_nn_conv2d_hstrip_shallowin_case2);
    RUN_TEST(test_nn_conv2d_hstrip_shallowin_case3);
}