
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>


#include "../../tst_common.h"

#include "nn_operator.h"
#include "nn_op_helper.h"
#include "xs3_vpu.h"

#include "unity.h"


#define DO_PRINT_EXTRA ((DO_PRINT_EXTRA_GLOBAL) && 0)





static void check_Y(
    const int8_t y_exp, 
    const unsigned row,
    const unsigned col,
    const unsigned chn,
    const unsigned line,
    const nn_image_t* Y,
    const nn_image_params_t* y_params)
{
    char str_buff[200];

    unsigned y_offset = IMG_ADDRESS_VECT(y_params, row, col, chn);

    int flg = 0;

    //Only sprintf-ing if the test will fail saves a ton of time.
    int8_t y = Y[y_offset];

    if(y != y_exp){
        sprintf(str_buff, "(row, col, chn) = (%u, %u, %u)  [test vector @ %u]", 
                row, col, chn, line);
    }

    TEST_ASSERT_EQUAL_MESSAGE(y_exp, y, str_buff);
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
        int16_t offset_scale[CHANS_OUT];
        int16_t offset[CHANS_OUT];
        int16_t shift2[CHANS_OUT];
    } BSS;

    nn_bss_block_t bss[BSS_BLOCK_COUNT(CHANS_OUT)];

    nn_image_t WORD_ALIGNED  Y[Y_HEIGHT][Y_WIDTH][CHANS_OUT];
    
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

    print_warns(start_case);

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
            BSS.offset_scale[k] = 0;
            BSS.offset[k]       = 0;
            BSS.shift2[k]   = 0;
        }

        nn_standard_BSS_layout(bss, (int32_t*) &BSS.bias, (int16_t*) &BSS.shift1, 
                                (int16_t*) &BSS.scale, (int16_t*) &BSS.offset_scale, (int16_t*) &BSS.offset, (int16_t*) &BSS.shift2, NULL, y_params.channels);

        const mem_stride_t x_v_stride = x_params.width * x_params.channels;
        const nn_tensor_t* K_init = &K[y_params.channels-1][0][0][0];


        memset(Y, 0xCC, sizeof(Y));
        nn_conv2d_hstrip_shallowin((nn_image_t*) Y, (nn_image_t*) X, K_init, (nn_bss_block_t*) &bss, 
                                        K_h, K_hstride, x_params.channels,
                                        x_v_stride, y_params.channels, y_params.width);

    
        PRINTF("\t\t\tChecking...\n");
        for(unsigned row = 0; row < y_params.height; row++){
            for(unsigned col = 0; col < y_params.width; col++){
                for(unsigned chn = 0; chn < y_params.channels; chn++){
                    int8_t y_exp = casse->expected + chn;
                    check_Y(y_exp, row, col, chn, casse->line, (nn_image_t*) Y, &y_params);
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
        int16_t offset_scale[CHANS_OUT];
        int16_t offset[CHANS_OUT];
        int16_t shift2[CHANS_OUT];
    } BSS;

    nn_bss_block_t bss[BSS_BLOCK_COUNT(CHANS_OUT)];

    nn_image_t WORD_ALIGNED  Y[Y_HEIGHT][Y_WIDTH][CHANS_OUT];
    
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

    print_warns(start_case);

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
            BSS.offset_scale[k] = 0;
            BSS.offset[k]       = 0;
            BSS.shift2[k]   = 0;
        }

        nn_standard_BSS_layout(bss, (int32_t*) &BSS.bias, (int16_t*) &BSS.shift1, 
                                (int16_t*) &BSS.scale, (int16_t*) &BSS.offset_scale, (int16_t*) &BSS.offset, (int16_t*) &BSS.shift2, NULL, CHANS_OUT);
            
        const mem_stride_t x_v_stride = x_params.width*x_params.channels;
        const nn_tensor_t* K_init = &K[y_params.channels-1][0][0][0];
        nn_image_t* X_patch_start = &X[0][0][0];


        memset(Y, 0xCC, sizeof(Y));
        nn_conv2d_hstrip_shallowin((nn_image_t*) Y, X_patch_start, K_init, (nn_bss_block_t*) &bss, 
                                        K_h, K_hstride, x_params.channels, 
                                        x_v_stride, y_params.channels, y_params.width);

    
        PRINTF("\t\t\tChecking...\n");
        for(unsigned row = 0; row < y_params.height; row++){
            for(unsigned col = 0; col < y_params.width; col++){
                for(unsigned chn = 0; chn < y_params.channels; chn++){
                    
                    int8_t y_exp = (col+1) * casse->expected - chn;

                    check_Y(y_exp, row, col, chn, casse->line, (nn_image_t*) Y, &y_params);
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
        int16_t offset_scale[CHANS_OUT];
        int16_t offset[CHANS_OUT];
        int16_t shift2[CHANS_OUT];
    } BSS;

    nn_bss_block_t bss[BSS_BLOCK_COUNT(CHANS_OUT)];

    nn_image_t WORD_ALIGNED  Y[Y_HEIGHT][Y_WIDTH][CHANS_OUT];
    
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
        BSS.offset_scale[k] = 0;
        BSS.offset[k]       = 0;
        BSS.shift2[k]   = 3;
    }

    nn_standard_BSS_layout(bss, (int32_t*) &BSS.bias, (int16_t*) &BSS.shift1, 
                            (int16_t*) &BSS.scale, (int16_t*) &BSS.offset_scale, (int16_t*) &BSS.offset, (int16_t*) &BSS.shift2, NULL, CHANS_OUT);
        
    const mem_stride_t x_v_stride = x_params.width*x_params.channels;
    const nn_tensor_t* K_init = &K[y_params.channels-1][0][0][0];
    nn_image_t* X_patch_start = &X[0][0][0];

    memset(Y, 0xCC, sizeof(Y));
    nn_conv2d_hstrip_shallowin((nn_image_t*) Y, X_patch_start, K_init, (nn_bss_block_t*) &bss, 
                                    K_h, K_hstride, x_params.channels, 
                                    x_v_stride, y_params.channels, y_params.width);


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

                check_Y(y_exp, row, col, chn, __LINE__, (nn_image_t*) Y, &y_params);
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
        int16_t offset_scale[CHANS_OUT];
        int16_t offset[CHANS_OUT];
        int16_t shift2[CHANS_OUT];
    } BSS;

    nn_bss_block_t bss[BSS_BLOCK_COUNT(CHANS_OUT)];

    nn_image_t WORD_ALIGNED  Y[Y_HEIGHT][Y_WIDTH][CHANS_OUT];
    
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
        BSS.offset_scale[k] = 0;
        BSS.offset[k]       = 0;
        BSS.shift2[k]   = 3;
    }

    nn_standard_BSS_layout(bss, (int32_t*) &BSS.bias, (int16_t*) &BSS.shift1, 
                            (int16_t*) &BSS.scale, (int16_t*) &BSS.offset_scale, (int16_t*) &BSS.offset, (int16_t*) &BSS.shift2, NULL, CHANS_OUT);
        
    const mem_stride_t x_v_stride = x_params.width*x_params.channels;
    const nn_tensor_t* K_init = &K[y_params.channels-1][0][0][0];
    nn_image_t* X_patch_start = &X[0][0][0];

    memset(Y, 0xCC, sizeof(Y));
    nn_conv2d_hstrip_shallowin((nn_image_t*) Y, X_patch_start, K_init, (nn_bss_block_t*) &bss, 
                                    K_h, K_hstride, x_params.channels, 
                                    x_v_stride, y_params.channels, y_params.width);


    PRINTF("\t\t\tChecking...\n");
    for(unsigned row = 0; row < Y_HEIGHT; row++){
        for(unsigned col = 0; col < Y_WIDTH; col++){
            for(unsigned chn = 0; chn < y_params.channels; chn++){
                
                int32_t acc = BSS.bias[chn];

                for(unsigned xr = 0; xr < X_HEIGHT; xr++){
                    for(unsigned xc = 0; xc < X_WIDTH; xc++){
                        for(unsigned xchn = 0; xchn < x_params.channels; xchn++){
                            int32_t xv = X[xr][col+xc][xchn];
                            int32_t kv = K[chn][xr][xc][xchn];
                            acc += xv*kv;
                        }
                    }
                }

                acc = (acc + (1<<(BSS.shift1[chn]-1))) >> BSS.shift1[chn];
                acc = acc * BSS.scale[chn];
                acc = (acc + (1<<(BSS.shift2[chn]-1))) >> BSS.shift2[chn];

                int8_t y_exp = acc;

                check_Y(y_exp, row, col, chn, __LINE__, (nn_image_t*) Y, &y_params);
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