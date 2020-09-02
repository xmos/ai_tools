
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


#if CONFIG_SYMMETRIC_SATURATION_conv2d_deep
  #define NEG_SAT_VAL   (-127)
#else
  #define NEG_SAT_VAL   (-128)
#endif 



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

    //Only sprintf-ing if the test will fail saves a ton of time.
    int8_t y = Y[y_offset];

    if(y != y_exp){
        sprintf(str_buff, "(row, col, chn) = (%u, %u, %u)  [test vector @ %u]", 
                row, col, chn, line);
    }

    TEST_ASSERT_EQUAL_MESSAGE(y_exp, y, str_buff);
}















///////////////////////////////////////////////////
///     1 pixel; no padding
///////////////////////////////////////////////////
#define DEBUG_ON        TEST_DEBUG_ON && 0
#define CHANS_IN        (VPU_INT8_EPV + 4)
#define CHANS_OUT       (VPU_INT8_ACC_PERIOD)
#define X_HEIGHT        (1)
#define X_WIDTH         (1)
#define Y_HEIGHT        (1)
#define Y_WIDTH         (X_WIDTH)
#define K_h             (1)
#define K_w             (1)
#define K_hstride       (1)
#define ZERO_POINT      (0)
void test_nn_conv2d_hstrip_deep_padded_case0()
{
    PRINTF("%s...\n", __func__);

    struct {
        int32_t bias[CHANS_OUT];
        int16_t shift1[CHANS_OUT];
        int16_t scale[CHANS_OUT];
        int16_t offset_scale[CHANS_OUT];
        int16_t offset[CHANS_OUT];
        int16_t shift2[CHANS_OUT];
    } BSO;

    nn_image_t WORD_ALIGNED  Y[Y_HEIGHT][Y_WIDTH][CHANS_OUT];
    nn_image_t WORD_ALIGNED  X[X_HEIGHT][X_WIDTH][CHANS_IN];
    nn_tensor_t WORD_ALIGNED  K[CHANS_OUT][K_h][K_w][CHANS_IN];
    nn_bso_block_t bso[BSO_BLOCK_COUNT(CHANS_OUT)];


    int8_t zero_point_vec[VPU_INT8_EPV];
    memset(zero_point_vec, ZERO_POINT, sizeof(zero_point_vec));
    
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


        memset(X, casse->x, x_params.height * x_params.width * x_params.channels * sizeof(int8_t));
        memset(K, casse->k, y_params.channels * K_h * K_w * x_params.channels * sizeof(int8_t));


        for(int k = 0; k < y_params.channels; k++){
            BSO.bias[k]     = k;
            BSO.shift1[k]   = 0;
            BSO.scale[k]    = 1;
            BSO.offset_scale[k] = 0;
            BSO.offset[k]       = 0;
            BSO.shift2[k]   = 0;
        }

        nn_standard_BSO_layout(bso, (int32_t*) &BSO.bias, (int16_t*) &BSO.shift1, 
                                (int16_t*) &BSO.scale, (int16_t*) &BSO.offset_scale, (int16_t*) &BSO.offset, (int16_t*) &BSO.shift2, NULL, CHANS_OUT);

        memset(Y, 0xCC, sizeof(Y));
        nn_conv2d_hstrip_deep_padded((nn_image_t*) Y, (nn_image_t*) X, KERNEL_4D_COG_LAST_CHAN_START(K, 0), 
                                        (nn_bso_block_t*) &bso, K_h, K_w, K_hstride, x_params.channels, 0, 0, 0, 0,
                                        (x_params.width-K_w)*x_params.channels, -K_h*K_w*x_params.channels,
                                        y_params.channels, 1, zero_point_vec);

    
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
#undef DEBUG_ON  
#undef CHANS_IN  
#undef CHANS_OUT 
#undef X_HEIGHT  
#undef X_WIDTH   
#undef Y_HEIGHT  
#undef Y_WIDTH   
#undef K_h       
#undef K_w       
#undef K_hstride 
#undef ZERO_POINT













///////////////////////////////////////////////////
///     1 pixel; Top and bottom padding
///////////////////////////////////////////////////
#define DEBUG_ON        TEST_DEBUG_ON && 0
#define CHANS_IN        (VPU_INT8_EPV + 4)
#define CHANS_OUT       (VPU_INT8_ACC_PERIOD)
#define X_HEIGHT        (1)
#define X_WIDTH         (1)
#define Y_HEIGHT        (1)
#define Y_WIDTH         (X_WIDTH)
#define K_h             (3)
#define K_w             (1)
#define K_hstride       (1)
void test_nn_conv2d_hstrip_deep_padded_case1()
{
    PRINTF("%s...\n", __func__);

    struct {
        int32_t bias[CHANS_OUT];
        int16_t shift1[CHANS_OUT];
        int16_t scale[CHANS_OUT];
        int16_t offset_scale[CHANS_OUT];
        int16_t offset[CHANS_OUT];
        int16_t shift2[CHANS_OUT];
    } BSO;

    nn_image_t WORD_ALIGNED  X[X_HEIGHT][X_WIDTH][CHANS_IN];
    nn_tensor_t WORD_ALIGNED  K[CHANS_OUT][K_h][K_w][CHANS_IN];
    nn_bso_block_t bso[BSO_BLOCK_COUNT(CHANS_OUT)];
    nn_image_t WORD_ALIGNED  Y[Y_HEIGHT][Y_WIDTH][CHANS_OUT];

    
    typedef struct {
        int8_t x;
        int8_t k;
        int8_t zero_point;
        int8_t expected;
        unsigned line;
    } test_case_t;

    test_case_t casses[] = {
        //  x,  k,  zero_point,     expected,   line
        {   0,  0,           0,            0,   __LINE__ },
        {   1,  0,           0,            0,   __LINE__ },
        {   0,  1,           0,            0,   __LINE__ },
        {   1,  1,           0,     CHANS_IN,   __LINE__ },
        {   2,  1,           0,   2*CHANS_IN,   __LINE__ },
        {   1,  2,           0,   2*CHANS_IN,   __LINE__ },
        {  -2,  1,           0,  -2*CHANS_IN,   __LINE__ },
        {   1, -2,           0,  -2*CHANS_IN,   __LINE__ },
        {   0,  0,           1,            0,   __LINE__ },
        {   1,  0,           1,            0,   __LINE__ },
        {   0,  1,           1,   2*CHANS_IN,   __LINE__ },
        {   1,  1,           1,   3*CHANS_IN,   __LINE__ },
        {  -1,  1,           2,   3*CHANS_IN,   __LINE__ },
        {  -2,  1,           2,   2*CHANS_IN,   __LINE__ },
        {  -4,  1,           2,            0,   __LINE__ },
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

        int8_t zero_point_vec[VPU_INT8_EPV];
        memset(zero_point_vec, casse->zero_point, sizeof(zero_point_vec));

        memset(X, casse->x, x_params.height * x_params.width * x_params.channels * sizeof(int8_t));
        memset(K, casse->k, y_params.channels * K_h * K_w * x_params.channels * sizeof(int8_t));


        for(int k = 0; k < y_params.channels; k++){
            BSO.bias[k]     = k;
            BSO.shift1[k]   = 0;
            BSO.scale[k]    = 1;
            BSO.offset_scale[k] = 0;
            BSO.offset[k]       = 0;
            BSO.shift2[k]   = 0;
        }

        nn_standard_BSO_layout(bso, (int32_t*) &BSO.bias, (int16_t*) &BSO.shift1, 
                                (int16_t*) &BSO.scale, (int16_t*) &BSO.offset_scale, (int16_t*) &BSO.offset, (int16_t*) &BSO.shift2, NULL, CHANS_OUT);

        // Start the convolution window from each of 3 different positions
        // results same for each position
        for(int pad_t = 0; pad_t < 3; pad_t++){
            PRINTF("\t\t\tpad_t = %u\n", pad_t);
            
            int pad_b = 2 - pad_t;
            int pad_l = 0;
            int pad_r = 0;

            nn_image_t* X_patch_start = &X[-pad_t][-pad_l][0];

            memset(Y, 0xCC, sizeof(Y));
            nn_conv2d_hstrip_deep_padded((nn_image_t*) Y, X_patch_start, KERNEL_4D_COG_LAST_CHAN_START(K, 0), 
                                        (nn_bso_block_t*) &bso, K_h, K_w, K_hstride, x_params.channels, 
                                        pad_t, pad_b, pad_l, pad_r, (x_params.width-K_w)*x_params.channels, 
                                        -K_h*K_w*x_params.channels, y_params.channels, 1, zero_point_vec);

        
            PRINTF("\t\t\t\tChecking...\n");
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
}
#undef DEBUG_ON  
#undef CHANS_IN  
#undef CHANS_OUT 
#undef X_HEIGHT  
#undef X_WIDTH   
#undef Y_HEIGHT  
#undef Y_WIDTH   
#undef K_h       
#undef K_w       
#undef K_hstride 



















///////////////////////////////////////////////////
///     1 pixel; Left and right padding
///////////////////////////////////////////////////
#define DEBUG_ON        TEST_DEBUG_ON && 0
#define CHANS_IN        (VPU_INT8_EPV + 4)
#define CHANS_OUT       (VPU_INT8_ACC_PERIOD)
#define X_HEIGHT        (1)
#define X_WIDTH         (1)
#define Y_HEIGHT        (1)
#define Y_WIDTH         (1)
#define K_h             (1)
#define K_w             (3)
#define K_hstride       (1)
void test_nn_conv2d_hstrip_deep_padded_case2()
{
    PRINTF("%s...\n", __func__);

    struct {
        int32_t bias[CHANS_OUT];
        int16_t shift1[CHANS_OUT];
        int16_t scale[CHANS_OUT];
        int16_t offset_scale[CHANS_OUT];
        int16_t offset[CHANS_OUT];
        int16_t shift2[CHANS_OUT];
    } BSO;

    nn_image_t WORD_ALIGNED  X[X_HEIGHT][X_WIDTH][CHANS_IN];
    nn_tensor_t WORD_ALIGNED  K[CHANS_OUT][K_h][K_w][CHANS_IN];
    nn_bso_block_t bso[BSO_BLOCK_COUNT(CHANS_OUT)];
    nn_image_t WORD_ALIGNED  Y[Y_HEIGHT][Y_WIDTH][CHANS_OUT];

    
    typedef struct {
        int8_t x;
        int8_t k;
        int8_t zero_point;
        int8_t expected;
        unsigned line;
    } test_case_t;

    test_case_t casses[] = {
        //  x,  k,  zero_point,     expected,   line
        {   0,  0,           0,            0,   __LINE__ },
        {   1,  0,           0,            0,   __LINE__ },
        {   0,  1,           0,            0,   __LINE__ },
        {   1,  1,           0,     CHANS_IN,   __LINE__ },
        {   2,  1,           0,   2*CHANS_IN,   __LINE__ },
        {   1,  2,           0,   2*CHANS_IN,   __LINE__ },
        {  -2,  1,           0,  -2*CHANS_IN,   __LINE__ },
        {   1, -2,           0,  -2*CHANS_IN,   __LINE__ },
        {   0,  0,           1,            0,   __LINE__ },
        {   1,  0,           1,            0,   __LINE__ },
        {   0,  1,           1,   2*CHANS_IN,   __LINE__ },
        {   1,  1,           1,   3*CHANS_IN,   __LINE__ },
        {  -1,  1,           2,   3*CHANS_IN,   __LINE__ },
        {  -2,  1,           2,   2*CHANS_IN,   __LINE__ },
        {  -4,  1,           2,            0,   __LINE__ },
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

        int8_t zero_point_vec[VPU_INT8_EPV];
        memset(zero_point_vec, casse->zero_point, sizeof(zero_point_vec));

        memset(X, casse->x, x_params.height * x_params.width * x_params.channels * sizeof(int8_t));
        memset(K, casse->k, y_params.channels * K_h * K_w * x_params.channels * sizeof(int8_t));

        for(int k = 0; k < y_params.channels; k++){
            BSO.bias[k]     = 0;
            BSO.shift1[k]   = 0;
            BSO.scale[k]    = 1;
            BSO.offset_scale[k] = 1;
            BSO.offset[k]       = k;
            BSO.shift2[k]   = 0;
        }

        nn_standard_BSO_layout(bso, (int32_t*) &BSO.bias, (int16_t*) &BSO.shift1, 
                                (int16_t*) &BSO.scale, (int16_t*) &BSO.offset_scale, (int16_t*) &BSO.offset, (int16_t*) &BSO.shift2, NULL, CHANS_OUT);

        // Start the convolution window from each of 3 different positions
        // results same for each position
        for(int pad_l = 0; pad_l < 3; pad_l++){
            PRINTF("\t\t\tpad_l = %u\n", pad_l);
            
            int pad_t = 0;
            int pad_b = 0;
            int pad_r = 2 - pad_l;

            nn_image_t* X_patch_start = &X[-pad_t][-pad_l][0];

            memset(Y, 0xCC, sizeof(Y));
            nn_conv2d_hstrip_deep_padded((nn_image_t*) Y, X_patch_start, KERNEL_4D_COG_LAST_CHAN_START(K, 0), 
                                            (nn_bso_block_t*) &bso, K_h, K_w, K_hstride, x_params.channels, 
                                            pad_t, pad_b, pad_l, pad_r, (x_params.width-K_w)*x_params.channels, 
                                            -K_h*K_w*x_params.channels, y_params.channels, 1, zero_point_vec);

        
            PRINTF("\t\t\t\tChecking...\n");
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
}
#undef DEBUG_ON  
#undef CHANS_IN  
#undef CHANS_OUT 
#undef X_HEIGHT  
#undef X_WIDTH   
#undef Y_HEIGHT  
#undef Y_WIDTH   
#undef K_h       
#undef K_w       
#undef K_hstride 



















///////////////////////////////////////////////////
///     1 pixel; Padding on all sides
///////////////////////////////////////////////////
#define DEBUG_ON        TEST_DEBUG_ON && 0
#define CHANS_IN        (VPU_INT8_EPV + 4)
#define CHANS_OUT       (VPU_INT8_ACC_PERIOD)
#define X_HEIGHT        (1)
#define X_WIDTH         (1)
#define Y_HEIGHT        (1)
#define Y_WIDTH         (1)
#define K_h             (3)
#define K_w             (3)
#define K_hstride       (1)
void test_nn_conv2d_hstrip_deep_padded_case3()
{
    PRINTF("%s...\n", __func__);

    struct {
        int32_t bias[CHANS_OUT];
        int16_t shift1[CHANS_OUT];
        int16_t scale[CHANS_OUT];
        int16_t offset_scale[CHANS_OUT];
        int16_t offset[CHANS_OUT];
        int16_t shift2[CHANS_OUT];
    } BSO;

    nn_image_t WORD_ALIGNED  X[X_HEIGHT][X_WIDTH][CHANS_IN];
    nn_tensor_t WORD_ALIGNED  K[CHANS_OUT][K_h][K_w][CHANS_IN];
    nn_bso_block_t bso[BSO_BLOCK_COUNT(CHANS_OUT)];
    nn_image_t WORD_ALIGNED  Y[Y_HEIGHT][Y_WIDTH][CHANS_OUT];

    
    typedef struct {
        int8_t x;
        int8_t k;
        int8_t zero_point;
        int8_t expected;
        unsigned line;
    } test_case_t;

    test_case_t casses[] = {
        //  x,  k,  zero_point,     expected,   line
        {   0,  0,           0,            0,   __LINE__ },
        {   1,  0,           0,            0,   __LINE__ },
        {   0,  1,           0,            0,   __LINE__ },
        {   4,  4,           0,     CHANS_IN,   __LINE__ },
        {   8,  4,           0,   2*CHANS_IN,   __LINE__ },
        {   4,  8,           0,   2*CHANS_IN,   __LINE__ },
        {  -8,  4,           0,  -2*CHANS_IN,   __LINE__ },
        {   4, -8,           0,  -2*CHANS_IN,   __LINE__ },
        {   0,  0,           1,            0,   __LINE__ },
        {   4,  0,           1,            0,   __LINE__ },
        {   0,  4,           1,   2*CHANS_IN,   __LINE__ },
        {   4,  4,           1,   3*CHANS_IN,   __LINE__ },
        {  -4,  4,           2,   3*CHANS_IN,   __LINE__ },
        {  -8,  4,           2,   2*CHANS_IN,   __LINE__ },
        { -16,  4,           2,            0,   __LINE__ },
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

        int8_t zero_point_vec[VPU_INT8_EPV];
        memset(zero_point_vec, casse->zero_point, sizeof(zero_point_vec));

        memset(X, casse->x, x_params.height * x_params.width * x_params.channels * sizeof(int8_t));
        memset(K, casse->k, y_params.channels * K_h * K_w * x_params.channels * sizeof(int8_t));


        for(int k = 0; k < y_params.channels; k++){
            BSO.bias[k]     = 16*k;
            BSO.shift1[k]   = 4;
            BSO.scale[k]    = 8;
            BSO.offset_scale[k] = 0;
            BSO.offset[k]       = 0;
            BSO.shift2[k]   = 3;
        }

        nn_standard_BSO_layout(bso, (int32_t*) &BSO.bias, (int16_t*) &BSO.shift1, 
                                (int16_t*) &BSO.scale, (int16_t*) &BSO.offset_scale, (int16_t*) &BSO.offset, (int16_t*) &BSO.shift2, NULL, CHANS_OUT);

        // Start the convolution window from each of 9 different positions
        // results same for each position
        for(int pad_t = 1; pad_t < K_h; pad_t++){
            PRINTF("\t\t\tpad_t = %u\n", pad_t);
            for(int pad_l = 0; pad_l < K_w; pad_l++){
                PRINTF("\t\t\t\tpad_l = %d\n", pad_l);
                
                int pad_b = 2 - pad_t;
                int pad_r = 2 - pad_l;

                nn_image_t* X_patch_start = &X[-pad_t][-pad_l][0];

                memset(Y, 0xCC, sizeof(Y));
                nn_conv2d_hstrip_deep_padded((nn_image_t*) Y, X_patch_start, KERNEL_4D_COG_LAST_CHAN_START(K, 0), 
                                                (nn_bso_block_t*) &bso, K_h, K_w, K_hstride, x_params.channels, 
                                                pad_t, pad_b, pad_l, pad_r, (x_params.width-K_w)*x_params.channels, 
                                                -K_h*K_w*x_params.channels, y_params.channels, 1, zero_point_vec);

            
                PRINTF("\t\t\t\t\tChecking...\n");
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
    }
}
#undef DEBUG_ON  
#undef CHANS_IN  
#undef CHANS_OUT 
#undef X_HEIGHT  
#undef X_WIDTH   
#undef Y_HEIGHT  
#undef Y_WIDTH   
#undef K_h       
#undef K_w       
#undef K_hstride 















///////////////////////////////////////////////////
///     1x1 conv window; 3 output pixels
///////////////////////////////////////////////////
#define DEBUG_ON        TEST_DEBUG_ON && 0
#define CHANS_IN        (VPU_INT8_EPV + 4)
#define CHANS_OUT       (VPU_INT8_ACC_PERIOD)
#define X_HEIGHT        (1)
#define X_WIDTH         (3)
#define Y_HEIGHT        (1)
#define Y_WIDTH         (3)
#define K_h             (1)
#define K_w             (1)
#define K_hstride       (1)
#define ZERO_POINT      (0)
void test_nn_conv2d_hstrip_deep_padded_case4()
{
    PRINTF("%s...\n", __func__);

    struct {
        int32_t bias[CHANS_OUT];
        int16_t shift1[CHANS_OUT];
        int16_t scale[CHANS_OUT];
        int16_t offset_scale[CHANS_OUT];
        int16_t offset[CHANS_OUT];
        int16_t shift2[CHANS_OUT];
    } BSO;

    nn_image_t WORD_ALIGNED  X[X_HEIGHT][X_WIDTH][CHANS_IN];
    nn_tensor_t WORD_ALIGNED  K[CHANS_OUT][K_h][K_w][CHANS_IN];
    nn_bso_block_t bso[BSO_BLOCK_COUNT(CHANS_OUT)];
    nn_image_t WORD_ALIGNED  Y[Y_HEIGHT][Y_WIDTH][CHANS_OUT];

    int8_t zero_point_vec[VPU_INT8_EPV];
    memset(zero_point_vec, ZERO_POINT, sizeof(zero_point_vec));
    
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


        memset(X, casse->x, x_params.height * x_params.width * x_params.channels * sizeof(int8_t));
        memset(K, casse->k, y_params.channels * K_h * K_w * x_params.channels * sizeof(int8_t));


        for(int k = 0; k < y_params.channels; k++){
            BSO.bias[k]     =-k;
            BSO.shift1[k]   = 0;
            BSO.scale[k]    = 1;
            BSO.offset_scale[k] = 0;
            BSO.offset[k]       = 0;
            BSO.shift2[k]   = 0;
        }

        nn_standard_BSO_layout(bso, (int32_t*) &BSO.bias, (int16_t*) &BSO.shift1, 
                                (int16_t*) &BSO.scale, (int16_t*) &BSO.offset_scale, (int16_t*) &BSO.offset, (int16_t*) &BSO.shift2, NULL, CHANS_OUT);

        
        int pad_t = 0;
        int pad_b = 0;
        int pad_l = 0;
        int pad_r = -2;

        nn_image_t* X_patch_start = &X[-pad_t][-pad_l][0];

        memset(Y, 0xCC, sizeof(Y));
        nn_conv2d_hstrip_deep_padded((nn_image_t*) Y, X_patch_start, KERNEL_4D_COG_LAST_CHAN_START(K, 0), 
                                        (nn_bso_block_t*) &bso, K_h, K_w, K_hstride, x_params.channels, 
                                        pad_t, pad_b, pad_l, pad_r, (x_params.width-K_w)*x_params.channels, 
                                        -K_h*K_w*x_params.channels, y_params.channels, Y_WIDTH, zero_point_vec);

    
        PRINTF("\t\t\tChecking...\n");
        for(unsigned row = 0; row < y_params.height; row++){
            for(unsigned col = 0; col < y_params.width; col++){
                for(unsigned chn = 0; chn < y_params.channels; chn++){
                    
                    int8_t y_exp = casse->expected - chn;

                    check_Y(y_exp, row, col, chn, casse->line, (nn_image_t*) Y, &y_params);
                }
            }
        }
    }
}
#undef DEBUG_ON  
#undef CHANS_IN  
#undef CHANS_OUT 
#undef X_HEIGHT  
#undef X_WIDTH   
#undef Y_HEIGHT  
#undef Y_WIDTH   
#undef K_h       
#undef K_w       
#undef K_hstride 
#undef ZERO_POINT










///////////////////////////////////////////////////
///     3x3 conv window; 3x3 input image; 1x3 output image
///////////////////////////////////////////////////
#define DEBUG_ON        TEST_DEBUG_ON && 0
#define CHANS_IN        (VPU_INT8_EPV + 4)
#define CHANS_OUT       (VPU_INT8_ACC_PERIOD)
#define X_HEIGHT        (3)
#define X_WIDTH         (3)
#define Y_HEIGHT        (1)
#define Y_WIDTH         (3)
#define K_h             (3)
#define K_w             (3)
#define K_hstride       (1)
void test_nn_conv2d_hstrip_deep_padded_case5()
{
    PRINTF("%s...\n", __func__);

    struct {
        int32_t bias[CHANS_OUT];
        int16_t shift1[CHANS_OUT];
        int16_t scale[CHANS_OUT];
        int16_t offset_scale[CHANS_OUT];
        int16_t offset[CHANS_OUT];
        int16_t shift2[CHANS_OUT];
    } BSO;

    nn_image_t WORD_ALIGNED  X[X_HEIGHT][X_WIDTH][CHANS_IN];
    nn_tensor_t WORD_ALIGNED  K[CHANS_OUT][K_h][K_w][CHANS_IN];
    nn_bso_block_t bso[BSO_BLOCK_COUNT(CHANS_OUT)];
    nn_image_t WORD_ALIGNED  Y[Y_HEIGHT][Y_WIDTH][CHANS_OUT];
    
    typedef struct {
        int8_t x;
        int8_t k;
        int8_t zero_point;
        int16_t shift1;
        unsigned line;
    } test_case_t;

    test_case_t casses[] = {
        //  x,  k,  zero_point,    shift1,     line
        {   0,  0,           0,         6,     __LINE__ },
        {   4,  0,           0,         6,     __LINE__ },
        {   0,  4,           0,         6,     __LINE__ },
        {   4,  4,           0,         6,     __LINE__ },
        {   8,  4,           0,         7,     __LINE__ },
        {   4,  8,           0,         7,     __LINE__ },
        {  -8,  4,           0,         7,     __LINE__ },
        {   4, -8,           0,         7,     __LINE__ },
        {   0,  0,           4,         0,     __LINE__ },
        {   4,  0,           4,         0,     __LINE__ },
        {   0,  4,           4,         6,     __LINE__ },
        {   4,  4,           4,         6,     __LINE__ },
        {  -4,  4,           8,         6,     __LINE__ },
        {  -8,  4,           8,         8,     __LINE__ },
        { -16,  4,           8,         9,     __LINE__ },
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

        int8_t zero_point_vec[VPU_INT8_EPV];
        memset(zero_point_vec, casse->zero_point, sizeof(zero_point_vec));


        memset(X, casse->x, x_params.height * x_params.width * x_params.channels * sizeof(int8_t));
        memset(K, casse->k, y_params.channels * K_h * K_w * x_params.channels * sizeof(int8_t));


        for(int k = 0; k < y_params.channels; k++){
            BSO.bias[k]     = 0;
            BSO.shift1[k]   = casse->shift1;
            BSO.scale[k]    = 8;
            BSO.offset_scale[k] = 0;
            BSO.offset[k]       = 0;
            BSO.shift2[k]   = 3;
        }

        nn_standard_BSO_layout(bso, (int32_t*) &BSO.bias, (int16_t*) &BSO.shift1, 
                                (int16_t*) &BSO.scale, (int16_t*) &BSO.offset_scale, (int16_t*) &BSO.offset, (int16_t*) &BSO.shift2, NULL, CHANS_OUT);

        
        int pad_t = 0;
        int pad_b = 0;
        int pad_l = 0;
        int pad_r = 0;

        nn_image_t* X_patch_start = &X[-pad_t][-pad_l][0];

        memset(Y, 0xCC, sizeof(Y));
        nn_conv2d_hstrip_deep_padded((nn_image_t*) Y, X_patch_start, KERNEL_4D_COG_LAST_CHAN_START(K, 0), 
                                        (nn_bso_block_t*) &bso, K_h, K_w, K_hstride, x_params.channels, 
                                        pad_t, pad_b, pad_l, pad_r, (x_params.width-K_w)*x_params.channels, 
                                        -K_h*K_w*x_params.channels, y_params.channels, Y_WIDTH, zero_point_vec);


        nn_image_t Y_exp[Y_HEIGHT][Y_WIDTH];

        Y_exp[0][0] = ((CHANS_IN * casse->k * ( 9 * ((int32_t)casse->x) + 0 * ((int32_t)casse->zero_point))) 
                        + (1<<(casse->shift1-1)) ) >> casse->shift1;
        Y_exp[0][1] = ((CHANS_IN * casse->k * ( 6 * ((int32_t)casse->x) + 3 * ((int32_t)casse->zero_point))) 
                        + (1<<(casse->shift1-1)) ) >> casse->shift1;
        Y_exp[0][2] = ((CHANS_IN * casse->k * ( 3 * ((int32_t)casse->x) + 6 * ((int32_t)casse->zero_point))) 
                        + (1<<(casse->shift1-1)) ) >> casse->shift1;
    
        PRINTF("\t\t\tChecking...\n");
        for(unsigned row = 0; row < y_params.height; row++){
            for(unsigned col = 0; col < y_params.width; col++){
                for(unsigned chn = 0; chn < y_params.channels; chn++){
                    
                    int8_t y_exp = Y_exp[row][col];

                    check_Y(y_exp, row, col, chn, casse->line, (nn_image_t*) Y, &y_params);
                }
            }
        }
    }
}
#undef DEBUG_ON  
#undef CHANS_IN  
#undef CHANS_OUT 
#undef X_HEIGHT  
#undef X_WIDTH   
#undef Y_HEIGHT  
#undef Y_WIDTH   
#undef K_h       
#undef K_w       
#undef K_hstride 










#define CHANS_IN        (VPU_INT8_EPV)
#define CHANS_OUT       (VPU_INT8_ACC_PERIOD)
#define X_HEIGHT        (1)
#define X_WIDTH         (1)
#define Y_HEIGHT        (1)
#define Y_WIDTH         (1)
#define K_h             (1)
#define K_w             (1)
#define K_hstride       (1)
void test_nn_conv2d_hstrip_deep_padded_case6()
{
    PRINTF("%s...\n", __func__);

    struct {
        int32_t bias[CHANS_OUT];
        int16_t shift1[CHANS_OUT];
        int16_t scale[CHANS_OUT];
        int16_t offset_scale[CHANS_OUT];
        int16_t offset[CHANS_OUT];
        int16_t shift2[CHANS_OUT];
    } BSO;

    nn_image_t WORD_ALIGNED  X[X_HEIGHT][X_WIDTH][CHANS_IN];
    nn_tensor_t WORD_ALIGNED  K[CHANS_OUT][K_h][K_w][CHANS_IN];
    nn_bso_block_t bso[BSO_BLOCK_COUNT(CHANS_OUT)];
    nn_image_t WORD_ALIGNED  Y[Y_HEIGHT][Y_WIDTH][CHANS_OUT];

    nn_image_params_t x_params = { X_HEIGHT, X_WIDTH, CHANS_IN };
    nn_image_params_t y_params = { Y_HEIGHT, Y_WIDTH, CHANS_OUT };

    int8_t zero_point_vec[VPU_INT8_EPV];
    memset(zero_point_vec, 0, sizeof(zero_point_vec));

    memset(X, 0, x_params.height * x_params.width * x_params.channels * sizeof(int8_t));
    memset(K, 0, y_params.channels * K_h * K_w * x_params.channels * sizeof(int8_t));

    for(int k = 0; k < y_params.channels; k++){
        BSO.bias[k]     = ((k % 2) == 0)? -128 : -127;
        BSO.shift1[k]   = 0;
        BSO.scale[k]    = 1;
        BSO.offset_scale[k] = 0;
        BSO.offset[k]       = 0;
        BSO.shift2[k]   = 0;
    }

    nn_standard_BSO_layout(bso, (int32_t*) &BSO.bias, (int16_t*) &BSO.shift1, 
                            (int16_t*) &BSO.scale, (int16_t*) &BSO.offset_scale, (int16_t*) &BSO.offset, (int16_t*) &BSO.shift2, NULL, CHANS_OUT);
    
    int pad_t = 0;
    int pad_b = 0;
    int pad_l = 0;
    int pad_r = 0;

    nn_image_t* X_patch_start = &X[-pad_t][-pad_l][0];

    memset(Y, 0xCC, sizeof(Y));
    nn_conv2d_hstrip_deep_padded((nn_image_t*) Y, X_patch_start, KERNEL_4D_COG_LAST_CHAN_START(K, 0), 
                                    (nn_bso_block_t*) &bso, K_h, K_w, K_hstride, x_params.channels, 
                                    pad_t, pad_b, pad_l, pad_r, (x_params.width-K_w)*x_params.channels, 
                                    -K_h*K_w*x_params.channels, y_params.channels, Y_WIDTH, zero_point_vec);

    PRINTF("\t\t\tChecking...\n");
    for(unsigned row = 0; row < y_params.height; row++){
        for(unsigned col = 0; col < y_params.width; col++){
            for(unsigned chn = 0; chn < y_params.channels; chn++){
                
                int8_t y_exp = ((chn % 2) == 0)? NEG_SAT_VAL : -127;

                TEST_ASSERT_EQUAL(y_exp, Y[row][col][chn]);
            }
        }
    }
}
#undef CHANS_IN  
#undef CHANS_OUT 
#undef X_HEIGHT  
#undef X_WIDTH   
#undef Y_HEIGHT  
#undef Y_WIDTH   
#undef K_h       
#undef K_w       
#undef K_hstride 



void test_nn_conv2d_hstrip_deep_padded()
{
    UNITY_SET_FILE();

    RUN_TEST(test_nn_conv2d_hstrip_deep_padded_case0);
    RUN_TEST(test_nn_conv2d_hstrip_deep_padded_case1);
    RUN_TEST(test_nn_conv2d_hstrip_deep_padded_case2);
    RUN_TEST(test_nn_conv2d_hstrip_deep_padded_case3);
    RUN_TEST(test_nn_conv2d_hstrip_deep_padded_case4);
    RUN_TEST(test_nn_conv2d_hstrip_deep_padded_case5);
    RUN_TEST(test_nn_conv2d_hstrip_deep_padded_case6);
}