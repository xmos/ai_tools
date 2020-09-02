
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
void test_nn_conv2d_hstrip_deep_case0()
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
        nn_conv2d_hstrip_deep((nn_image_t*) Y, (nn_image_t*) X, KERNEL_4D_COG_LAST_CHAN_START(K, 0), 
                                        (nn_bso_block_t*) &bso, K_h, K_w, K_hstride, x_params.channels,
                                        (x_params.width-K_w)*x_params.channels, -K_h*K_w*x_params.channels,
                                        y_params.channels, 1);

    
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












#define DEBUG_ON        TEST_DEBUG_ON && 0
#define CHANS_IN        (VPU_INT8_EPV + 4)
#define CHANS_OUT       (VPU_INT8_ACC_PERIOD)
#define X_HEIGHT        (3)
#define X_WIDTH         (8)
#define Y_HEIGHT        (1)
#define Y_WIDTH         (4)
#define K_h             (3)
#define K_w             (2)
#define K_hstride       (2)
void test_nn_conv2d_hstrip_deep_case1()
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


    const int8_t x_vals[] = {1,3,5,7};
    const int8_t k_vals[] = {2,4,10};

    const int8_t y_exp[] = {18,54,90,126}; // minus the channel index

    for(int i = 0; i < x_params.height; i++){
        for(int j = 0; j < x_params.width; j++){
            for(int k = 0; k < x_params.channels; k++){
                X[i][j][k] = x_vals[j/K_w];
            }
        }
    }    
    
    for(int h = 0; h < y_params.channels; h++){
        for(int i = 0; i < K_h; i++){
            for(int j = 0; j < K_w; j++){
                for(int k = 0; k < x_params.channels; k++){
                    K[h][i][j][k] = k_vals[i];
                }
            }
        }
    }

    for(int k = 0; k < y_params.channels; k++){
        BSO.shift1[k]   = 6;
        BSO.bias[k]     = -(1<<BSO.shift1[k]) * k;
        BSO.scale[k]    = 64;
        BSO.offset_scale[k] = 0;
        BSO.offset[k]       = 0;
        BSO.shift2[k]   = 6;
    }

    nn_standard_BSO_layout(bso, (int32_t*) &BSO.bias, (int16_t*) &BSO.shift1, 
                            (int16_t*) &BSO.scale, (int16_t*) &BSO.offset_scale, (int16_t*) &BSO.offset, (int16_t*) &BSO.shift2, NULL, CHANS_OUT);

    memset(Y, 0xCC, sizeof(Y));
    nn_conv2d_hstrip_deep((nn_image_t*) Y, (nn_image_t*) X, KERNEL_4D_COG_LAST_CHAN_START(K, 0), 
                                    (nn_bso_block_t*) &bso, K_h, K_w, K_hstride, x_params.channels,
                                    (x_params.width-K_w)*x_params.channels, -K_h*K_w*x_params.channels,
                                    y_params.channels, Y_WIDTH);


    PRINTF("\t\t\tChecking...\n");
    for(unsigned row = 0; row < y_params.height; row++){
        for(unsigned col = 0; col < y_params.width; col++){
            for(unsigned chn = 0; chn < y_params.channels; chn++){
                
                int8_t y_expected = y_exp[col] - chn;

                check_Y(y_expected, row, col, chn, __LINE__, (nn_image_t*) Y, &y_params);
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
void test_nn_conv2d_hstrip_deep_case2()
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
            BSO.bias[k]     = 0;
            BSO.shift1[k]   = 0;
            BSO.scale[k]    = 1;
            BSO.offset_scale[k] = k;
            BSO.offset[k]       = 1;
            BSO.shift2[k]   = 0;
        }

        nn_standard_BSO_layout(bso, (int32_t*) &BSO.bias, (int16_t*) &BSO.shift1, 
                                (int16_t*) &BSO.scale, (int16_t*) &BSO.offset_scale, (int16_t*) &BSO.offset, (int16_t*) &BSO.shift2, NULL, CHANS_OUT);

        memset(Y, 0xCC, sizeof(Y));
        nn_conv2d_hstrip_deep((nn_image_t*) Y, (nn_image_t*) X, KERNEL_4D_COG_LAST_CHAN_START(K, 0), 
                                        (nn_bso_block_t*) &bso, K_h, K_w, K_hstride, x_params.channels,
                                        (x_params.width-K_w)*x_params.channels, -K_h*K_w*x_params.channels,
                                        y_params.channels, 1);

    
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












#define CHANS_IN        (VPU_INT8_EPV)
#define CHANS_OUT       (VPU_INT8_ACC_PERIOD)
#define X_HEIGHT        (1)
#define X_WIDTH         (1)
#define Y_HEIGHT        (1)
#define Y_WIDTH         (1)
#define K_h             (1)
#define K_w             (1)
#define K_hstride       (1)
void test_nn_conv2d_hstrip_deep_case3()
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

    memset(X, 0, x_params.height * x_params.width * x_params.channels * sizeof(int8_t));
    memset(K, 0, y_params.channels * K_h * K_w * x_params.channels * sizeof(int8_t));

    for(int k = 0; k < y_params.channels; k++){
        BSO.bias[k]     = ((k % 2) == 0)? NEG_SAT_VAL : -127;
        BSO.shift1[k]   = 0;
        BSO.scale[k]    = 1;
        BSO.offset_scale[k] = 0;
        BSO.offset[k]       = 0;
        BSO.shift2[k]   = 0;
    }

    nn_standard_BSO_layout(bso, (int32_t*) &BSO.bias, (int16_t*) &BSO.shift1, 
                            (int16_t*) &BSO.scale, (int16_t*) &BSO.offset_scale, 
                            (int16_t*) &BSO.offset, (int16_t*) &BSO.shift2, NULL, CHANS_OUT);

    memset(Y, 0xCC, sizeof(Y));
    nn_conv2d_hstrip_deep((nn_image_t*) Y, (nn_image_t*) X, KERNEL_4D_COG_LAST_CHAN_START(K, 0), 
                                    (nn_bso_block_t*) &bso, K_h, K_w, K_hstride, x_params.channels,
                                    (x_params.width-K_w)*x_params.channels, -K_h*K_w*x_params.channels,
                                    y_params.channels, 1);


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




void test_nn_conv2d_hstrip_deep()
{
    UNITY_SET_FILE();

    RUN_TEST(test_nn_conv2d_hstrip_deep_case0);
    RUN_TEST(test_nn_conv2d_hstrip_deep_case1);
    RUN_TEST(test_nn_conv2d_hstrip_deep_case2);
    RUN_TEST(test_nn_conv2d_hstrip_deep_case3);
}