
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>


#include "tst_common.h"

#include "nn_operator.h"
#include "nn_types.h"
#include "xs3_vpu.h"

// #include "dsp_xs3_vector.h"
#include "unity.h"


#define DO_PRINT_EXTRA ((DO_PRINT_EXTRA_GLOBAL) && 0)




static void check_Y(
    const nn_image_t y_exp, 
    const nn_image_t* Y,
    const nn_image_params_t* y_params,
    const unsigned row,
    const unsigned col,
    const unsigned chn,
    const unsigned line)
{
    char str_buff[200];

    unsigned y_offset = IMG_ADDRESS_VECT(y_params, row, col, chn);

    int8_t y = Y[y_offset];
    // printf( "Y[%d][%d][%d] = %d\n",row,col,chn,y );

    //Only sprintf-ing if the test will fail saves a ton of time.
    if(y != y_exp)
        sprintf(str_buff, "(row, col, chn) = (%u, %u, %u)  [test vector @ line %u]", row, col, chn, line);

    TEST_ASSERT_EQUAL_MESSAGE(y_exp, y, str_buff);
}


#define K_W_ARRAY   (VPU_INT8_EPV / CHANS_IN)




///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
#define CHANS_IN        ( 4 )
#define CHANS_OUT       ( 4 )
#define K_H             ( 1 )
#define K_W             ( 1 )
#define X_HEIGHT        ( 1 )
#define X_WIDTH         ( 1 )
#define Y_HEIGHT        ( 1 )
#define Y_WIDTH         ( 1 )
#define ZERO_POINT      ( -128 )
void test_conv2d_im2col_case0()
{

    nn_tensor_t WORD_ALIGNED K[CHANS_OUT][ (((K_H*K_W*CHANS_IN+3)>>2)<<2) ];
    nn_image_t  WORD_ALIGNED COL[((K_H*K_W*CHANS_IN+VPU_INT8_EPV-1)>>VPU_INT8_EPV_LOG2)<<VPU_INT8_EPV_LOG2];
    nn_image_t  WORD_ALIGNED X[X_HEIGHT][X_WIDTH][CHANS_IN];
    nn_image_t  WORD_ALIGNED Y[Y_HEIGHT][Y_WIDTH][CHANS_OUT];
    
    struct {
        int32_t bias[CHANS_OUT];
        int16_t shift1[CHANS_OUT];
        int16_t scale[CHANS_OUT];
        int16_t offset_scale[CHANS_OUT];
        int16_t offset[CHANS_OUT];
        int16_t shift2[CHANS_OUT];
    } BSO;

    nn_bso_block_t bso[BSO_BLOCK_COUNT(CHANS_OUT)];

    PRINTF("%s...\n", __func__);

    nn_conv2d_im2col_plan_t plan;
    nn_conv2d_im2col_job_t job;

    nn_conv2d_window_params_t conv2d_window = { { K_H, K_W }, { 0, 0 }, { 1, 1 } }; 

    nn_image_params_t x_params = { X_HEIGHT, X_WIDTH, CHANS_IN  };
    nn_image_params_t y_params = { Y_HEIGHT, Y_WIDTH, CHANS_OUT };
    
    nn_conv2d_job_params_t job_params = {  {  0,  0,  0}, {  Y_HEIGHT, Y_WIDTH, CHANS_OUT}  };
    conv2d_im2col_init(&plan, &job, &x_params, &y_params, &job_params, &conv2d_window, ZERO_POINT, 1);

    memset(X, 0, x_params.height * x_params.width * x_params.channels);

    memset(K, 0, sizeof(K));
    for(channel_count_t cout = 0; cout < CHANS_OUT; cout++)
        for(int row = 0; row < K_H; row++)
            for(int col = 0; col < K_W; col++)
                for(int cin = 0; cin < x_params.channels; cin++)
                    K[cout][row*K_W*CHANS_IN + col*CHANS_IN + cin] = 0;



    for(int k = 0; k < CHANS_OUT; k++){
        BSO.bias[k] = 0;
        BSO.shift1[k] = 0;
        BSO.scale[k] = 1;
        BSO.offset_scale[k] = 0;
        BSO.offset[k]       = 0;
        BSO.shift2[k] = 0;
    }
    nn_standard_BSO_layout(bso, (int32_t*) &BSO.bias, (int16_t*) &BSO.shift1, 
                        (int16_t*) &BSO.scale, (int16_t*) &BSO.offset_scale, (int16_t*) &BSO.offset, (int16_t*) &BSO.shift2, NULL, y_params.channels);

    memset(Y, 0xCC, y_params.height * y_params.width * y_params.channels);

    conv2d_im2col((nn_image_t*) Y, (nn_image_t*) X, (nn_image_t*) COL, (nn_tensor_t*) K, bso, &plan, &job);

    for(int row = 0; row < y_params.height; row++){
        for(int col = 0; col < y_params.width; col++){
            for(int chn = 0; chn < y_params.channels; chn++){
                int8_t y_exp = 0;
                check_Y(y_exp, (nn_image_t*) Y, &y_params, row, col, chn, __LINE__);
            }
        }
    }

}
#undef CHANS_IN
#undef CHANS_OUT
#undef K_H
#undef K_W
#undef X_HEIGHT
#undef X_WIDTH
#undef Y_HEIGHT
#undef Y_WIDTH
#undef ZERO_POINT





///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
#define CHANS_IN        ( 4 )
#define CHANS_OUT       ( 4 )
#define K_H             ( 1 )
#define K_W             ( 1 )
#define X_HEIGHT        ( 1 )
#define X_WIDTH         ( 1 )
#define Y_HEIGHT        ( 1 )
#define Y_WIDTH         ( 1 )
#define ZERO_POINT      ( -128 )
void test_conv2d_im2col_case1()
{

    nn_tensor_t WORD_ALIGNED K[CHANS_OUT][ (((K_H*K_W*CHANS_IN+3)>>2)<<2) ];
    nn_image_t  WORD_ALIGNED COL[((K_H*K_W*CHANS_IN+VPU_INT8_EPV-1)>>VPU_INT8_EPV_LOG2)<<VPU_INT8_EPV_LOG2];
    nn_image_t  WORD_ALIGNED X[X_HEIGHT][X_WIDTH][CHANS_IN];
    nn_image_t  WORD_ALIGNED Y[Y_HEIGHT][Y_WIDTH][CHANS_OUT];
    
    struct {
        int32_t bias[CHANS_OUT];
        int16_t shift1[CHANS_OUT];
        int16_t scale[CHANS_OUT];
        int16_t offset_scale[CHANS_OUT];
        int16_t offset[CHANS_OUT];
        int16_t shift2[CHANS_OUT];
    } BSO;

    nn_bso_block_t bso[BSO_BLOCK_COUNT(CHANS_OUT)];

    PRINTF("%s...\n", __func__);

    nn_conv2d_im2col_plan_t plan;
    nn_conv2d_im2col_job_t job;

    nn_conv2d_window_params_t conv2d_window = { { K_H, K_W }, { 0, 0 }, { 1, 1 } }; 

    nn_image_params_t x_params = { X_HEIGHT, X_WIDTH, CHANS_IN };
    nn_image_params_t y_params = { Y_HEIGHT, Y_WIDTH, CHANS_OUT };

    nn_conv2d_job_params_t job_params = {  {  0,  0,  0}, {  Y_HEIGHT, Y_WIDTH, CHANS_OUT}  };
    conv2d_im2col_init(&plan, &job, &x_params, &y_params, &job_params, &conv2d_window, ZERO_POINT, 1);

    memset(X, 0, x_params.height * x_params.width * x_params.channels);
    memset(K, 0, sizeof(K));

    for(int bias = -10; bias < 10; bias++){

        PRINTF("\tbias_mult = %d...\n", bias);

        for(int k = 0; k < CHANS_OUT; k++){
            BSO.bias[k] = bias * k;
            BSO.shift1[k] = 0;
            BSO.scale[k] = 1;
            BSO.offset_scale[k] = 0;
            BSO.offset[k]       = 0;
            BSO.shift2[k] = 0;
        }
        nn_standard_BSO_layout(bso, (int32_t*) &BSO.bias, (int16_t*) &BSO.shift1, 
                            (int16_t*) &BSO.scale, (int16_t*) &BSO.offset_scale, (int16_t*) &BSO.offset, (int16_t*) &BSO.shift2, NULL, y_params.channels);

        memset(Y, 0xCC, y_params.height * y_params.width * y_params.channels);
        conv2d_im2col((nn_image_t*) Y, (nn_image_t*) X, (nn_image_t*) COL, (nn_tensor_t*) K, bso, &plan, &job);

        for(int row = 0; row < y_params.height; row++){
            for(int col = 0; col < y_params.width; col++){
                for(int chn = 0; chn < y_params.channels; chn++){
                    int8_t y_exp = bias * chn;
                    check_Y(y_exp, (nn_image_t*) Y, &y_params, row, col, chn, __LINE__);
                }
            }
        }
    }
}
#undef CHANS_IN
#undef CHANS_OUT
#undef K_H
#undef K_W
#undef X_HEIGHT
#undef X_WIDTH
#undef Y_HEIGHT
#undef Y_WIDTH
#undef ZERO_POINT





///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
#define CHANS_IN        ( 4 )
#define CHANS_OUT       ( 4 )
#define K_H             ( 1 )
#define K_W             ( 1 )
#define X_HEIGHT        ( 1 )
#define X_WIDTH         ( 1 )
#define Y_HEIGHT        ( 1 )
#define Y_WIDTH         ( 1 )
#define ZERO_POINT      ( -128 )
void test_conv2d_im2col_case2()
{

    nn_tensor_t WORD_ALIGNED K[CHANS_OUT][ (((K_H*K_W*CHANS_IN+3)>>2)<<2) ];
    nn_image_t  WORD_ALIGNED COL[((K_H*K_W*CHANS_IN+VPU_INT8_EPV-1)>>VPU_INT8_EPV_LOG2)<<VPU_INT8_EPV_LOG2];
    nn_image_t  WORD_ALIGNED X[X_HEIGHT][X_WIDTH][CHANS_IN];
    nn_image_t  WORD_ALIGNED Y[Y_HEIGHT][Y_WIDTH][CHANS_OUT];
    
    struct {
        int32_t bias[CHANS_OUT];
        int16_t shift1[CHANS_OUT];
        int16_t scale[CHANS_OUT];
        int16_t offset_scale[CHANS_OUT];
        int16_t offset[CHANS_OUT];
        int16_t shift2[CHANS_OUT];
    } BSO;

    nn_bso_block_t bso[BSO_BLOCK_COUNT(CHANS_OUT)];

    PRINTF("%s...\n", __func__);

    nn_conv2d_im2col_plan_t plan;
    nn_conv2d_im2col_job_t job;

    nn_conv2d_window_params_t conv2d_window = { { K_H, K_W }, { 0, 0 }, { 1, 1 } }; 

    nn_image_params_t x_params = { X_HEIGHT, X_WIDTH, CHANS_IN };
    nn_image_params_t y_params = { Y_HEIGHT, Y_WIDTH, CHANS_OUT };

    nn_conv2d_job_params_t job_params = {  {  0,  0,  0}, {  Y_HEIGHT, Y_WIDTH, CHANS_OUT}  };
    conv2d_im2col_init(&plan, &job, &x_params, &y_params, &job_params, &conv2d_window, ZERO_POINT, 1);

    memset(X, 0, x_params.height * x_params.width * x_params.channels);
    memset(K, 0, sizeof(K));
        
    for(int cout = 0; cout < y_params.channels; cout++)
        for(int row = 0; row < conv2d_window.shape.height; row++)
            for(int col = 0; col < conv2d_window.shape.width; col++)
                for(int cin = 0; cin < x_params.channels; cin++)
                    K[cout][row*K_W*CHANS_IN + col*CHANS_IN + cin] = cout*(((K_H*K_W*CHANS_IN+3)>>2)<<2)+ row*K_W*CHANS_IN + col*CHANS_IN + cin;

    for(int shift1 = 0; shift1 < 4; shift1++){

        PRINTF("\tshift1 = %d...\n", shift1);

        for(int k = 0; k < CHANS_OUT; k++){
            BSO.bias[k] = 16 * k;
            BSO.shift1[k] = shift1;
            BSO.scale[k] = 1;
            BSO.offset_scale[k] = 0;
            BSO.offset[k]       = 0;
            BSO.shift2[k] = 0;
        }
        nn_standard_BSO_layout(bso, (int32_t*) &BSO.bias, (int16_t*) &BSO.shift1, 
                            (int16_t*) &BSO.scale, (int16_t*) &BSO.offset_scale, (int16_t*) &BSO.offset, (int16_t*) &BSO.shift2, NULL, y_params.channels);

        memset(Y, 0xCC, y_params.height * y_params.width * y_params.channels);

        conv2d_im2col((nn_image_t*) Y, (nn_image_t*) X, (nn_image_t*) COL, (nn_tensor_t*) K, bso, &plan, &job);

        for(int row = 0; row < y_params.height; row++){
            for(int col = 0; col < y_params.width; col++){
                for(int chn = 0; chn < y_params.channels; chn++){
                    int8_t y_exp = (16 * chn) >> shift1;
                    check_Y(y_exp, (nn_image_t*) Y, &y_params, row, col, chn, __LINE__);
                }
            }
        }
    }

}
#undef CHANS_IN
#undef CHANS_OUT
#undef K_H
#undef K_W
#undef X_HEIGHT
#undef X_WIDTH
#undef Y_HEIGHT
#undef Y_WIDTH
#undef ZERO_POINT





///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
#define CHANS_IN        ( 4 )
#define CHANS_OUT       ( 4 )
#define K_H             ( 1 )
#define K_W             ( 1 )
#define X_HEIGHT        ( 1 )
#define X_WIDTH         ( 1 )
#define Y_HEIGHT        ( 1 )
#define Y_WIDTH         ( 1 )
#define ZERO_POINT      ( -128 )
void test_conv2d_im2col_case3()
{

    nn_tensor_t WORD_ALIGNED K[CHANS_OUT][ (((K_H*K_W*CHANS_IN+3)>>2)<<2) ];
    nn_image_t  WORD_ALIGNED COL[((K_H*K_W*CHANS_IN+VPU_INT8_EPV-1)>>VPU_INT8_EPV_LOG2)<<VPU_INT8_EPV_LOG2];
    nn_image_t  WORD_ALIGNED X[X_HEIGHT][X_WIDTH][CHANS_IN];
    nn_image_t  WORD_ALIGNED Y[Y_HEIGHT][Y_WIDTH][CHANS_OUT];
    
    struct {
        int32_t bias[CHANS_OUT];
        int16_t shift1[CHANS_OUT];
        int16_t scale[CHANS_OUT];
        int16_t offset_scale[CHANS_OUT];
        int16_t offset[CHANS_OUT];
        int16_t shift2[CHANS_OUT];
    } BSO;

    nn_bso_block_t bso[BSO_BLOCK_COUNT(CHANS_OUT)];

    PRINTF("%s...\n", __func__);

    nn_conv2d_im2col_plan_t plan;
    nn_conv2d_im2col_job_t job;

    nn_conv2d_window_params_t conv2d_window = { { K_H, K_W }, { 0, 0 }, { 1, 1 } }; 

    nn_image_params_t x_params = { X_HEIGHT, X_WIDTH, CHANS_IN };
    nn_image_params_t y_params = { Y_HEIGHT, Y_WIDTH, CHANS_OUT };

    nn_conv2d_job_params_t job_params = {  {  0,  0,  0}, {  Y_HEIGHT, Y_WIDTH, CHANS_OUT}  };
    conv2d_im2col_init(&plan, &job, &x_params, &y_params, &job_params, &conv2d_window, ZERO_POINT, 1);

    memset(X, 0, x_params.height * x_params.width * x_params.channels);
    memset(K, 0, sizeof(K));

    for(int scale = -10; scale < 10; scale++){

        PRINTF("\tscale = %d...\n", scale);

        for(int k = 0; k < CHANS_OUT; k++){
            BSO.bias[k] = k;
            BSO.shift1[k] = 0;
            BSO.scale[k] = scale;
            BSO.offset_scale[k] = 0;
            BSO.offset[k]       = 0;
            BSO.shift2[k] = 0;
        }
        nn_standard_BSO_layout(bso, (int32_t*) &BSO.bias, (int16_t*) &BSO.shift1, 
                            (int16_t*) &BSO.scale, (int16_t*) &BSO.offset_scale, (int16_t*) &BSO.offset, (int16_t*) &BSO.shift2, NULL, y_params.channels);

        memset(Y, 0xCC, y_params.height * y_params.width * y_params.channels);

        conv2d_im2col((nn_image_t*) Y, (nn_image_t*) X, (nn_image_t*) COL, (nn_tensor_t*) K, bso, &plan, &job);

        for(int row = 0; row < y_params.height; row++){
            for(int col = 0; col < y_params.width; col++){
                for(int chn = 0; chn < y_params.channels; chn++){
                    int8_t y_exp = scale * chn;
                    check_Y(y_exp, (nn_image_t*) Y, &y_params, row, col, chn, __LINE__);
                }
            }
        }
    }

}
#undef CHANS_IN
#undef CHANS_OUT
#undef K_H
#undef K_W
#undef X_HEIGHT
#undef X_WIDTH
#undef Y_HEIGHT
#undef Y_WIDTH
#undef ZERO_POINT





///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
#define CHANS_IN        ( 4 )
#define CHANS_OUT       ( 4 )
#define K_H             ( 1 )
#define K_W             ( 1 )
#define X_HEIGHT        ( 1 )
#define X_WIDTH         ( 1 )
#define Y_HEIGHT        ( 1 )
#define Y_WIDTH         ( 1 )
#define ZERO_POINT      ( -128 )
void test_conv2d_im2col_case4()
{

    nn_tensor_t WORD_ALIGNED K[CHANS_OUT][ (((K_H*K_W*CHANS_IN+3)>>2)<<2) ];
    nn_image_t  WORD_ALIGNED COL[((K_H*K_W*CHANS_IN+VPU_INT8_EPV-1)>>VPU_INT8_EPV_LOG2)<<VPU_INT8_EPV_LOG2];
    nn_image_t  WORD_ALIGNED X[X_HEIGHT][X_WIDTH][CHANS_IN];
    nn_image_t  WORD_ALIGNED Y[Y_HEIGHT][Y_WIDTH][CHANS_OUT];
    
    struct {
        int32_t bias[CHANS_OUT];
        int16_t shift1[CHANS_OUT];
        int16_t scale[CHANS_OUT];
        int16_t offset_scale[CHANS_OUT];
        int16_t offset[CHANS_OUT];
        int16_t shift2[CHANS_OUT];
    } BSO;

    nn_bso_block_t bso[BSO_BLOCK_COUNT(CHANS_OUT)];

    PRINTF("%s...\n", __func__);

    nn_conv2d_im2col_plan_t plan;
    nn_conv2d_im2col_job_t job;

    nn_conv2d_window_params_t conv2d_window = { { K_H, K_W }, { 0, 0 }, { 1, 1 } }; 

    nn_image_params_t x_params = { X_HEIGHT, X_WIDTH, CHANS_IN };
    nn_image_params_t y_params = { Y_HEIGHT, Y_WIDTH, CHANS_OUT };

    nn_conv2d_job_params_t job_params = {  {  0,  0,  0}, {  Y_HEIGHT, Y_WIDTH, CHANS_OUT}  };
    conv2d_im2col_init(&plan, &job, &x_params, &y_params, &job_params, &conv2d_window, ZERO_POINT, 1);

    memset(X, 0, x_params.height * x_params.width * x_params.channels);
    memset(K, 0, sizeof(K));

    for(int shift2 = 0; shift2 < 4; shift2++){

        PRINTF("\tshift2 = %d...\n", shift2);

        for(int k = 0; k < CHANS_OUT; k++){
            BSO.bias[k] = 16 * k;
            BSO.shift1[k] = 0;
            BSO.scale[k] = 1;
            BSO.offset_scale[k] = 0;
            BSO.offset[k]       = 0;
            BSO.shift2[k] = shift2;
        }
        nn_standard_BSO_layout(bso, (int32_t*) &BSO.bias, (int16_t*) &BSO.shift1, 
                            (int16_t*) &BSO.scale, (int16_t*) &BSO.offset_scale, (int16_t*) &BSO.offset, (int16_t*) &BSO.shift2, NULL, y_params.channels);

        memset(Y, 0xCC, y_params.height * y_params.width * y_params.channels);

        conv2d_im2col((nn_image_t*) Y, (nn_image_t*) X, (nn_image_t*) COL, (nn_tensor_t*) K, bso, &plan, &job);

        for(int row = 0; row < y_params.height; row++){
            for(int col = 0; col < y_params.width; col++){
                for(int chn = 0; chn < y_params.channels; chn++){
                    int8_t y_exp = (16 * chn) >> shift2;
                    check_Y(y_exp, (nn_image_t*) Y, &y_params, row, col, chn, __LINE__);
                }
            }
        }
    }

}
#undef CHANS_IN
#undef CHANS_OUT
#undef K_H
#undef K_W
#undef X_HEIGHT
#undef X_WIDTH
#undef Y_HEIGHT
#undef Y_WIDTH
#undef ZERO_POINT





///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
#define CHANS_IN        ( 4 )
#define CHANS_OUT       ( 4 )
#define K_H             ( 1 )
#define K_W             ( 1 )
#define X_HEIGHT        ( 1 )
#define X_WIDTH         ( 1 )
#define Y_HEIGHT        ( 1 )
#define Y_WIDTH         ( 1 )
#define ZERO_POINT      ( -128 )
void test_conv2d_im2col_case5()
{

    nn_tensor_t WORD_ALIGNED K[CHANS_OUT][ (((K_H*K_W*CHANS_IN+3)>>2)<<2) ];
    nn_image_t  WORD_ALIGNED COL[((K_H*K_W*CHANS_IN+VPU_INT8_EPV-1)>>VPU_INT8_EPV_LOG2)<<VPU_INT8_EPV_LOG2];
    nn_image_t  WORD_ALIGNED X[X_HEIGHT][X_WIDTH][CHANS_IN];
    nn_image_t  WORD_ALIGNED Y[Y_HEIGHT][Y_WIDTH][CHANS_OUT];
    
    struct {
        int32_t bias[CHANS_OUT];
        int16_t shift1[CHANS_OUT];
        int16_t scale[CHANS_OUT];
        int16_t offset_scale[CHANS_OUT];
        int16_t offset[CHANS_OUT];
        int16_t shift2[CHANS_OUT];
    } BSO;

    nn_bso_block_t bso[BSO_BLOCK_COUNT(CHANS_OUT)];

    PRINTF("%s...\n", __func__);

    nn_conv2d_im2col_plan_t plan;
    nn_conv2d_im2col_job_t job;

    nn_conv2d_window_params_t conv2d_window = { { K_H, K_W }, { 0, 0 }, { 1, 1 } }; 

    nn_image_params_t x_params = { X_HEIGHT, X_WIDTH, CHANS_IN };
    nn_image_params_t y_params = { Y_HEIGHT, Y_WIDTH, CHANS_OUT };

    nn_conv2d_job_params_t job_params = {  {  0,  0,  0}, {  Y_HEIGHT, Y_WIDTH, CHANS_OUT}  };
    conv2d_im2col_init(&plan, &job, &x_params, &y_params, &job_params, &conv2d_window, ZERO_POINT, 1);

    memset(X, 1, x_params.height * x_params.width * x_params.channels);
    memset(K, 0, sizeof(K));

    for(int k = 0; k < CHANS_OUT; k++){
        BSO.bias[k] = k;
        BSO.shift1[k] = 0;
        BSO.scale[k] = 1;
        BSO.offset_scale[k] = 0;
        BSO.offset[k]       = 0;
        BSO.shift2[k] = 0;
    }
    nn_standard_BSO_layout(bso, (int32_t*) &BSO.bias, (int16_t*) &BSO.shift1, 
                        (int16_t*) &BSO.scale, (int16_t*) &BSO.offset_scale, (int16_t*) &BSO.offset, (int16_t*) &BSO.shift2, NULL, y_params.channels);

    memset(Y, 0xCC, y_params.height * y_params.width * y_params.channels);

    conv2d_im2col((nn_image_t*) Y, (nn_image_t*) X, (nn_image_t*) COL, (nn_tensor_t*) K, bso, &plan, &job);

    for(int row = 0; row < y_params.height; row++){
        for(int col = 0; col < y_params.width; col++){
            for(int chn = 0; chn < y_params.channels; chn++){
                int8_t y_exp = chn;
                check_Y(y_exp, (nn_image_t*) Y, &y_params, row, col, chn, __LINE__);
            }
        }
    }

}
#undef CHANS_IN
#undef CHANS_OUT
#undef K_H
#undef K_W
#undef X_HEIGHT
#undef X_WIDTH
#undef Y_HEIGHT
#undef Y_WIDTH
#undef ZERO_POINT





///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
#define CHANS_IN        ( 4 )
#define CHANS_OUT       ( 4 )
#define K_H             ( 1 )
#define K_W             ( 1 )
#define X_HEIGHT        ( 1 )
#define X_WIDTH         ( 1 )
#define Y_HEIGHT        ( 1 )
#define Y_WIDTH         ( 1 )
#define ZERO_POINT      ( -128 )
void test_conv2d_im2col_case6()
{

    nn_tensor_t WORD_ALIGNED K[CHANS_OUT][ (((K_H*K_W*CHANS_IN+3)>>2)<<2) ];
    nn_image_t  WORD_ALIGNED COL[((K_H*K_W*CHANS_IN+VPU_INT8_EPV-1)>>VPU_INT8_EPV_LOG2)<<VPU_INT8_EPV_LOG2];
    nn_image_t  WORD_ALIGNED X[X_HEIGHT][X_WIDTH][CHANS_IN];
    nn_image_t  WORD_ALIGNED Y[Y_HEIGHT][Y_WIDTH][CHANS_OUT];
    
    struct {
        int32_t bias[CHANS_OUT];
        int16_t shift1[CHANS_OUT];
        int16_t scale[CHANS_OUT];
        int16_t offset_scale[CHANS_OUT];
        int16_t offset[CHANS_OUT];
        int16_t shift2[CHANS_OUT];
    } BSO;

    nn_bso_block_t bso[BSO_BLOCK_COUNT(CHANS_OUT)];

    PRINTF("%s...\n", __func__);

    nn_conv2d_im2col_plan_t plan;
    nn_conv2d_im2col_job_t job;

    nn_conv2d_window_params_t conv2d_window = { { K_H, K_W }, { 0, 0 }, { 1, 1 } }; 

    nn_image_params_t x_params = { X_HEIGHT, X_WIDTH, CHANS_IN };
    nn_image_params_t y_params = { Y_HEIGHT, Y_WIDTH, CHANS_OUT };

    nn_conv2d_job_params_t job_params = {  {  0,  0,  0}, {  Y_HEIGHT, Y_WIDTH, CHANS_OUT}  };
    conv2d_im2col_init(&plan, &job, &x_params, &y_params, &job_params, &conv2d_window, ZERO_POINT, 1);

    memset(X, 0, x_params.height * x_params.width * x_params.channels);

    memset(K, 0, sizeof(K));
    
    for(int cout = 0; cout < y_params.channels; cout++)
        for(int row = 0; row < conv2d_window.shape.height; row++)
            for(int col = 0; col < conv2d_window.shape.width; col++)
                for(int cin = 0; cin < x_params.channels; cin++)
                    K[cout][row*K_W*CHANS_IN + col*CHANS_IN + cin] = 1;

    for(int k = 0; k < CHANS_OUT; k++){
        BSO.bias[k] = k;
        BSO.shift1[k] = 0;
        BSO.scale[k] = 1;
        BSO.offset_scale[k] = 0;
        BSO.offset[k]       = 0;
        BSO.shift2[k] = 0;
    }
    nn_standard_BSO_layout(bso, (int32_t*) &BSO.bias, (int16_t*) &BSO.shift1, 
                        (int16_t*) &BSO.scale, (int16_t*) &BSO.offset_scale, (int16_t*) &BSO.offset, (int16_t*) &BSO.shift2, NULL, y_params.channels);

    memset(Y, 0xCC, y_params.height * y_params.width * y_params.channels);

    conv2d_im2col((nn_image_t*) Y, (nn_image_t*) X, (nn_image_t*) COL, (nn_tensor_t*) K, bso, &plan, &job);

    for(int row = 0; row < y_params.height; row++){
        for(int col = 0; col < y_params.width; col++){
            for(int chn = 0; chn < y_params.channels; chn++){
                int8_t y_exp = chn;
                check_Y(y_exp, (nn_image_t*) Y, &y_params, row, col, chn, __LINE__);
            }
        }
    }

}
#undef CHANS_IN
#undef CHANS_OUT
#undef K_H
#undef K_W
#undef X_HEIGHT
#undef X_WIDTH
#undef Y_HEIGHT
#undef Y_WIDTH
#undef ZERO_POINT





///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
#define CHANS_IN        ( 4 )
#define CHANS_OUT       ( 4 )
#define K_H             ( 1 )
#define K_W             ( 1 )
#define X_HEIGHT        ( 1 )
#define X_WIDTH         ( 1 )
#define Y_HEIGHT        ( 1 )
#define Y_WIDTH         ( 1 )
#define ZERO_POINT      ( -128 )
void test_conv2d_im2col_case7()
{

    nn_tensor_t WORD_ALIGNED K[CHANS_OUT][ (((K_H*K_W*CHANS_IN+3)>>2)<<2) ];
    nn_image_t  WORD_ALIGNED COL[((K_H*K_W*CHANS_IN+VPU_INT8_EPV-1)>>VPU_INT8_EPV_LOG2)<<VPU_INT8_EPV_LOG2];
    nn_image_t  WORD_ALIGNED X[X_HEIGHT][X_WIDTH][CHANS_IN];
    nn_image_t  WORD_ALIGNED Y[Y_HEIGHT][Y_WIDTH][CHANS_OUT];
    
    struct {
        int32_t bias[CHANS_OUT];
        int16_t shift1[CHANS_OUT];
        int16_t scale[CHANS_OUT];
        int16_t offset_scale[CHANS_OUT];
        int16_t offset[CHANS_OUT];
        int16_t shift2[CHANS_OUT];
    } BSO;

    nn_bso_block_t bso[BSO_BLOCK_COUNT(CHANS_OUT)];

    PRINTF("%s...\n", __func__);
    nn_conv2d_im2col_plan_t plan;
    nn_conv2d_im2col_job_t job;

    nn_conv2d_window_params_t conv2d_window = { { K_H, K_W }, { 0, 0 }, { 1, 1 } }; 

    nn_image_params_t x_params = { X_HEIGHT, X_WIDTH, CHANS_IN };
    nn_image_params_t y_params = { Y_HEIGHT, Y_WIDTH, CHANS_OUT };

    nn_conv2d_job_params_t job_params = {  {  0,  0,  0}, {  Y_HEIGHT, Y_WIDTH, CHANS_OUT}  };
    conv2d_im2col_init(&plan, &job, &x_params, &y_params, &job_params, &conv2d_window, ZERO_POINT, 1);

    memset(X, 1, x_params.height * x_params.width * x_params.channels);

    memset(K, 0, sizeof(K));
    
    for(int cout = 0; cout < y_params.channels; cout++)
        for(int row = 0; row < conv2d_window.shape.height; row++)
            for(int col = 0; col < conv2d_window.shape.width; col++)
                for(int cin = 0; cin < x_params.channels; cin++)
                    K[cout][row*K_W*CHANS_IN + col*CHANS_IN + cin] = 1;

    for(int k = 0; k < CHANS_OUT; k++){
        BSO.bias[k] = k;
        BSO.shift1[k] = 0;
        BSO.scale[k] = 1;
        BSO.offset_scale[k] = 0;
        BSO.offset[k]       = 0;
        BSO.shift2[k] = 0;
    }
    nn_standard_BSO_layout(bso, (int32_t*) &BSO.bias, (int16_t*) &BSO.shift1, 
                        (int16_t*) &BSO.scale, (int16_t*) &BSO.offset_scale, (int16_t*) &BSO.offset, (int16_t*) &BSO.shift2, NULL, y_params.channels);

    memset(Y, 0xCC, y_params.height * y_params.width * y_params.channels);

    conv2d_im2col((nn_image_t*) Y, (nn_image_t*) X, (nn_image_t*) COL, (nn_tensor_t*) K, bso, &plan, &job);

    for(int row = 0; row < y_params.height; row++){
        for(int col = 0; col < y_params.width; col++){
            for(int chn = 0; chn < y_params.channels; chn++){
                int8_t y_exp = CHANS_IN + chn;
                check_Y(y_exp, (nn_image_t*) Y, &y_params, row, col, chn, __LINE__);
            }
        }
    }

}
#undef CHANS_IN
#undef CHANS_OUT
#undef K_H
#undef K_W
#undef X_HEIGHT
#undef X_WIDTH
#undef Y_HEIGHT
#undef Y_WIDTH
#undef ZERO_POINT





///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
#define CHANS_IN        ( 8 )
#define CHANS_OUT       ( 4 )
#define K_H             ( 1 )
#define K_W             ( 1 )
#define X_HEIGHT        ( 1 )
#define X_WIDTH         ( 1 )
#define Y_HEIGHT        ( 1 )
#define Y_WIDTH         ( 1 )
#define ZERO_POINT      ( -128 )
void test_conv2d_im2col_case8()
{

    nn_tensor_t WORD_ALIGNED K[CHANS_OUT][ (((K_H*K_W*CHANS_IN+3)>>2)<<2) ];
    nn_image_t  WORD_ALIGNED COL[((K_H*K_W*CHANS_IN+VPU_INT8_EPV-1)>>VPU_INT8_EPV_LOG2)<<VPU_INT8_EPV_LOG2];
    nn_image_t  WORD_ALIGNED X[X_HEIGHT][X_WIDTH][CHANS_IN];
    nn_image_t  WORD_ALIGNED Y[Y_HEIGHT][Y_WIDTH][CHANS_OUT];
    
    struct {
        int32_t bias[CHANS_OUT];
        int16_t shift1[CHANS_OUT];
        int16_t scale[CHANS_OUT];
        int16_t offset_scale[CHANS_OUT];
        int16_t offset[CHANS_OUT];
        int16_t shift2[CHANS_OUT];
    } BSO;

    nn_bso_block_t bso[BSO_BLOCK_COUNT(CHANS_OUT)];

    PRINTF("%s...\n", __func__);

    nn_conv2d_im2col_plan_t plan;
    nn_conv2d_im2col_job_t job;

    nn_conv2d_window_params_t conv2d_window = { { K_H, K_W }, { 0, 0 }, { 1, 1 } }; 

    nn_image_params_t x_params = { X_HEIGHT, X_WIDTH, CHANS_IN };
    nn_image_params_t y_params = { Y_HEIGHT, Y_WIDTH, CHANS_OUT };

    nn_conv2d_job_params_t job_params = {  {  0,  0,  0}, {  Y_HEIGHT, Y_WIDTH, CHANS_OUT}  };
    conv2d_im2col_init(&plan, &job, &x_params, &y_params, &job_params, &conv2d_window, ZERO_POINT, 1);

    memset(X, 1, x_params.height * x_params.width * x_params.channels);

    memset(K, 0, sizeof(K));

    for(int cout = 0; cout < y_params.channels; cout++)
        for(int row = 0; row < conv2d_window.shape.height; row++)
            for(int col = 0; col < conv2d_window.shape.width; col++)
                for(int cin = 0; cin < x_params.channels; cin++)
                    K[cout][row*K_W*CHANS_IN + col*CHANS_IN + cin] = 1;


    for(int k = 0; k < CHANS_OUT; k++){
        BSO.bias[k] = k;
        BSO.shift1[k] = 0;
        BSO.scale[k] = 1;
        BSO.offset_scale[k] = 0;
        BSO.offset[k]       = 0;
        BSO.shift2[k] = 0;
    }
    nn_standard_BSO_layout(bso, (int32_t*) &BSO.bias, (int16_t*) &BSO.shift1, 
                        (int16_t*) &BSO.scale, (int16_t*) &BSO.offset_scale, (int16_t*) &BSO.offset, (int16_t*) &BSO.shift2, NULL, y_params.channels);

    memset(Y, 0xCC, y_params.height * y_params.width * y_params.channels);

    conv2d_im2col((nn_image_t*) Y, (nn_image_t*) X, (nn_image_t*) COL, (nn_tensor_t*) K, bso, &plan, &job);

    for(int row = 0; row < y_params.height; row++){
        for(int col = 0; col < y_params.width; col++){
            for(int chn = 0; chn < y_params.channels; chn++){
                int8_t y_exp = CHANS_IN + chn;
                check_Y(y_exp, (nn_image_t*) Y, &y_params, row, col, chn, __LINE__);
            }
        }
    }

}
#undef CHANS_IN
#undef CHANS_OUT
#undef K_H
#undef K_W
#undef X_HEIGHT
#undef X_WIDTH
#undef Y_HEIGHT
#undef Y_WIDTH
#undef ZERO_POINT





///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
#define CHANS_IN        ( 4 )
#define CHANS_OUT       ( 36 )
#define K_H             ( 1 )
#define K_W             ( 1 )
#define X_HEIGHT        ( 1 )
#define X_WIDTH         ( 1 )
#define Y_HEIGHT        ( 1 )
#define Y_WIDTH         ( 1 )
#define ZERO_POINT      ( -128 )
void test_conv2d_im2col_case9()
{

    nn_tensor_t WORD_ALIGNED K[CHANS_OUT][ (((K_H*K_W*CHANS_IN+3)>>2)<<2) ];
    nn_image_t  WORD_ALIGNED COL[((K_H*K_W*CHANS_IN+VPU_INT8_EPV-1)>>VPU_INT8_EPV_LOG2)<<VPU_INT8_EPV_LOG2];
    nn_image_t  WORD_ALIGNED X[X_HEIGHT][X_WIDTH][CHANS_IN];
    nn_image_t  WORD_ALIGNED Y[Y_HEIGHT][Y_WIDTH][CHANS_OUT];
    
    struct {
        int32_t bias[CHANS_OUT];
        int16_t shift1[CHANS_OUT];
        int16_t scale[CHANS_OUT];
        int16_t offset_scale[CHANS_OUT];
        int16_t offset[CHANS_OUT];
        int16_t shift2[CHANS_OUT];
    } BSO;

    nn_bso_block_t bso[BSO_BLOCK_COUNT(CHANS_OUT)];

    PRINTF("%s...\n", __func__);

    nn_conv2d_im2col_plan_t plan;
    nn_conv2d_im2col_job_t job;

    nn_conv2d_window_params_t conv2d_window = { { K_H, K_W }, { 0, 0 }, { 1, 1 } }; 

    nn_image_params_t x_params = { X_HEIGHT, X_WIDTH, CHANS_IN };
    nn_image_params_t y_params = { Y_HEIGHT, Y_WIDTH, CHANS_OUT };

    nn_conv2d_job_params_t job_params = {  {  0,  0,  0}, {  Y_HEIGHT, Y_WIDTH, CHANS_OUT}  };
    conv2d_im2col_init(&plan, &job, &x_params, &y_params, &job_params, &conv2d_window, ZERO_POINT, 1);

    memset(X, 1, x_params.height * x_params.width * x_params.channels);
    
    memset(K, 0, sizeof(K));
    for(int cout = 0; cout < y_params.channels; cout++)
        for(int row = 0; row < conv2d_window.shape.height; row++)
            for(int col = 0; col < conv2d_window.shape.width; col++)
                for(int cin = 0; cin < x_params.channels; cin++)
                    K[cout][row*K_W*CHANS_IN + col*CHANS_IN + cin] = 1;

    for(int k = 0; k < CHANS_OUT; k++){
        BSO.bias[k] = k;
        BSO.shift1[k] = 0;
        BSO.scale[k] = 1;
        BSO.offset_scale[k] = 0;
        BSO.offset[k]       = 0;
        BSO.shift2[k] = 0;
    }
    nn_standard_BSO_layout(bso, (int32_t*) &BSO.bias, (int16_t*) &BSO.shift1, 
                        (int16_t*) &BSO.scale, (int16_t*) &BSO.offset_scale, (int16_t*) &BSO.offset, (int16_t*) &BSO.shift2, NULL, y_params.channels);

    memset(Y, 0xCC, y_params.height * y_params.width * y_params.channels);

    conv2d_im2col((nn_image_t*) Y, (nn_image_t*) X, (nn_image_t*) COL, (nn_tensor_t*) K, bso, &plan, &job);

    for(int row = 0; row < y_params.height; row++){
        for(int col = 0; col < y_params.width; col++){
            for(int chn = 0; chn < y_params.channels; chn++){
                int8_t y_exp = CHANS_IN + chn;
                check_Y(y_exp, (nn_image_t*) Y, &y_params, row, col, chn, __LINE__);
            }
        }
    }

}
#undef CHANS_IN
#undef CHANS_OUT
#undef K_H
#undef K_W
#undef X_HEIGHT
#undef X_WIDTH
#undef Y_HEIGHT
#undef Y_WIDTH
#undef ZERO_POINT





///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
#define CHANS_IN        ( 4 )
#define CHANS_OUT       ( 4 )
#define K_H             ( 1 )
#define K_W             ( 1 )
#define X_HEIGHT        ( 1 )
#define X_WIDTH         ( 1 )
#define Y_HEIGHT        ( 1 )
#define Y_WIDTH         ( 1 )
#define ZERO_POINT      ( -128 )
void test_conv2d_im2col_case10()
{

    nn_tensor_t WORD_ALIGNED K[CHANS_OUT][ (((K_H*K_W*CHANS_IN+3)>>2)<<2) ];
    nn_image_t  WORD_ALIGNED COL[((K_H*K_W*CHANS_IN+VPU_INT8_EPV-1)>>VPU_INT8_EPV_LOG2)<<VPU_INT8_EPV_LOG2];
    nn_image_t  WORD_ALIGNED X[X_HEIGHT][X_WIDTH][CHANS_IN];
    nn_image_t  WORD_ALIGNED Y[Y_HEIGHT][Y_WIDTH][CHANS_OUT];
    
    struct {
        int32_t bias[CHANS_OUT];
        int16_t shift1[CHANS_OUT];
        int16_t scale[CHANS_OUT];
        int16_t offset_scale[CHANS_OUT];
        int16_t offset[CHANS_OUT];
        int16_t shift2[CHANS_OUT];
    } BSO;

    nn_bso_block_t bso[BSO_BLOCK_COUNT(CHANS_OUT)];

    PRINTF("%s...\n", __func__);

    nn_conv2d_im2col_plan_t plan;
    nn_conv2d_im2col_job_t job;

    nn_conv2d_window_params_t conv2d_window = { { K_H, K_W }, { 0, 0 }, { 1, 1 } }; 

    nn_image_params_t x_params = { X_HEIGHT, X_WIDTH, CHANS_IN };
    nn_image_params_t y_params = { Y_HEIGHT, Y_WIDTH, CHANS_OUT };

    nn_conv2d_job_params_t job_params = {  {  0,  0,  0}, {  Y_HEIGHT, Y_WIDTH, CHANS_OUT}  };
    conv2d_im2col_init(&plan, &job, &x_params, &y_params, &job_params, &conv2d_window, ZERO_POINT, 1);

    for(int row = 0; row < x_params.height; row++)
        for(int col = 0; col < x_params.width; col++)
            for(int chn = 0; chn < x_params.channels; chn++)
                X[row][col][chn] = chn;
    
    memset(K, 0, sizeof(K));            
    for(int cout = 0; cout < y_params.channels; cout++)
        for(int row = 0; row < conv2d_window.shape.height; row++)
            for(int col = 0; col < conv2d_window.shape.width; col++)
                for(int cin = 0; cin < x_params.channels; cin++)
                    K[cout][row*K_W*CHANS_IN + col*CHANS_IN + cin] = cout;

    for(int k = 0; k < CHANS_OUT; k++){
        BSO.bias[k] = k;
        BSO.shift1[k] = 0;
        BSO.scale[k] = 1;
        BSO.offset_scale[k] = 0;
        BSO.offset[k]       = 0;
        BSO.shift2[k] = 0;
    }
    nn_standard_BSO_layout(bso, (int32_t*) &BSO.bias, (int16_t*) &BSO.shift1, 
                        (int16_t*) &BSO.scale, (int16_t*) &BSO.offset_scale, (int16_t*) &BSO.offset, (int16_t*) &BSO.shift2, NULL, y_params.channels);

    memset(Y, 0xCC, y_params.height * y_params.width * y_params.channels);

    conv2d_im2col((nn_image_t*) Y, (nn_image_t*) X, (nn_image_t*) COL, (nn_tensor_t*) K, bso, &plan, &job);

    for(int row = 0; row < y_params.height; row++){
        for(int col = 0; col < y_params.width; col++){
            for(int chn = 0; chn < y_params.channels; chn++){
                int8_t y_exp = chn*((CHANS_IN-1)*(CHANS_IN/2)) + chn;
                check_Y(y_exp, (nn_image_t*) Y, &y_params, row, col, chn, __LINE__);
            }
        }
    }

}
#undef CHANS_IN
#undef CHANS_OUT
#undef K_H
#undef K_W
#undef X_HEIGHT
#undef X_WIDTH
#undef Y_HEIGHT
#undef Y_WIDTH
#undef ZERO_POINT





///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
#define CHANS_IN        ( 4 )
#define CHANS_OUT       ( 4 )
#define K_H             ( 1 )
#define K_W             ( 1 )
#define X_HEIGHT        ( 3 )
#define X_WIDTH         ( 2 )
#define Y_HEIGHT        ( 3 )
#define Y_WIDTH         ( 2 )
#define ZERO_POINT      ( -128 )
void test_conv2d_im2col_case11()
{

    nn_tensor_t WORD_ALIGNED K[CHANS_OUT][ (((K_H*K_W*CHANS_IN+3)>>2)<<2) ];
    nn_image_t  WORD_ALIGNED COL[((K_H*K_W*CHANS_IN+VPU_INT8_EPV-1)>>VPU_INT8_EPV_LOG2)<<VPU_INT8_EPV_LOG2];
    nn_image_t  WORD_ALIGNED X[X_HEIGHT][X_WIDTH][CHANS_IN];
    nn_image_t  WORD_ALIGNED Y[Y_HEIGHT][Y_WIDTH][CHANS_OUT];
    
    struct {
        int32_t bias[CHANS_OUT];
        int16_t shift1[CHANS_OUT];
        int16_t scale[CHANS_OUT];
        int16_t offset_scale[CHANS_OUT];
        int16_t offset[CHANS_OUT];
        int16_t shift2[CHANS_OUT];
    } BSO;

    nn_bso_block_t bso[BSO_BLOCK_COUNT(CHANS_OUT)];

    PRINTF("%s...\n", __func__);

    nn_conv2d_im2col_plan_t plan;
    nn_conv2d_im2col_job_t job;

    nn_conv2d_window_params_t conv2d_window = { { K_H, K_W }, { 0, 0 }, { 1, 1 } }; 

    nn_image_params_t x_params = { X_HEIGHT, X_WIDTH, CHANS_IN };
    nn_image_params_t y_params = { Y_HEIGHT, Y_WIDTH, CHANS_OUT };

    nn_conv2d_job_params_t job_params = {  {  0,  0,  0}, {  Y_HEIGHT, Y_WIDTH, CHANS_OUT}  };
    conv2d_im2col_init(&plan, &job, &x_params, &y_params, &job_params, &conv2d_window, ZERO_POINT, 1);

    for(int row = 0; row < x_params.height; row++)
        for(int col = 0; col < x_params.width; col++)
            for(int cin = 0; cin < x_params.channels; cin++)
                X[row][col][cin] = 1;
                
    memset(K, 0, sizeof(K));   
    for(int cout = 0; cout < y_params.channels; cout++)
        for(int row = 0; row < conv2d_window.shape.height; row++)
            for(int col = 0; col < conv2d_window.shape.width; col++)
                for(int cin = 0; cin < x_params.channels; cin++)
                    K[cout][row*K_W*CHANS_IN + col*CHANS_IN + cin] = 1;

    for(int k = 0; k < CHANS_OUT; k++){
        BSO.bias[k] = 0;
        BSO.shift1[k] = 0;
        BSO.scale[k] = 1;
        BSO.offset_scale[k] = 0;
        BSO.offset[k]       = 0;
        BSO.shift2[k] = 0;
    }
    nn_standard_BSO_layout(bso, (int32_t*) &BSO.bias, (int16_t*) &BSO.shift1, 
                        (int16_t*) &BSO.scale, (int16_t*) &BSO.offset_scale, (int16_t*) &BSO.offset, (int16_t*) &BSO.shift2, NULL, y_params.channels);

    memset(Y, 0xCC, y_params.height * y_params.width * y_params.channels);

    conv2d_im2col((nn_image_t*) Y, (nn_image_t*) X, (nn_image_t*) COL, (nn_tensor_t*) K, bso, &plan, &job);

    for(int row = 0; row < y_params.height; row++){
        for(int col = 0; col < y_params.width; col++){
            for(int cout = 0; cout < y_params.channels; cout++){
                int8_t y_exp = 1 * CHANS_IN;
                check_Y(y_exp, (nn_image_t*) Y, &y_params, row, col, cout, __LINE__);
            }
        }
    }

}
#undef CHANS_IN
#undef CHANS_OUT
#undef K_H
#undef K_W
#undef X_HEIGHT
#undef X_WIDTH
#undef Y_HEIGHT
#undef Y_WIDTH
#undef ZERO_POINT





///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
#define CHANS_IN        ( 4 )
#define CHANS_OUT       ( 4 )
#define K_H             ( 1 )
#define K_W             ( 3 )
#define X_HEIGHT        ( 1 )
#define X_WIDTH         ( 4 )
#define Y_HEIGHT        ( 1 )
#define Y_WIDTH         ( 2 )
#define K_V_STRIDE      ( 1 )
#define K_H_STRIDE      ( 1 )
#define ZERO_POINT      ( -128 )
void test_conv2d_im2col_case12()
{

    nn_tensor_t WORD_ALIGNED K[CHANS_OUT][ (((K_H*K_W*CHANS_IN+3)>>2)<<2) ];
    nn_image_t  WORD_ALIGNED COL[((K_H*K_W*CHANS_IN+VPU_INT8_EPV-1)>>VPU_INT8_EPV_LOG2)<<VPU_INT8_EPV_LOG2];
    nn_image_t  WORD_ALIGNED X[X_HEIGHT][X_WIDTH][CHANS_IN];
    nn_image_t  WORD_ALIGNED Y[Y_HEIGHT][Y_WIDTH][CHANS_OUT];
    
    struct {
        int32_t bias[CHANS_OUT];
        int16_t shift1[CHANS_OUT];
        int16_t scale[CHANS_OUT];
        int16_t offset_scale[CHANS_OUT];
        int16_t offset[CHANS_OUT];
        int16_t shift2[CHANS_OUT];
    } BSO;

    nn_bso_block_t bso[BSO_BLOCK_COUNT(CHANS_OUT)];

    PRINTF("%s...\n", __func__);

    nn_conv2d_im2col_plan_t plan;
    nn_conv2d_im2col_job_t job;

    nn_conv2d_window_params_t conv2d_window = { { K_H, K_W }, { 0, 0 }, { K_V_STRIDE, K_H_STRIDE } }; 

    nn_image_params_t x_params = { X_HEIGHT, X_WIDTH, CHANS_IN };
    nn_image_params_t y_params = { Y_HEIGHT, Y_WIDTH, CHANS_OUT };

    nn_conv2d_job_params_t job_params = {  {  0,  0,  0}, {  Y_HEIGHT, Y_WIDTH, CHANS_OUT}  };
    conv2d_im2col_init(&plan, &job, &x_params, &y_params, &job_params, &conv2d_window, ZERO_POINT, 1);

    for(int row = 0; row < x_params.height; row++)
        for(int col = 0; col < x_params.width; col++)
            for(int cin = 0; cin < x_params.channels; cin++)
                X[row][col][cin] = 1;
                
    memset(K, 0, sizeof(K));   
    for(int cout = 0; cout < y_params.channels; cout++)
        for(int row = 0; row < conv2d_window.shape.height; row++)
            for(int col = 0; col < conv2d_window.shape.width; col++)
                for(int cin = 0; cin < x_params.channels; cin++)
                    K[cout][row*K_W*CHANS_IN + col*CHANS_IN + cin] = col+1;

    for(int k = 0; k < CHANS_OUT; k++){
        BSO.bias[k] = 0;
        BSO.shift1[k] = 0;
        BSO.scale[k] = 1;
        BSO.offset_scale[k] = 0;
        BSO.offset[k]       = 0;
        BSO.shift2[k] = 0;
    }
    nn_standard_BSO_layout(bso, (int32_t*) &BSO.bias, (int16_t*) &BSO.shift1, 
                        (int16_t*) &BSO.scale, (int16_t*) &BSO.offset_scale, (int16_t*) &BSO.offset, (int16_t*) &BSO.shift2, NULL, y_params.channels);

    memset(Y, 0xCC, y_params.height * y_params.width * y_params.channels);

    conv2d_im2col((nn_image_t*) Y, (nn_image_t*) X, (nn_image_t*) COL, (nn_tensor_t*) K, bso, &plan, &job);

    for(int row = 0; row < y_params.height; row++){
        for(int col = 0; col < y_params.width; col++){
            for(int cout = 0; cout < y_params.channels; cout++){
                int8_t y_exp = 6 * x_params.channels;
                check_Y(y_exp, (nn_image_t*) Y, &y_params, row, col, cout, __LINE__);
            }
        }
    }
}
#undef CHANS_IN
#undef CHANS_OUT
#undef K_H
#undef K_W
#undef X_HEIGHT
#undef X_WIDTH
#undef Y_HEIGHT
#undef Y_WIDTH
#undef K_V_STRIDE
#undef K_H_STRIDE
#undef ZERO_POINT





///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
#define CHANS_IN        ( 4 )
#define CHANS_OUT       ( 4 )
#define K_H             ( 2 )
#define K_W             ( 3 )
#define X_HEIGHT        ( 4 )
#define X_WIDTH         ( 6 )
#define Y_HEIGHT        ( 2 )
#define Y_WIDTH         ( 2 )
#define K_V_STRIDE      ( 2 )
#define K_H_STRIDE      ( 3 )
#define ZERO_POINT      ( -128 )
void test_conv2d_im2col_case13()
{

    nn_tensor_t WORD_ALIGNED K[CHANS_OUT][ (((K_H*K_W*CHANS_IN+3)>>2)<<2) ];
    nn_image_t  WORD_ALIGNED COL[((K_H*K_W*CHANS_IN+VPU_INT8_EPV-1)>>VPU_INT8_EPV_LOG2)<<VPU_INT8_EPV_LOG2];
    nn_image_t  WORD_ALIGNED X[X_HEIGHT][X_WIDTH][CHANS_IN];
    nn_image_t  WORD_ALIGNED Y[Y_HEIGHT][Y_WIDTH][CHANS_OUT];
    
    struct {
        int32_t bias[CHANS_OUT];
        int16_t shift1[CHANS_OUT];
        int16_t scale[CHANS_OUT];
        int16_t offset_scale[CHANS_OUT];
        int16_t offset[CHANS_OUT];
        int16_t shift2[CHANS_OUT];
    } BSO;

    nn_bso_block_t bso[BSO_BLOCK_COUNT(CHANS_OUT)];

    PRINTF("%s...\n", __func__);

    nn_conv2d_im2col_plan_t plan;
    nn_conv2d_im2col_job_t job;

    nn_conv2d_window_params_t conv2d_window = { { K_H, K_W }, { 0, 0 }, { K_V_STRIDE, K_H_STRIDE } }; 

    nn_image_params_t x_params = { X_HEIGHT, X_WIDTH, CHANS_IN };
    nn_image_params_t y_params = { Y_HEIGHT, Y_WIDTH, CHANS_OUT };

    nn_conv2d_job_params_t job_params = {  {  0,  0,  0}, {  Y_HEIGHT, Y_WIDTH, CHANS_OUT}  };
    conv2d_im2col_init(&plan, &job, &x_params, &y_params, &job_params, &conv2d_window, ZERO_POINT, 1);

    for(int row = 0; row < x_params.height; row++)
        for(int col = 0; col < x_params.width; col++)
            for(int cin = 0; cin < x_params.channels; cin++)
                X[row][col][cin] = 1;
                
    memset(K, 0, sizeof(K));   
    for(int cout = 0; cout < y_params.channels; cout++)
        for(int row = 0; row < conv2d_window.shape.height; row++)
            for(int col = 0; col < conv2d_window.shape.width; col++)
                for(int cin = 0; cin < x_params.channels; cin++)
                    K[cout][row*K_W*CHANS_IN + col*CHANS_IN + cin] = K_W * row + col + 1;

    for(int k = 0; k < CHANS_OUT; k++){
        BSO.bias[k] = 0;
        BSO.shift1[k] = 0;
        BSO.scale[k] = 1;
        BSO.offset_scale[k] = 0;
        BSO.offset[k]       = 0;
        BSO.shift2[k] = 0;
    }
    nn_standard_BSO_layout(bso, (int32_t*) &BSO.bias, (int16_t*) &BSO.shift1, 
                        (int16_t*) &BSO.scale, (int16_t*) &BSO.offset_scale, (int16_t*) &BSO.offset, (int16_t*) &BSO.shift2, NULL, y_params.channels);

    memset(Y, 0xCC, y_params.height * y_params.width * y_params.channels);

    conv2d_im2col((nn_image_t*) Y, (nn_image_t*) X, (nn_image_t*) COL, (nn_tensor_t*) K, bso, &plan, &job);

    for(int row = 0; row < y_params.height; row++){
        for(int col = 0; col < y_params.width; col++){
            for(int cout = 0; cout < y_params.channels; cout++){
                int8_t y_exp = 21 * x_params.channels;
                check_Y(y_exp, (nn_image_t*) Y, &y_params, row, col, cout, __LINE__);
            }
        }
    }
}
#undef CHANS_IN
#undef CHANS_OUT
#undef K_H
#undef K_W
#undef X_HEIGHT
#undef X_WIDTH
#undef Y_HEIGHT
#undef Y_WIDTH
#undef K_V_STRIDE
#undef K_H_STRIDE
#undef ZERO_POINT





///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
#define CHANS_IN        ( 4 )
#define CHANS_OUT       ( 4 )
#define K_H             ( 3 )
#define K_W             ( 3 )
#define X_HEIGHT        ( 3 )
#define X_WIDTH         ( 3 )
#define Y_HEIGHT        ( 3 )
#define Y_WIDTH         ( 3 )
#define K_V_STRIDE      ( 1 )
#define K_H_STRIDE      ( 1 )
#define ZERO_POINT      ( 2 )
void test_conv2d_im2col_case14()
{

    nn_tensor_t WORD_ALIGNED K[CHANS_OUT][ (((K_H*K_W*CHANS_IN+3)>>2)<<2) ];
    nn_image_t  WORD_ALIGNED COL[((K_H*K_W*CHANS_IN+VPU_INT8_EPV-1)>>VPU_INT8_EPV_LOG2)<<VPU_INT8_EPV_LOG2];
    nn_image_t  WORD_ALIGNED X[X_HEIGHT][X_WIDTH][CHANS_IN];
    nn_image_t  WORD_ALIGNED Y[Y_HEIGHT][Y_WIDTH][CHANS_OUT];
    
    struct {
        int32_t bias[CHANS_OUT];
        int16_t shift1[CHANS_OUT];
        int16_t scale[CHANS_OUT];
        int16_t offset_scale[CHANS_OUT];
        int16_t offset[CHANS_OUT];
        int16_t shift2[CHANS_OUT];
    } BSO;

    nn_bso_block_t bso[BSO_BLOCK_COUNT(CHANS_OUT)];

    PRINTF("%s...\n", __func__);

    nn_conv2d_im2col_plan_t plan;
    nn_conv2d_im2col_job_t job;

    nn_conv2d_window_params_t conv2d_window = { { K_H, K_W }, { -1, -1 }, { K_V_STRIDE, K_H_STRIDE } }; 

    nn_image_params_t x_params = { X_HEIGHT, X_WIDTH, CHANS_IN };
    nn_image_params_t y_params = { Y_HEIGHT, Y_WIDTH, CHANS_OUT };

    nn_conv2d_job_params_t job_params = {  {  0,  0,  0}, {  Y_HEIGHT, Y_WIDTH, CHANS_OUT}  };
    conv2d_im2col_init(&plan, &job, &x_params, &y_params, &job_params, &conv2d_window, ZERO_POINT, 1);

    for(int row = 0; row < x_params.height; row++)
        for(int col = 0; col < x_params.width; col++)
            for(int cin = 0; cin < x_params.channels; cin++)
                X[row][col][cin] = 1;
                
    memset(K, 0, sizeof(K));   
    for(int cout = 0; cout < y_params.channels; cout++)
        for(int row = 0; row < conv2d_window.shape.height; row++)
            for(int col = 0; col < conv2d_window.shape.width; col++)
                for(int cin = 0; cin < x_params.channels; cin++)
                    K[cout][row*K_W*CHANS_IN + col*CHANS_IN + cin] = 1;

    for(int k = 0; k < CHANS_OUT; k++){
        BSO.bias[k] = k;
        BSO.shift1[k] = 0;
        BSO.scale[k] = 1;
        BSO.offset_scale[k] = 0;
        BSO.offset[k]       = 0;
        BSO.shift2[k] = 0;
    }
    nn_standard_BSO_layout(bso, (int32_t*) &BSO.bias, (int16_t*) &BSO.shift1, 
                        (int16_t*) &BSO.scale, (int16_t*) &BSO.offset_scale, (int16_t*) &BSO.offset, (int16_t*) &BSO.shift2, NULL, y_params.channels);

    memset(Y, 0xCC, y_params.height * y_params.width * y_params.channels);

    conv2d_im2col((nn_image_t*) Y, (nn_image_t*) X, (nn_image_t*) COL, (nn_tensor_t*) K, bso, &plan, &job);

    nn_image_t  WORD_ALIGNED Y_exp[Y_HEIGHT][Y_WIDTH] = {
        { 14, 12, 14},
        { 12,  9, 12},
        { 14, 12, 14},
    };

    for(int row = 0; row < y_params.height; row++){
        for(int col = 0; col < y_params.width; col++){
            for(int cout = 0; cout < y_params.channels; cout++){
                int8_t y_exp = CHANS_IN * Y_exp[row][col] + cout;
                check_Y(y_exp, (nn_image_t*) Y, &y_params, row, col, cout, __LINE__);
            }
        }
    }
}
#undef CHANS_IN
#undef CHANS_OUT
#undef K_H
#undef K_W
#undef X_HEIGHT
#undef X_WIDTH
#undef Y_HEIGHT
#undef Y_WIDTH
#undef K_V_STRIDE
#undef K_H_STRIDE
#undef ZERO_POINT





///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
#define CHANS_IN        ( 4 )
#define CHANS_OUT       ( 32 )
#define K_H             ( 3 )
#define K_W             ( 3 )
#define X_HEIGHT        ( 5 )
#define X_WIDTH         ( 5 )
#define Y_HEIGHT        ( 3 )
#define Y_WIDTH         ( 3 )
#define K_V_STRIDE      ( 2 )
#define K_H_STRIDE      ( 2 )
#define ZERO_POINT      ( 8 )
void test_conv2d_im2col_case15()
{

    nn_tensor_t WORD_ALIGNED K[CHANS_OUT][ (((K_H*K_W*CHANS_IN+3)>>2)<<2) ];
    nn_image_t  WORD_ALIGNED COL[((K_H*K_W*CHANS_IN+VPU_INT8_EPV-1)>>VPU_INT8_EPV_LOG2)<<VPU_INT8_EPV_LOG2];
    nn_image_t  WORD_ALIGNED X[X_HEIGHT][X_WIDTH][CHANS_IN];
    nn_image_t  WORD_ALIGNED Y[Y_HEIGHT][Y_WIDTH][CHANS_OUT];
    
    struct {
        int32_t bias[CHANS_OUT];
        int16_t shift1[CHANS_OUT];
        int16_t scale[CHANS_OUT];
        int16_t offset_scale[CHANS_OUT];
        int16_t offset[CHANS_OUT];
        int16_t shift2[CHANS_OUT];
    } BSO;

    nn_bso_block_t bso[BSO_BLOCK_COUNT(CHANS_OUT)];

    PRINTF("%s...\n", __func__);

    nn_conv2d_im2col_plan_t plan;
    nn_conv2d_im2col_job_t job;

    nn_conv2d_window_params_t conv2d_window = { { K_H, K_W }, { -1, -1 }, { K_V_STRIDE, K_H_STRIDE } }; 

    nn_image_params_t x_params = { X_HEIGHT, X_WIDTH, CHANS_IN };
    nn_image_params_t y_params = { Y_HEIGHT, Y_WIDTH, CHANS_OUT };

    nn_conv2d_job_params_t job_params = {  {  0,  0,  0}, {  Y_HEIGHT, Y_WIDTH, CHANS_OUT}  };
    conv2d_im2col_init(&plan, &job, &x_params, &y_params, &job_params, &conv2d_window, ZERO_POINT, 1);

    nn_image_t X_vals[X_HEIGHT][X_WIDTH] = {
        {  2,  4, 2, 8, 4 },
        {  4,  2, 4, 2, 8 },
        {  8,  4, 2, 2, 4 },
        {  2, 16, 2, 8, 2 },
        {  2,  4, 2, 2, 4 },
    };
    for(int row = 0; row < x_params.height; row++)
        for(int col = 0; col < x_params.width; col++)
            for(int cin = 0; cin < x_params.channels; cin++)
                X[row][col][cin] = X_vals[row][col];
                
    memset(K, 0, sizeof(K));   
    for(int cout = 0; cout < y_params.channels; cout++)
        for(int row = 0; row < conv2d_window.shape.height; row++)
            for(int col = 0; col < conv2d_window.shape.width; col++)
                for(int cin = 0; cin < x_params.channels; cin++)
                    K[cout][row*K_W*CHANS_IN + col*CHANS_IN + cin] = 1;

    for(int k = 0; k < CHANS_OUT; k++){
        BSO.bias[k] = 0;
        BSO.shift1[k] = 1;
        BSO.scale[k] = 2;
        BSO.offset_scale[k] = 1<<3;
        BSO.offset[k] = k;
        BSO.shift2[k] = 3;
    }
    nn_standard_BSO_layout(bso, (int32_t*) &BSO.bias, (int16_t*) &BSO.shift1, 
                        (int16_t*) &BSO.scale, (int16_t*) &BSO.offset_scale, (int16_t*) &BSO.offset, (int16_t*) &BSO.shift2, NULL, y_params.channels);

    memset(Y, 0xCC, y_params.height * y_params.width * y_params.channels);

    conv2d_im2col((nn_image_t*) Y, (nn_image_t*) X, (nn_image_t*) COL, (nn_tensor_t*) K, bso, &plan, &job);

/*       __ __
       |8  8  8| 8  8  8  8 
       |8  2  4| 2  8  4  8
       |8__4__2| 4  2  8  8
        8  8  4  2  2  4  8
        8  2 16  2  8  2  8
        8  2  4  2  2  4  8
        8  8  8  8  8  8  8

*/

    nn_image_t  WORD_ALIGNED Y_exp[Y_HEIGHT][Y_WIDTH] = {
        { 26, 23, 31},
        { 30, 21, 25},
        { 32, 29, 28},
    };

    for(int row = 0; row < y_params.height; row++){
        for(int col = 0; col < y_params.width; col++){
            for(int cout = 0; cout < y_params.channels; cout++){
                int8_t y_exp = Y_exp[row][col] + cout;
                check_Y(y_exp, (nn_image_t*) Y, &y_params, row, col, cout, __LINE__);
            }
        }
    }
}
#undef CHANS_IN
#undef CHANS_OUT
#undef K_H
#undef K_W
#undef X_HEIGHT
#undef X_WIDTH
#undef Y_HEIGHT
#undef Y_WIDTH
#undef K_V_STRIDE
#undef K_H_STRIDE
#undef ZERO_POINT





///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
#define CHANS_IN        ( 4 )
#define CHANS_OUT       ( 32 )
#define K_H             ( 3 )
#define K_W             ( 3 )
#define X_HEIGHT        ( 5 )
#define X_WIDTH         ( 5 )
#define Y_HEIGHT        ( 3 )
#define Y_WIDTH         ( 3 )
#define K_V_STRIDE      ( 2 )
#define K_H_STRIDE      ( 2 )
#define ZERO_POINT      ( 8 )
void test_conv2d_im2col_case16()
{

    nn_tensor_t WORD_ALIGNED K[CHANS_OUT][ (((K_H*K_W*CHANS_IN+3)>>2)<<2) ];
    nn_image_t  WORD_ALIGNED COL[((K_H*K_W*CHANS_IN+VPU_INT8_EPV-1)>>VPU_INT8_EPV_LOG2)<<VPU_INT8_EPV_LOG2];
    nn_image_t  WORD_ALIGNED X[X_HEIGHT][X_WIDTH][CHANS_IN];
    nn_image_t  WORD_ALIGNED Y[Y_HEIGHT][Y_WIDTH][CHANS_OUT];
    
    struct {
        int32_t bias[CHANS_OUT];
        int16_t shift1[CHANS_OUT];
        int16_t scale[CHANS_OUT];
        int16_t offset_scale[CHANS_OUT];
        int16_t offset[CHANS_OUT];
        int16_t shift2[CHANS_OUT];
    } BSO;

    nn_bso_block_t bso[BSO_BLOCK_COUNT(CHANS_OUT)];

    PRINTF("%s...\n", __func__);

    nn_conv2d_im2col_plan_t plan;
    nn_conv2d_im2col_job_t job[5];

    nn_conv2d_window_params_t conv2d_window = { { K_H, K_W }, { -1, -1 }, { K_V_STRIDE, K_H_STRIDE } }; 

    nn_image_params_t x_params = { X_HEIGHT, X_WIDTH, CHANS_IN };
    nn_image_params_t y_params = { Y_HEIGHT, Y_WIDTH, CHANS_OUT };

    nn_conv2d_job_params_t job_params[] = {
        {  {  0,  0,  0}, {  1, 3, 32}  },
        {  {  1,  0,  0}, {  2, 1, 16}  },
        {  {  1,  1,  0}, {  2, 2, 16}  },
        {  {  1,  0, 16}, {  1, 3, 16}  },
        {  {  2,  0, 16}, {  1, 2, 16}  },
        //Leaves Y[2,2,16:32] uncalculated
    };

    const unsigned job_count = sizeof(job_params) / sizeof(nn_conv2d_job_params_t);

    conv2d_im2col_init(&plan, job, &x_params, &y_params, job_params, &conv2d_window, ZERO_POINT, job_count);

    nn_image_t X_vals[X_HEIGHT][X_WIDTH] = {
        {  2,  4, 2, 8, 4 },
        {  4,  2, 4, 2, 8 },
        {  8,  4, 2, 2, 4 },
        {  2, 16, 2, 8, 2 },
        {  2,  4, 2, 2, 4 },
    };
    for(int row = 0; row < x_params.height; row++)
        for(int col = 0; col < x_params.width; col++)
            for(int cin = 0; cin < x_params.channels; cin++)
                X[row][col][cin] = X_vals[row][col];
                
    memset(K, 0, sizeof(K));   
    for(int cout = 0; cout < y_params.channels; cout++)
        for(int row = 0; row < conv2d_window.shape.height; row++)
            for(int col = 0; col < conv2d_window.shape.width; col++)
                for(int cin = 0; cin < x_params.channels; cin++)
                    K[cout][row*K_W*CHANS_IN + col*CHANS_IN + cin] = 1;

    for(int k = 0; k < CHANS_OUT; k++){
        BSO.bias[k] = k * (1<<3);
        BSO.shift1[k] = 1;
        BSO.scale[k] = 2;
        BSO.offset_scale[k] = 0;
        BSO.offset[k]       = 0;
        BSO.shift2[k] = 3;
    }
    nn_standard_BSO_layout(bso, (int32_t*) &BSO.bias, (int16_t*) &BSO.shift1, 
                        (int16_t*) &BSO.scale, (int16_t*) &BSO.offset_scale, (int16_t*) &BSO.offset, (int16_t*) &BSO.shift2, NULL, y_params.channels);

    memset(Y, 0xCC, y_params.height * y_params.width * y_params.channels);

    for(int i = 0; i < job_count; i++)
        conv2d_im2col((nn_image_t*) Y, (nn_image_t*) X, (nn_image_t*) COL, (nn_tensor_t*) K, bso, &plan, &job[i]);

/*       __ __
       |8  8  8| 8  8  8  8 
       |8  2  4| 2  8  4  8
       |8__4__2| 4  2  8  8
        8  8  4  2  2  4  8
        8  2 16  2  8  2  8
        8  2  4  2  2  4  8
        8  8  8  8  8  8  8

*/

    nn_image_t  WORD_ALIGNED Y_exp[Y_HEIGHT][Y_WIDTH] = {
        { 26, 23, 31},
        { 30, 21, 25},
        { 32, 29, 28},
    };

    for(int row = 0; row < y_params.height; row++){
        for(int col = 0; col < y_params.width; col++){
            for(int cout = 0; cout < y_params.channels; cout++){
                int8_t y_exp = Y_exp[row][col] + cout;
                if(row == 2 && col == 2 && cout >= 16)
                    y_exp = 0xCC;
                check_Y(y_exp, (nn_image_t*) Y, &y_params, row, col, cout, __LINE__);
            }
        }
    }
}
#undef CHANS_IN
#undef CHANS_OUT
#undef K_H
#undef K_W
#undef X_HEIGHT
#undef X_WIDTH
#undef Y_HEIGHT
#undef Y_WIDTH
#undef K_V_STRIDE
#undef K_H_STRIDE
#undef ZERO_POINT





///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
#define CHANS_IN        ( 8 )
#define CHANS_OUT       ( 20 )
#define K_H             ( 3 )
#define K_W             ( 3 )
#define X_HEIGHT        ( 5 )
#define X_WIDTH         ( 5 )
#define Y_HEIGHT        ( 3 )
#define Y_WIDTH         ( 3 )
#define K_V_STRIDE      ( 2 )
#define K_H_STRIDE      ( 2 )
#define ZERO_POINT      ( 8 )
void test_conv2d_im2col_case17()
{

    nn_tensor_t WORD_ALIGNED K[CHANS_OUT][ (((K_H*K_W*CHANS_IN+3)>>2)<<2) ];
    nn_image_t  WORD_ALIGNED COL[((K_H*K_W*CHANS_IN+VPU_INT8_EPV-1)>>VPU_INT8_EPV_LOG2)<<VPU_INT8_EPV_LOG2];
    nn_image_t  WORD_ALIGNED X[X_HEIGHT][X_WIDTH][CHANS_IN];
    nn_image_t  WORD_ALIGNED Y[Y_HEIGHT][Y_WIDTH][CHANS_OUT];
    
    struct {
        int32_t bias[CHANS_OUT];
        int16_t shift1[CHANS_OUT];
        int16_t scale[CHANS_OUT];
        int16_t offset_scale[CHANS_OUT];
        int16_t offset[CHANS_OUT];
        int16_t shift2[CHANS_OUT];
    } BSO;

    nn_bso_block_t bso[BSO_BLOCK_COUNT(CHANS_OUT)];

    PRINTF("%s...\n", __func__);

    nn_conv2d_im2col_plan_t plan;
    nn_conv2d_im2col_job_t job;

    nn_conv2d_window_params_t conv2d_window = { { K_H, K_W }, { -1, -1 }, { K_V_STRIDE, K_H_STRIDE } }; 

    nn_image_params_t x_params = { X_HEIGHT, X_WIDTH, CHANS_IN };
    nn_image_params_t y_params = { Y_HEIGHT, Y_WIDTH, CHANS_OUT };

    nn_conv2d_job_params_t job_params = {  {  0,  0,  0}, {  Y_HEIGHT, Y_WIDTH, CHANS_OUT}  };
    conv2d_im2col_init(&plan, &job, &x_params, &y_params, &job_params, &conv2d_window, ZERO_POINT, 1);

    nn_image_t X_vals[X_HEIGHT][X_WIDTH] = {
        {  6,  6, 2, 8, 4 },
        {  6,  6, 4, 2, 8 },
        {  8,  4, 2, 2, 4 },
        {  2, 16, 2, 8, 2 },
        {  2,  4, 2, 2, 4 },
    };
    for(int row = 0; row < x_params.height; row++)
        for(int col = 0; col < x_params.width; col++)
            for(int cin = 0; cin < x_params.channels; cin++)
                X[row][col][cin] = X_vals[row][col];
                
    memset(K, 0, sizeof(K));   
    for(int cout = 0; cout < y_params.channels; cout++)
        for(int row = 0; row < conv2d_window.shape.height; row++)
            for(int col = 0; col < conv2d_window.shape.width; col++)
                for(int cin = 0; cin < x_params.channels; cin++)
                    K[cout][row*K_W*CHANS_IN + col*CHANS_IN + cin] = 1;

    for(int k = 0; k < CHANS_OUT; k++){
        BSO.bias[k] = 0;// k * (1<<6);
        BSO.shift1[k] = 1;
        BSO.scale[k] = 2;
        BSO.offset_scale[k] = 0;
        BSO.offset[k]       = 0;
        BSO.shift2[k] = 6;
    }
    nn_standard_BSO_layout(bso, (int32_t*) &BSO.bias, (int16_t*) &BSO.shift1, 
                        (int16_t*) &BSO.scale, (int16_t*) &BSO.offset_scale, (int16_t*) &BSO.offset, (int16_t*) &BSO.shift2, NULL, y_params.channels);

    memset(Y, 0xCC, y_params.height * y_params.width * y_params.channels);

    conv2d_im2col((nn_image_t*) Y, (nn_image_t*) X, (nn_image_t*) COL, (nn_tensor_t*) K, bso, &plan, &job);

/*       __ __
       |8  8  8| 8  8  8  8 
       |8  6  6| 2  8  4  8
       |8__6__6| 4  2  8  8
        8  8  4  2  2  4  8
        8  2 16  2  8  2  8
        8  2  4  2  2  4  8
        8  8  8  8  8  8  8

        52*20*

*/

    int32_t Y_exp[Y_HEIGHT][Y_WIDTH] = {
        { 8+8+8+8+6+6+8+6+6 , 8+8+8+6+2+8+6+4+2 , 8+8+8+8+4+8+2+8+8},
        { 8+6+6+8+8+4+8+2+16, 6+4+2+4+2+2+16+2+8, 2+8+8+2+4+8+8+2+8},
        { 8+2+16+8+2+4+8+8+8, 16+2+8+4+2+2+8+8+8, 8+2+8+2+4+8+8+8+8},
    };

    for(int row = 0; row < y_params.height; row++){
        for(int col = 0; col < y_params.width; col++){
            for(int cout = 0; cout < y_params.channels; cout++){

                int32_t acc = x_params.channels * Y_exp[row][col] + BSO.bias[cout];
                // printf("%ld\n", acc);
                acc = acc >> BSO.shift1[cout];
                acc *= BSO.scale[cout];

                acc = acc + (1 << (BSO.shift2[cout] - 1));
                acc = acc >> BSO.shift2[cout];

                int8_t y_exp = acc;
                check_Y(y_exp, (nn_image_t*) Y, &y_params, row, col, cout, __LINE__);
            }
        }
    }
}
#undef CHANS_IN
#undef CHANS_OUT
#undef K_H
#undef K_W
#undef X_HEIGHT
#undef X_WIDTH
#undef Y_HEIGHT
#undef Y_WIDTH
#undef K_V_STRIDE
#undef K_H_STRIDE
#undef ZERO_POINT







void test_conv2d_im2col()
{
    UNITY_SET_FILE();
    
    RUN_TEST(test_conv2d_im2col_case0);
    RUN_TEST(test_conv2d_im2col_case1);
    RUN_TEST(test_conv2d_im2col_case2);
    RUN_TEST(test_conv2d_im2col_case3);
    RUN_TEST(test_conv2d_im2col_case4);
    RUN_TEST(test_conv2d_im2col_case5);
    RUN_TEST(test_conv2d_im2col_case6);
    RUN_TEST(test_conv2d_im2col_case7);
    RUN_TEST(test_conv2d_im2col_case8);
    RUN_TEST(test_conv2d_im2col_case9);
    RUN_TEST(test_conv2d_im2col_case10);
    RUN_TEST(test_conv2d_im2col_case11);
    RUN_TEST(test_conv2d_im2col_case12);
    RUN_TEST(test_conv2d_im2col_case13);
    RUN_TEST(test_conv2d_im2col_case14);
    RUN_TEST(test_conv2d_im2col_case15);
    RUN_TEST(test_conv2d_im2col_case16);
    RUN_TEST(test_conv2d_im2col_case17);
}