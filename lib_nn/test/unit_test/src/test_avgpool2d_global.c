
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>


#include "tst_common.h"

#include "nn_operator.h"
#include "xs3_vpu.h"
#include "../src/nn_op_helper.h"

#include "unity.h"

#define DO_PRINT_EXTRA ((DO_PRINT_EXTRA_GLOBAL) && 0)



#if CONFIG_SYMMETRIC_SATURATION_avgpool2d_global
  #define NEG_SAT_VAL   (-127)
#else
  #define NEG_SAT_VAL   (-128)
#endif 



static void compute_scale_shift(
    int8_t* out_scale,
    uint16_t* out_shift,
    nn_image_params_t* x_params)
{

    //Figure out the scale and shift
    uint32_t pixels = x_params->height * x_params->width;

    //Find c = ceil(log2(pix)), which can be achieve via clz()
    const int c = ceil_log2(pixels);

    assert(c != -1); //pix == 0

    if(pixels == (1<<c)){
        //window pixel count is already a power of 2   (2^c)
        *out_scale = 1;
        *out_shift = c;
    } else {
        const unsigned q = 31 - c - 6;
        // 2^31 / pixels
        const unsigned g = 0x80000000 / pixels;
        const unsigned h = (g + (1 << (q-1))) >> q; //Rounding down-shift

        assert(h > (1<<6));
        assert(h < (1<<7));

        *out_scale = (int8_t)h;
        *out_shift = c+6;
    }

}



#define CHANS   (3*VPU_INT8_ACC_PERIOD - 4)
#define HEIGHT  (32)
#define WIDTH   (32)
void test_avgpool2d_global_case0()
{
    nn_image_t WORD_ALIGNED  X[HEIGHT][WIDTH][CHANS] = {{{0}}};
    nn_image_t WORD_ALIGNED  Y[CHANS];

    PRINTF("%s...\n", __func__);


    nn_image_params_t x_params = { HEIGHT, WIDTH, CHANS };
    nn_image_params_t y_params = { 1, 1, CHANS };

    nn_avgpool2d_global_plan_t plan;
    nn_avgpool2d_global_job_t job;

    avgpool2d_global_init(&plan, &job, &x_params, NULL, 1);

    int8_t scale;
    uint16_t shift;
    compute_scale_shift(&scale, &shift, &x_params);

    
    int32_t bias = ((int)0x807FFFFF) >> (24-shift);

    memset(X, 0, sizeof(X));

    avgpool2d_global((nn_image_t*) Y, (nn_image_t*) X, bias, scale, shift, &plan, &job);

    char str_buff[200] = {0};

    PRINTF("\t\tChecking...\n");

    for(unsigned chn = 0; chn < y_params.channels; chn++){
        nn_image_t y_exp = NEG_SAT_VAL;
        nn_image_t y = Y[chn];
        TEST_ASSERT_EQUAL(y_exp, y);
    }
}
#undef WIDTH
#undef HEIGHT
#undef CHANS


#define MAX_CHANS   (4*VPU_INT8_ACC_PERIOD)
#define MAX_HEIGHT  (32)
#define MAX_WIDTH   (32)
void test_avgpool2d_global_case1()
{
    nn_image_t WORD_ALIGNED  X[MAX_HEIGHT][MAX_WIDTH][MAX_CHANS] = {{{0}}};
    nn_image_t WORD_ALIGNED  Y[MAX_CHANS];

    PRINTF("%s...\n", __func__);

    
    typedef struct {
        uint32_t height;    
        uint32_t width;
        uint32_t channels;
        int32_t bias;
    } test_case_t;

    const test_case_t casses[] = {
        //  X               //Chans     //Bias
        {   1,  1,          16,         0          },  // 0
        {   2,  2,          16,         0          },  
        {   4,  4,          16,         0          },
        {   6,  6,          16,         0          },  
        {   8,  8,          16,         0          },
        {   4,  2,          16,         0          },  // 5
        {   8,  6,          16,         0          },
        {   2, 16,          16,         0          },
        {  32, 16,          16,         0          },
        {  32, 32,          16,         0          },
        {   2,  2,          32,         0          },  // 10
        {   2,  2,          48,         0          },
        {   2,  2,          64,         0          },
        {   2,  2,           4,         0          },
        {   2,  2,           8,         0          },
        {   2,  2,          12,         0          },  // 15
        {   2,  2,          20,         0          },
        {   2,  2,          24,         0          },
        {   2,  2,          28,         0          },
        {   2,  2,          36,         0          },
        {   4,  8,          40,         0          },  // 20
        {  12,  6,          12,         0          },
        {  16,  2,          40,         0          },
        {   4, 24,          36,         0          },
        {  32, 32,          60,         0          },
        {   2,  2,          12,         1          },  // 25
        {   2,  2,          20,       -10          },
        {   2,  2,          24,        22          },
        {   2,  2,          28,       -13          },
        {   2,  2,          36,        40          },
        {   4,  8,          40,       -77          },  // 30
        {  12,  6,          12,         3          },
        {  16,  2,          40,       -20          },
        {   4, 24,          36,        44          },
        {  32, 32,          60,      -100          },
    };

    const unsigned N_casses = sizeof(casses)/sizeof(test_case_t);
    const unsigned start_case =  0;
    const unsigned stop_case  = -1;

    print_warns(start_case);

    memset(X, 120, sizeof(X));

    for(unsigned v = start_case; v < N_casses && v < stop_case; v++){
        const test_case_t* casse = (const test_case_t*) &casses[v];

        PRINTF("\ttest vector %u...\n", v);
            
        nn_image_params_t x_params = { casse->height, casse->width, casse->channels };
        nn_image_params_t y_params = { 1, 1, casse->channels };

        nn_avgpool2d_global_plan_t plan;
        nn_avgpool2d_global_job_t job;

        for(int r = 0; r < x_params.height; r++){
            for(int c = 0; c < x_params.width; c++){
                for(int ch = 0; ch < x_params.channels; ch++){
                    ((int8_t*)X)[IMG_ADDRESS_VECT(&x_params, r, c, ch)] = 24 + ch;
                }
            }
        }

        avgpool2d_global_init(&plan, &job, &x_params, NULL, 1);

        int8_t scale;
        uint16_t shift;
        compute_scale_shift(&scale, &shift, &x_params);

        int32_t bias = casse->bias * plan.X.pixels * scale;
        
        memset(Y, 0xCC, sizeof(Y));
        avgpool2d_global(Y, (nn_image_t*) X, bias, scale, shift, &plan, &job);

        char str_buff[200] = {0};
        PRINTF("\t\tChecking...\n");
        for(unsigned row = 0; row < y_params.height; row++){
            for(unsigned col = 0; col < y_params.width; col++){

                int32_t y_base = IMG_ADDRESS_VECT(&y_params, row, col, 0);

                for(unsigned chn = 0; chn < y_params.channels; chn++){
                    
                    int8_t y_exp = 24 + chn + casse->bias;

                    int8_t y = ((int8_t*)Y)[y_base + chn];

                    if(y != y_exp){
                        sprintf(str_buff, "(row, col, chn) = (%u, %u, %u)", row, col, chn);
                    }

                    TEST_ASSERT_EQUAL_MESSAGE(y_exp, y, str_buff);
                }
            }
        }

    }

}
#undef MAX_WIDTH
#undef MAX_HEIGHT
#undef MAX_CHANS



void test_avgpool2d_global()
{
    UNITY_SET_FILE();

    RUN_TEST(test_avgpool2d_global_case0);
    RUN_TEST(test_avgpool2d_global_case1);
}
