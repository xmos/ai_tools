
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>


#include "tst_common.h"

#include "nn_operator.h"
#include "xs3_vpu.h"

#include "unity.h"


#define DO_PRINT_EXTRA ((DO_PRINT_EXTRA_GLOBAL) && 0)

#if CONFIG_SYMMETRIC_SATURATION_avgpool2d
  #define NEG_SAT_VAL   (-127)
#else
  #define NEG_SAT_VAL   (-128)
#endif 





static void Check_Y(
    const nn_image_t y_exp,
    const nn_image_t* Y,
    const nn_image_params_t* y_params,
    const unsigned row,
    const unsigned col,
    const unsigned chn,
    const unsigned line)
{
    char str_buff[200];

    nn_image_t y = Y[IMG_ADDRESS_VECT(y_params, row, col, chn)];

    if(y != y_exp){
        sprintf(str_buff, "Y[%u][%u][%u] was wrong [line %u]", 
                row, col, chn, line);
    }

    TEST_ASSERT_EQUAL_MESSAGE(y_exp, y, str_buff);
}






#define MAX_CHANS   (4*VPU_INT8_ACC_PERIOD)
#define MAX_HEIGHT  (32)
#define MAX_WIDTH   (32)
void test_avgpool2d_case1()
{
    int8_t WORD_ALIGNED  X[MAX_HEIGHT][MAX_WIDTH][MAX_CHANS] = {{{0}}};
    
    int8_t WORD_ALIGNED  Y[MAX_HEIGHT][MAX_WIDTH][MAX_CHANS];

    PRINTF("%s...\n", __func__);

    
    typedef struct {
        struct {    uint32_t height;    uint32_t width;     } X;
        struct {    uint32_t height;    uint32_t width;     } Y;
        struct {    uint32_t height;    uint32_t width;
                    int32_t vstride;    int32_t hstride;    } W;
        unsigned channels;
        unsigned line;
    } test_case_t;

    const test_case_t casses[] = {
        //  X               // Y            // W                      //Chans
        {   { 1,  1},       { 1,  1},       {  1,  1,  1,  1},        16          , __LINE__},  //0
        {   { 1,  1},       { 1,  1},       {  1,  1,  1,  1},        32          , __LINE__},
        {   { 1,  1},       { 1,  1},       {  1,  1,  1,  1},         8          , __LINE__},
        {   { 1,  1},       { 1,  1},       {  1,  1,  1,  1},        28          , __LINE__},
        {   { 1,  1},       { 1,  1},       {  1,  1,  1,  1},         4          , __LINE__},
        {   { 1,  1},       { 1,  1},       {  1,  1,  1,  1},        MAX_CHANS   , __LINE__},

        {   { 1,  2},       { 1,  1},       {  1,  2,  1,  1},        16          , __LINE__},  //6
        {   { 2,  1},       { 1,  1},       {  2,  1,  1,  1},        16          , __LINE__},
        {   { 2,  2},       { 1,  1},       {  2,  2,  1,  1},        16          , __LINE__},
        {   { 1,  4},       { 1,  1},       {  1,  4,  1,  1},        16          , __LINE__},
        {   { 4,  1},       { 1,  1},       {  4,  1,  1,  1},        16          , __LINE__},
        {   { 4,  4},       { 1,  1},       {  4,  4,  1,  1},        16          , __LINE__},

        {   { 1,  3},       { 1,  1},       {  1,  3,  1,  1},        16          , __LINE__},  //12
        {   { 3,  1},       { 1,  1},       {  3,  1,  1,  1},        16          , __LINE__},
        {   { 3,  3},       { 1,  1},       {  3,  3,  1,  1},        16          , __LINE__},
        {   { 5,  3},       { 1,  1},       {  5,  3,  1,  1},        16          , __LINE__},  
        {   { 9,  1},       { 1,  1},       {  9,  1,  1,  1},        16          , __LINE__},
        {   { 3, 13},       { 1,  1},       {  3, 13,  1,  1},        16          , __LINE__},

        {   { 1,  2},       { 1,  2},       {  1,  1,  1,  1},        16          , __LINE__},  //18
        {   { 2,  1},       { 2,  1},       {  1,  1,  1,  1},        16          , __LINE__},
        {   { 2,  2},       { 2,  2},       {  1,  1,  1,  1},        16          , __LINE__},
        {   { 1,  3},       { 1,  3},       {  1,  1,  1,  1},        16          , __LINE__},
        {   { 3,  3},       { 3,  3},       {  1,  1,  1,  1},        16          , __LINE__},
        {   { 4,  1},       { 4,  1},       {  1,  1,  1,  1},        16          , __LINE__},
        {   { 5,  7},       { 5,  7},       {  1,  1,  1,  1},        16          , __LINE__},

        {   { 1,  1},       { 1,  1},       {  1,  1,  2,  2},        16          , __LINE__},  //25
        {   { 4,  2},       { 2,  2},       {  1,  1,  2,  1},        16          , __LINE__},
        {   { 2,  4},       { 2,  2},       {  1,  1,  1,  2},        16          , __LINE__},
        {   { 4,  4},       { 2,  2},       {  1,  1,  2,  2},        16          , __LINE__},
        {   { 9,  9},       { 3,  3},       {  1,  1,  3,  3},        16          , __LINE__},

        {   { 4,  4},       { 2,  2},       {  2,  2,  1,  2},        16          , __LINE__},  //30
        {   { 4,  4},       { 3,  3},       {  2,  2,  1,  1},        16          , __LINE__},
        {   { 9,  9},       { 3,  3},       {  3,  3,  3,  3},        16          , __LINE__},
        {   { 9,  9},       { 3,  3},       {  3,  3,  3,  3},        32          , __LINE__},
        {   {16, 16},       { 4,  4},       {  4,  4,  4,  4},        MAX_CHANS   , __LINE__},
        {   {25, 25},       { 5,  5},       {  5,  5,  5,  5},         8          , __LINE__},
        {   {32, 32},       { 4,  8},       {  8,  4,  8,  4},        24          , __LINE__},

    };

    const unsigned N_casses = sizeof(casses)/sizeof(test_case_t);
    const unsigned start_case =  0;
    const unsigned stop_case  = -1;

    print_warns(start_case);

    memset(X, 120, sizeof(X));

    PRINTF("\t\tSetting X...\n");
    for(int r = 0; r < MAX_HEIGHT; r++){
        for(int c = 0; c < MAX_WIDTH; c++){
            for(int ch = 0; ch < MAX_CHANS; ch++){
                X[r][c][ch] = (ch&1)? 120 : -120;
            }
        }
    }


    for(unsigned v = start_case; v < N_casses && v < stop_case; v++){
        const test_case_t* casse = (const test_case_t*) &casses[v];

        PRINTF("\ttest vector %u...\n", v);
            
        nn_image_params_t x_params = { casse->X.height, casse->X.width, casse->channels };
        nn_image_params_t y_params = { casse->Y.height, casse->Y.width, casse->channels };

        nn_window_params_t window_config;
        window_config.shape.height  = casse->W.height;
        window_config.shape.width   = casse->W.width;
        window_config.stride.vertical   = casse->W.vstride;
        window_config.stride.horizontal = casse->W.hstride;
        window_config.start.row     = 0;
        window_config.start.column  = 0;

        nn_avgpool2d_plan_t plan;
        nn_pool2d_job_t job;

        avgpool2d_init(&plan, &job, &x_params, &y_params, &window_config, NULL, 1);
        plan.impl = AVGPOOL2D_DEFAULT; // force non 2x2 implementation

        memset(Y, 0xCC, casse->Y.height * casse->Y.width * casse->channels);    //too expensive to write the whole image, so just do the part that's in play
        avgpool2d((int8_t*)Y, (int8_t*)X, &plan, &job);

        PRINTF("\t\tChecking...\n");
        for(unsigned row = 0; row < y_params.height; row++){
            for(unsigned col = 0; col < y_params.width; col++){

                int32_t y_base = IMG_ADDRESS_VECT(&y_params, row, col, 0);

                for(unsigned chn = 0; chn < y_params.channels; chn++){
                    int8_t y_exp = (chn&1)? 120 : -120;
                    Check_Y(y_exp, (nn_image_t*)Y, &y_params, row, col, chn, casse->line);
                }
            }
        }

    }

}
#undef MAX_WIDTH
#undef MAX_HEIGHT
#undef MAX_CHANS





#define CHANS       (20)
#define HEIGHT      (12)
#define WIDTH       (12)
void test_avgpool2d_case2()
{
    srand(34524666);
    int8_t WORD_ALIGNED  X[HEIGHT][WIDTH][CHANS] = {{{0}}};
    int8_t WORD_ALIGNED Y_exp[HEIGHT][WIDTH][CHANS];

    int8_t WORD_ALIGNED  Y[HEIGHT][WIDTH][CHANS];

    PRINTF("%s...\n", __func__);

    
    typedef struct {
        struct {    uint32_t height;        uint32_t width;     } window;
        struct {    uint32_t row;           uint32_t col;       } x_start;
        struct {    uint32_t row;           uint32_t col;       } y_start;
        unsigned line;
    } test_case_t;

    const test_case_t casses[] = {
        {       {   1,  1 },        {   0,  0 },        {   0,  0 }         , __LINE__},
        {       {   2,  2 },        {   0,  0 },        {   0,  0 }         , __LINE__},
        {       {   3,  3 },        {   0,  0 },        {   0,  0 }         , __LINE__},
        {       {   4,  4 },        {   0,  0 },        {   0,  0 }         , __LINE__},
        {       {   5,  5 },        {   0,  0 },        {   0,  0 }         , __LINE__},
        {       {   6,  6 },        {   0,  0 },        {   0,  0 }         , __LINE__},
        {       {   8,  8 },        {   0,  0 },        {   0,  0 }         , __LINE__},
        {       {  12, 12 },        {   0,  0 },        {   0,  0 }         , __LINE__},
        {       {   1,  2 },        {   0,  0 },        {   0,  0 }         , __LINE__},
        {       {   2,  1 },        {   0,  0 },        {   0,  0 }         , __LINE__},
        {       {   3,  8 },        {   0,  0 },        {   0,  0 }         , __LINE__},
        {       {  12,  4 },        {   0,  0 },        {   0,  0 }         , __LINE__},
        {       {   1,  1 },        {   1,  1 },        {   0,  0 }         , __LINE__},
        {       {   1,  1 },        {   0,  0 },        {   1,  1 }         , __LINE__},
        {       {   2,  2 },        {   4,  4 },        {   0,  0 }         , __LINE__},
    };

    const unsigned N_casses = sizeof(casses)/sizeof(test_case_t);
    const unsigned start_case =  0;
    const unsigned stop_case  = -1;

    print_warns(start_case);

    for(unsigned v = start_case; v < N_casses && v < stop_case; v++){
        const test_case_t* casse = (const test_case_t*) &casses[v];

        PRINTF("\ttest vector %u...\n", v);
            
        nn_image_params_t x_params = { HEIGHT, WIDTH, CHANS };
        nn_image_params_t y_params = { HEIGHT, WIDTH, CHANS };
        
        nn_window_params_t window_config;
        window_config.shape.height  = casse->window.height;
        window_config.shape.width   = casse->window.width;
        window_config.stride.vertical   = casse->window.height;
        window_config.stride.horizontal = casse->window.width;
        window_config.start.row     = casse->x_start.row;
        window_config.start.column  = casse->x_start.col;

        nn_avgpool2d_plan_t plan;
        nn_pool2d_job_t job;
        nn_window_op_job_params_t job_params;

        job_params.start.rows = casse->y_start.row;
        job_params.start.cols = casse->y_start.col;
        job_params.start.channels = 0;

        job_params.size.rows = (HEIGHT - casse->x_start.row) / casse->window.height;
        job_params.size.cols = (WIDTH - casse->x_start.col) / casse->window.width;
        job_params.size.channels = CHANS;

        if((HEIGHT - casse->y_start.row) < job_params.size.rows) 
            job_params.size.rows = (HEIGHT - casse->y_start.row);
        if((WIDTH  - casse->y_start.col) < job_params.size.cols) 
            job_params.size.cols = (WIDTH  - casse->y_start.col);

        avgpool2d_init(&plan, &job, &x_params, &y_params, &window_config, &job_params, 1);
        plan.impl = AVGPOOL2D_DEFAULT; // force non 2x2 implementation

        PRINTF("\t\tSetting X...\n");
        memset(Y_exp, 0xCC, sizeof(Y_exp));
        for(int vpos = 0; vpos < job_params.size.rows; vpos++){
            for(int hpos = 0; hpos < job_params.size.cols; hpos++){

                int x_top  = casse->x_start.row + (casse->y_start.row + vpos) * window_config.stride.vertical;
                int x_left = casse->x_start.col + (casse->y_start.col + hpos) * window_config.stride.horizontal;

                for(int chn = 0; chn < y_params.channels;chn++){

                    // const unsigned pix = casse->window.height * casse->window.width;
                    // const unsigned pix_mod2 = pix & 0x01;
                    int8_t avg = pseudo_rand_uint32() & 0xFF;

                    for(int xr = 0; xr < casse->window.height; xr++){
                        for(int xc = 0; xc < casse->window.width; xc++){
                            X[x_top+xr][x_left+xc][chn] = avg;
                        }
                    }

                    Y_exp[vpos + casse->y_start.row][hpos + casse->y_start.col][chn] = avg;
                }
            }
        }


        PRINTF("\t\tRunning avgpool2d()...\n");
        memset(Y, 0xCC, sizeof(Y));    
        avgpool2d((int8_t*)Y, (int8_t*)X, &plan, &job);

        PRINTF("\t\tChecking...\n");
        for(unsigned row = 0; row < y_params.height; row++){
            for(unsigned col = 0; col < y_params.width; col++){
                for(unsigned chn = 0; chn < y_params.channels; chn++){
                    
                    int8_t y_exp = Y_exp[row][col][chn];
                    if(y_exp == -128)   y_exp = NEG_SAT_VAL;

                    Check_Y(y_exp, (nn_image_t*)Y, &y_params, row, col, chn, casse->line);
                }
            }
        }

    }

}
#undef WIDTH
#undef HEIGHT
#undef CHANS



#define MAX_CHANS   (2*VPU_INT8_ACC_PERIOD - 4)
#define MAX_HEIGHT  (1)
#define MAX_WIDTH   (1)
void test_avgpool2d_case3()
{
    int8_t WORD_ALIGNED  X[MAX_HEIGHT][MAX_WIDTH][MAX_CHANS] = {{{0}}};
    int8_t WORD_ALIGNED  Y[MAX_HEIGHT][MAX_WIDTH][MAX_CHANS];

    memset(X, 0x80, sizeof(X));
            
    nn_image_params_t x_params = { MAX_HEIGHT, MAX_WIDTH, MAX_CHANS };
    nn_image_params_t y_params = { MAX_HEIGHT, MAX_WIDTH, MAX_CHANS };

    nn_window_params_t window_config;
    nn_avgpool2d_plan_t plan;
    nn_pool2d_job_t job;
    nn_window_op_job_params_t job_params;

    window_config.shape.height  = 1;
    window_config.shape.width   = 1;
    window_config.stride.vertical   = 1;
    window_config.stride.horizontal = 1;
    window_config.start.row     = 0;
    window_config.start.column  = 0;

    job_params.start.rows     = 0;
    job_params.start.cols     = 0;
    job_params.start.channels = 0;

    job_params.size.rows     = y_params.height;
    job_params.size.cols     = y_params.width;
    job_params.size.channels = y_params.channels;

    avgpool2d_init(&plan, &job, &x_params, &y_params, &window_config, &job_params, 1);
    plan.impl = AVGPOOL2D_DEFAULT; // force non 2x2 implementation

    avgpool2d((int8_t*)Y, (int8_t*)X, &plan, &job);

    for(unsigned chn = 0; chn < y_params.channels; chn++){
        
        int8_t y_exp = NEG_SAT_VAL;

        Check_Y(y_exp, (nn_image_t*)Y, &y_params, 0, 0, chn, 0);
    }

}
#undef MAX_WIDTH
#undef MAX_HEIGHT
#undef MAX_CHANS






#define MAX_CHANS   (4*VPU_INT8_ACC_PERIOD)
#define MAX_HEIGHT  (32)
#define MAX_WIDTH   (32)
void test_avgpool2d_2x2_case1()
{
    int8_t WORD_ALIGNED  X[MAX_HEIGHT][MAX_WIDTH][MAX_CHANS] = {{{0}}};
    int8_t WORD_ALIGNED  Y[MAX_HEIGHT][MAX_WIDTH][MAX_CHANS];

    PRINTF("%s...\n", __func__);

    
    typedef struct {
        uint32_t height;    
        uint32_t width;
        uint32_t channels;
        unsigned line;
    } test_case_t;

    const test_case_t casses[] = {
        //  X               //Chans
        // {   2,  2,          16          , __LINE__},  // 0
        {   4,  4,          16          , __LINE__},
        {   6,  6,          16          , __LINE__},
        {   8,  8,          16          , __LINE__},
        {   2,  4,          16          , __LINE__},
        {   4,  2,          16          , __LINE__},  // 5
        {   8,  6,          16          , __LINE__},
        {   2, 16,          16          , __LINE__},
        {  32, 16,          16          , __LINE__},
        {  32, 32,          16          , __LINE__},
        
        {   2,  2,          32          , __LINE__},  // 10
        {   2,  2,          48          , __LINE__},
        {   2,  2,          64          , __LINE__},
        {   2,  2,           4          , __LINE__},
        {   2,  2,           8          , __LINE__},
        {   2,  2,          12          , __LINE__},  // 15
        {   2,  2,          20          , __LINE__},
        {   2,  2,          24          , __LINE__},
        {   2,  2,          28          , __LINE__},
        {   2,  2,          36          , __LINE__},

        {   4,  8,          40          , __LINE__},  // 20
        {  12,  6,          12          , __LINE__},
        {  16,  2,          40          , __LINE__},
        {   4, 24,          36          , __LINE__},
        {  32, 32,          60          , __LINE__},
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
        nn_image_params_t y_params = { casse->height/2, casse->width/2, casse->channels };

        for(int r = 0; r < y_params.height; r++){
            for(int c = 0; c < y_params.width; c++){
                for(int ch = 0; ch < y_params.channels; ch++){
                    ((int8_t*)X)[IMG_ADDRESS_VECT(&x_params, 2*r+0, 2*c+0, ch)] = (ch&1)? 110 : -110;
                    ((int8_t*)X)[IMG_ADDRESS_VECT(&x_params, 2*r+1, 2*c+0, ch)] = (ch&1)?  90 : -90;
                    ((int8_t*)X)[IMG_ADDRESS_VECT(&x_params, 2*r+0, 2*c+1, ch)] = (ch&1)? 120 : -120;
                    ((int8_t*)X)[IMG_ADDRESS_VECT(&x_params, 2*r+1, 2*c+1, ch)] = (ch&1)?  80 : -80;
                }
            }
        }

        nn_window_params_t window_config;
        nn_avgpool2d_plan_t plan;
        nn_pool2d_job_t job;
        nn_window_op_job_params_t job_params;

        window_config.shape.height  = 2;
        window_config.shape.width   = 2;
        window_config.stride.vertical   = 2;
        window_config.stride.horizontal = 2;
        window_config.start.row     = 0;
        window_config.start.column  = 0;

        job_params.start.rows     = 0;
        job_params.start.cols     = 0;
        job_params.start.channels = 0;

        job_params.size.rows     = y_params.height;
        job_params.size.cols     = y_params.width;
        job_params.size.channels = y_params.channels;

        avgpool2d_init(&plan, &job, &x_params, &y_params, &window_config, &job_params, 1);
        plan.impl = AVGPOOL2D_2X2; // force non 2x2 implementation

        memset(Y, 0xCC, casse->height * casse->width * casse->channels / 4);
        avgpool2d((int8_t*)Y, (int8_t*)X, &plan, &job);

        PRINTF("\t\tChecking...\n");
        for(unsigned row = 0; row < y_params.height; row++){
            for(unsigned col = 0; col < y_params.width; col++){
                for(unsigned chn = 0; chn < y_params.channels; chn++){
                    
                    int8_t y_exp = (chn&1)? 100 : -100;

                    Check_Y(y_exp, (nn_image_t*)Y, &y_params, row, col, chn, casse->line);
                }
            }
        }

    }

}
#undef MAX_WIDTH
#undef MAX_HEIGHT
#undef MAX_CHANS




#define CHANS       (3*VPU_INT8_ACC_PERIOD - 4)
#define X_HEIGHT    (2)
#define X_WIDTH     (2)
#define Y_HEIGHT    (1)
#define Y_WIDTH     (1)
void test_avgpool2d_2x2_case2()
{
    int8_t WORD_ALIGNED  X[X_HEIGHT][X_WIDTH][CHANS] = {{{0}}};
    
    int8_t WORD_ALIGNED  Y[Y_HEIGHT][Y_WIDTH][CHANS];

    memset(X, 0x80, sizeof(X));
            
    nn_image_params_t x_params = { X_HEIGHT, X_WIDTH, CHANS };
    nn_image_params_t y_params = { Y_HEIGHT, Y_WIDTH, CHANS };

    nn_window_params_t window_config;
    nn_avgpool2d_plan_t plan;
    nn_pool2d_job_t job;
    nn_window_op_job_params_t job_params;

    window_config.shape.height  = 2;
    window_config.shape.width   = 2;
    window_config.stride.vertical   = 2;
    window_config.stride.horizontal = 2;
    window_config.start.row     = 0;
    window_config.start.column  = 0;

    avgpool2d_init(&plan, &job, &x_params, &y_params, &window_config, NULL, 1);
    plan.impl = AVGPOOL2D_2X2; // force non 2x2 implementation
    
    avgpool2d((int8_t*)Y, (int8_t*)X, &plan, &job);

    for(unsigned row = 0; row < y_params.height; row++){
        for(unsigned col = 0; col < y_params.width; col++){
            for(unsigned chn = 0; chn < y_params.channels; chn++){
                Check_Y(NEG_SAT_VAL, (nn_image_t*)Y, &y_params, row, col, chn, 0);
            }
        }
    }
}
#undef Y_WIDTH
#undef Y_HEIGHT
#undef X_WIDTH
#undef X_HEIGHT
#undef CHANS




#define CHANS       (3*VPU_INT8_ACC_PERIOD - 4)
#define X_HEIGHT    (4)
#define X_WIDTH     (4)
#define Y_HEIGHT    (2)
#define Y_WIDTH     (2)
void test_avgpool2d_2x2_case3()
{
    int8_t WORD_ALIGNED  X[X_HEIGHT][X_WIDTH][CHANS] = {{{0}}};
    int8_t WORD_ALIGNED  Y[Y_HEIGHT][Y_WIDTH][CHANS];

    memset(X, 0x80, sizeof(X));

    nn_image_params_t x_params = { X_HEIGHT, X_WIDTH, CHANS };
    nn_image_params_t y_params = { Y_HEIGHT, Y_WIDTH, CHANS };
    
    nn_window_params_t window_config;
    nn_avgpool2d_plan_t plan;
    nn_pool2d_job_t job;
    nn_window_op_job_params_t job_params;

    window_config.shape.height  = 2;
    window_config.shape.width   = 2;
    window_config.stride.vertical   = 2;
    window_config.stride.horizontal = 2;
    window_config.start.row     = 0;
    window_config.start.column  = 0;

    avgpool2d_init(&plan, &job, &x_params, &y_params, &window_config, NULL, 1);
    plan.impl = AVGPOOL2D_2X2; // force non 2x2 implementation

    avgpool2d((int8_t*)Y, (int8_t*)X, &plan, &job);

    for(unsigned chn = 0; chn < y_params.channels; chn++){
        int8_t y_exp = NEG_SAT_VAL;
        Check_Y(y_exp, (nn_image_t*)Y, &y_params, 0, 0, chn, 0);
    }

}
#undef Y_WIDTH
#undef Y_HEIGHT
#undef X_WIDTH
#undef X_HEIGHT
#undef CHANS


void test_avgpool2d()
{
    UNITY_SET_FILE();
    
    RUN_TEST(test_avgpool2d_case1);
    RUN_TEST(test_avgpool2d_case2);
    RUN_TEST(test_avgpool2d_case3);
    
    RUN_TEST(test_avgpool2d_2x2_case1);
    RUN_TEST(test_avgpool2d_2x2_case2);
    RUN_TEST(test_avgpool2d_2x2_case3);

}