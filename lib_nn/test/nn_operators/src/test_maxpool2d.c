
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <syscall.h>

#include "tst_common.h"

#include "nn_operator.h"
#include "xs3_vpu.h"

#include "unity.h"


#define DO_PRINT_EXTRA ((DO_PRINT_EXTRA_GLOBAL) && 0)



#define DEBUG_ON    (0 || TEST_DEBUG_ON)
#define MAX_CHANS   (2*VPU_INT8_EPV)
#define MAX_HEIGHT  (32)
#define MAX_WIDTH   (32)
#define REPS        (3)
void test_maxpool2d_case1()
{
    unsigned seed = 66535;

    int8_t WORD_ALIGNED  X[MAX_HEIGHT][MAX_WIDTH][MAX_CHANS] = {{{0}}};
    
    int8_t WORD_ALIGNED  Y[MAX_HEIGHT][MAX_WIDTH][MAX_CHANS];

    PRINTF("%s...\n", __func__);

    
    typedef struct {
        struct {    uint32_t height;    uint32_t width;     } X;
        struct {    uint32_t height;    uint32_t width;     } Y;
        struct {    uint32_t height;    uint32_t width;
                    int32_t vstride;    int32_t hstride;    } W;
        unsigned channels;
    } test_case_t;

    const test_case_t casses[] = {
        //  X               // Y            // W                      //Chans
        {   { 1,  1},       { 1,  1},       {  1,  1,  1,  1},        32          },  //0
        {   { 1,  1},       { 1,  1},       {  1,  1,  1,  1},        16          },
        {   { 1,  1},       { 1,  1},       {  1,  1,  1,  1},         8          },
        {   { 1,  1},       { 1,  1},       {  1,  1,  1,  1},        48          },
        {   { 1,  1},       { 1,  1},       {  1,  1,  1,  1},         4          },
        {   { 1,  1},       { 1,  1},       {  1,  1,  1,  1},        MAX_CHANS   },
        
        {   { 1,  2},       { 1,  1},       {  1,  2,  1,  1},        32          },  //6
        {   { 2,  1},       { 1,  1},       {  2,  1,  1,  1},        32          },
        {   { 2,  2},       { 1,  1},       {  2,  2,  1,  1},        32          },
        {   { 1,  4},       { 1,  1},       {  1,  4,  1,  1},        32          },
        {   { 4,  1},       { 1,  1},       {  4,  1,  1,  1},        32          },
        {   { 4,  4},       { 1,  1},       {  4,  4,  1,  1},        32          },

        {   { 1,  3},       { 1,  1},       {  1,  3,  1,  1},        32          },  //12
        {   { 3,  1},       { 1,  1},       {  3,  1,  1,  1},        32          },
        {   { 3,  3},       { 1,  1},       {  3,  3,  1,  1},        32          },
        {   { 5,  3},       { 1,  1},       {  5,  3,  1,  1},        32          },  
        {   { 9,  1},       { 1,  1},       {  9,  1,  1,  1},        32          },
        {   { 3, 13},       { 1,  1},       {  3, 13,  1,  1},        32          },

        {   { 1,  2},       { 1,  2},       {  1,  1,  1,  1},        32          },  //18
        {   { 2,  1},       { 2,  1},       {  1,  1,  1,  1},        32          },
        {   { 2,  2},       { 2,  2},       {  1,  1,  1,  1},        32          },
        {   { 1,  3},       { 1,  3},       {  1,  1,  1,  1},        32          },
        {   { 3,  3},       { 3,  3},       {  1,  1,  1,  1},        32          },
        {   { 4,  1},       { 4,  1},       {  1,  1,  1,  1},        32          },
        {   { 5,  7},       { 5,  7},       {  1,  1,  1,  1},        32          },
        
        {   { 1,  1},       { 1,  1},       {  1,  1,  2,  2},        32          },  //25
        {   { 4,  2},       { 2,  2},       {  1,  1,  2,  1},        32          },
        {   { 2,  4},       { 2,  2},       {  1,  1,  1,  2},        32          },
        {   { 4,  4},       { 2,  2},       {  1,  1,  2,  2},        32          },
        {   { 9,  9},       { 3,  3},       {  1,  1,  3,  3},        32          },
        
        {   { 4,  4},       { 2,  2},       {  2,  2,  2,  2},        32          },  //30
        {   { 4,  4},       { 3,  3},       {  2,  2,  1,  1},        32          },
        {   { 9,  9},       { 3,  3},       {  3,  3,  3,  3},        32          },
        {   { 9,  9},       { 3,  3},       {  3,  3,  3,  3},        32          },
        {   {16, 16},       { 4,  4},       {  4,  4,  4,  4},        MAX_CHANS   },
        {   {25, 25},       { 5,  5},       {  5,  5,  5,  5},         8          },
        {   {32, 32},       { 4,  8},       {  8,  4,  8,  4},        24          },

    };

    const unsigned N_casses = sizeof(casses)/sizeof(test_case_t);
    const unsigned start_case =  0;
    const unsigned stop_case  = -1;

    print_warns(start_case, 1, 1);

    for(unsigned v = start_case; v < N_casses && v < stop_case; v++){
        const test_case_t* casse = (const test_case_t*) &casses[v];

        PRINTF("\ttest vector %u...\n", v);
            
        nn_image_params_t x_params = { casse->X.height, casse->X.width, casse->channels };
        nn_image_params_t y_params = { casse->Y.height, casse->Y.width, casse->channels };

        nn_window_op_config_t config;
        nn_window_op_config_simple(&config, &x_params, &y_params, 
                                 casse->W.height,  casse->W.width,
                                 casse->W.vstride, casse->W.hstride);

        nn_window_op_plan_t plan;

        maxpool2d_init(&plan, &x_params, &y_params, &config);

#if (DEBUG_ON || 0)
        PRINTF("params.out_rows        = %d\n",     params.out_rows);
        PRINTF("params.out_cols        = %d\n",     params.out_cols);   
        PRINTF("params.out_chans       = %d\n",     params.out_chans);
        PRINTF("params.pool_rows       = %d\n",     params.pool_rows);
        PRINTF("params.pool_cols       = %d\n",     params.pool_cols);
        PRINTF("params.pool_col_incr_x = %d\n",     params.pool_col_incr_x);
        PRINTF("params.pool_row_incr_x = %d\n",     params.pool_row_incr_x);
        PRINTF("params.hstride_incr_x  = %d\n",     params.hstride_incr_x);
        PRINTF("params.hstride_incr_y  = %d\n",     params.hstride_incr_y);
        PRINTF("params.vstride_incr_x  = %d\n",     params.vstride_incr_x);
        PRINTF("params.vstride_incr_y  = %d\n",     params.vstride_incr_y);
        PRINTF("params.chan_grp_incr_x = %d\n",     params.chan_grp_incr_x);
        PRINTF("params.chan_grp_incr_y = %d\n",     params.chan_grp_incr_y);
        PRINTF("params.start_offset_x    = %d\n",   params.start_offset_x);
        PRINTF("params.start_offset_y    = %d\n",   params.start_offset_y);
#endif //DEBUG_ON


        const unsigned x_bytes = x_params.height * x_params.width * x_params.channels;
        const unsigned y_bytes = y_params.height * y_params.width * y_params.channels;

        pseudo_rand_bytes(&seed, (char*)X, x_bytes);

        for(unsigned rep = 0; rep < REPS; rep++){
            PRINTF("\t\tRep %d...\n", rep);

            PRINTF("\t\t\tC...\n");
            memset(Y, 0xCC, y_bytes);    //too expensive to write the whole image, so just do the part that's in play
            maxpool2d((int8_t*)Y, (int8_t*)X, &plan);

            char str_buff[200] = {0};
            PRINTF("\t\t\tChecking...\n");
            for(unsigned row = 0; row < y_params.height; row++){
                for(unsigned col = 0; col < y_params.width; col++){

                    for(unsigned chn = 0; chn < y_params.channels; chn++){
                        
                        int8_t mx = -128;

                        for(int wr = 0; wr < config.window.shape.height; wr++){
                            for(int wc = 0; wc < config.window.shape.width; wc++){
                                int32_t x_offset = IMG_ADDRESS_VECT(&x_params, 
                                                                    (casse->W.vstride * row + wr), 
                                                                    (casse->W.hstride * col + wc), chn);
                                // PRINTF("!! %ld\n", x_offset);
                                int8_t x_val = ((int8_t*)X)[x_offset];
                                mx = (x_val > mx)? x_val : mx;
                            }   
                        }

                        int8_t y_exp = mx;
                        
                        int8_t y = ((int8_t*)Y)[IMG_ADDRESS_VECT(&y_params, row, col, chn)];

                        if(y != y_exp){
                            sprintf(str_buff, "(row, col, chn) = (%u, %u, %u)", row, col, chn);
                        }

                        TEST_ASSERT_EQUAL_MESSAGE(y_exp, y, str_buff);
                    }
                }
            }
        }

    }

}
#undef REPS
#undef MAX_WIDTH
#undef MAX_HEIGHT
#undef MAX_CHANS
#undef DEBUG_ON











#define DEBUG_ON    (0 || TEST_DEBUG_ON)
#define CHANS       (2*VPU_INT8_ACC_PERIOD)
#define X_HEIGHT    (12)
#define X_WIDTH     (12)
#define Y_HEIGHT    (6)
#define Y_WIDTH     (6)
#define REPS        (4)
void test_maxpool2d_case2()
{
    unsigned seed = 123124;

    int8_t WORD_ALIGNED  X[X_HEIGHT][X_WIDTH][CHANS] = {{{0}}};
    int8_t WORD_ALIGNED  Y_exp[Y_HEIGHT][Y_WIDTH][CHANS] = {{{0}}};
    
    int8_t WORD_ALIGNED  Y[Y_HEIGHT][Y_WIDTH][CHANS];

    PRINTF("%s...\n", __func__);

    
    typedef struct {
        // struct {        nn_image_vect_t X;  nn_image_vect_t Y;  } start;   //choose start locations randomly
        struct {        nn_image_vect_t shape;                  } output;
        struct {
            struct {    uint32_t height;    uint32_t width;     } shape;
            struct {    int32_t vertical;   int32_t horizontal; } stride;
        } window;

    } test_case_t;

    const test_case_t casses[] = {
        //    Out Shape          Win shape    stride               
        {     {{ 6, 6, 32}},     {{ 2, 2},    { 2, 2}}},  // 0
        {     {{ 3, 3, 32}},     {{ 4, 4},    { 4, 4}}},
        {     {{ 3, 3, 32}},     {{ 2, 2},    { 2, 2}}},
        {     {{ 6, 6, 16}},     {{ 2, 2},    { 2, 2}}},
        {     {{ 6, 6, 16}},     {{ 2, 2},    { 2, 2}}},
        {     {{ 6, 6, 16}},     {{ 2, 2},    { 2, 2}}},
        {     {{ 6, 6, 16}},     {{ 2, 2},    { 2, 2}}},
        {     {{ 5, 5, 32}},     {{ 2, 2},    { 2, 2}}},
        {     {{ 5, 5, 32}},     {{ 2, 2},    { 2, 2}}}, // 8 
        {     {{ 1, 2,  8}},     {{ 3, 2},    { 3, 2}}},

    };

    const unsigned N_casses = sizeof(casses)/sizeof(test_case_t);
    const unsigned start_case =  0;
    const unsigned stop_case  = -1;

    print_warns(start_case, 1, 1);

    for(unsigned v = start_case; v < N_casses && v < stop_case; v++){
        const test_case_t* casse = (const test_case_t*) &casses[v];

        PRINTF("\tTest vector %u...\n", v);

        nn_image_params_t x_params = { X_HEIGHT, X_WIDTH, CHANS };
        nn_image_params_t y_params = { Y_HEIGHT, Y_WIDTH, CHANS };


        nn_window_op_config_t window_config;

        memset(&window_config, 0, sizeof(window_config));
        window_config.output.shape.height = casse->output.shape.rows;
        window_config.output.shape.width = casse->output.shape.cols;
        window_config.output.shape.channels = casse->output.shape.channels;
        window_config.output.stride.vertical.rows = 1;
        window_config.output.stride.horizontal.cols = 1;

        window_config.window.shape.height = casse->window.shape.height;
        window_config.window.shape.width  = casse->window.shape.width;
        window_config.window.outer_stride.vertical.rows = casse->window.stride.vertical;
        window_config.window.outer_stride.horizontal.cols = casse->window.stride.horizontal;
        window_config.window.inner_stride.vertical.rows = 1;
        window_config.window.inner_stride.horizontal.cols = 1;

        for(unsigned rep = 0; rep < REPS; rep++){

            PRINTF("\t\tRep %d...\n", rep);
            
            //Choose start in X and Y randomly after the first rep
            if(rep) {
                //Have to figure out our max allowable range, of course
                const unsigned x_row_span = casse->window.stride.vertical   * (casse->output.shape.rows - 1) + casse->window.shape.height;
                const unsigned x_col_span = casse->window.stride.horizontal * (casse->output.shape.cols - 1) + casse->window.shape.width;

                const unsigned x_max_row = X_HEIGHT - x_row_span + 1; //Does not INCLUDE this row
                const unsigned x_max_col = X_WIDTH  - x_col_span + 1;

                const unsigned y_max_row = Y_HEIGHT - casse->output.shape.rows + 1;
                const unsigned y_max_col = Y_WIDTH  - casse->output.shape.cols + 1;

                const unsigned max_chan = CHANS - casse->output.shape.channels + 1;

                window_config.window.start.rows = pseudo_rand_uint16(&seed) % x_max_row;
                window_config.window.start.cols = pseudo_rand_uint16(&seed) % x_max_col;
                window_config.window.start.channels = (pseudo_rand_uint16(&seed) % max_chan) & 0xFFFFFFFC; //has to be word-aligned
                
                window_config.output.start.rows = pseudo_rand_uint16(&seed) % y_max_row;
                window_config.output.start.cols = pseudo_rand_uint16(&seed) % y_max_col;
                window_config.output.start.channels = (pseudo_rand_uint16(&seed) % max_chan) & 0xFFFFFFFC; 
            }
            
            nn_window_op_plan_t plan;

            maxpool2d_init(&plan, &x_params, &y_params, &window_config);

            memset(Y_exp, 0xCC, sizeof(Y_exp));
            memset(X, 0xAA, sizeof(X));
            PRINTF("\t\t\tSetting X...\n");
            for(int wh = 0; wh < window_config.output.shape.height; wh++) {
                for(int ww = 0; ww < window_config.output.shape.width; ww++) {
                    for(int ch = 0; ch < window_config.output.shape.channels; ch++) {
                        int8_t y_exp = wh + 2 * ww + ch;
                        unsigned yr = window_config.output.start.rows + wh;
                        unsigned yc = window_config.output.start.cols + ww;
                        unsigned yd = window_config.output.start.channels + ch;

                        Y_exp[yr][yc][yd] = y_exp;

                        for(int r = 0; r < window_config.window.shape.height; r++) {
                            for(int c = 0; c < window_config.window.shape.width; c++) {
                                unsigned xr = window_config.window.start.rows     + wh * window_config.window.outer_stride.vertical.rows   + r;
                                unsigned xc = window_config.window.start.cols     + ww * window_config.window.outer_stride.horizontal.cols + c;
                                unsigned xd = window_config.window.start.channels + ch;
                                X[xr][xc][xd] = y_exp;
                            }
                        }
                    }
                }
            }

    #if (DEBUG_ON || 0)
        PRINTF("plan.window.output.rows                 = %d\n", plan.window.output.rows              );
        PRINTF("plan.window.output.cols                 = %d\n", plan.window.output.cols              );
        PRINTF("plan.window.output.channels             = %d\n", plan.window.output.channels          );
        PRINTF("plan.window.window.rows                 = %d\n", plan.window.window.rows              );
        PRINTF("plan.window.window.cols                 = %d\n\n", plan.window.window.cols              );
        PRINTF("plan.window.start_stride.x              = %d\n", plan.window.start_stride.x           );
        PRINTF("plan.window.inner_stride.horizontal.x   = %d\n", plan.window.inner_stride.horizontal.x);
        PRINTF("plan.window.inner_stride.vertical.x     = %d\n", plan.window.inner_stride.vertical.x  );
        PRINTF("plan.window.outer_stride.horizontal.x   = %d\n", plan.window.outer_stride.horizontal.x);
        PRINTF("plan.window.outer_stride.vertical.x     = %d\n", plan.window.outer_stride.vertical.x  );
        PRINTF("plan.window.chan_grp_stride.x           = %d\n\n", plan.window.chan_grp_stride.x        );
        PRINTF("plan.window.start_stride.y              = %d\n", plan.window.start_stride.y           );
        PRINTF("plan.window.outer_stride.horizontal.y   = %d\n", plan.window.outer_stride.horizontal.y);
        PRINTF("plan.window.outer_stride.vertical.y     = %d\n", plan.window.outer_stride.vertical.y  );
        PRINTF("plan.window.chan_grp_stride.y           = %d\n\n", plan.window.chan_grp_stride.y        );
        PRINTF("plan.window.scale                       = 0x%08X\n", plan.scale);
        PRINTF("plan.shift                              = 0x%08X\n", plan.shift);
    #endif //DEBUG_ON

            PRINTF("\t\t\tC...\n");
            memset(Y, 0xCC, sizeof(Y));
            maxpool2d((int8_t*)Y, (int8_t*)X, &plan);


            unsigned hadsomething = 0;
            char str_buff[200] = {0};
            PRINTF("\t\t\tChecking...\n");
            for(unsigned row = 0; row < Y_HEIGHT; row++){
                for(unsigned col = 0; col < Y_WIDTH; col++){
                    for(unsigned y_chn = 0; y_chn < y_params.channels; y_chn++){
                        
                        int8_t y_exp = Y_exp[row][col][y_chn];
                        if(y_exp != (int8_t) 0xCC)
                            hadsomething = 1;

                        int8_t y = Y[row][col][y_chn];

                        if(y != y_exp){
                            sprintf(str_buff, "(row, col, chn) = (%u, %u, %u)", row, col, y_chn);
                        }

                        TEST_ASSERT_EQUAL_MESSAGE(y_exp, y, str_buff);
                    }
                }
            }
            
            TEST_ASSERT(hadsomething);
        }
    }

}
#undef REPS
#undef CHANS
#undef Y_WIDTH
#undef Y_HEIGHT
#undef X_WIDTH
#undef X_HEIGHT
#undef DEBUG_ON

void test_maxpool2d()
{
    UNITY_SET_FILE();
    
    RUN_TEST(test_maxpool2d_case1);
    RUN_TEST(test_maxpool2d_case2);
}